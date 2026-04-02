import argparse
import os

import joblib
import pandas as pd
from tqdm.auto import tqdm

from preprocessing_utils import eic_text_preprocessing
from preprocess_clinical_notes import tokenize_clinical_note

memory = joblib.Memory(".")


def ParallelExecutor(use_bar="tqdm", **joblib_args):
    """Utility for tqdm progress bar in joblib.Parallel"""
    all_bar_funcs = {
        "tqdm": lambda args: lambda x: tqdm(x, **args),
        "False": lambda args: iter,
        "None": lambda args: iter,
    }

    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return joblib.Parallel(n_jobs=joblib_args.get("n_jobs", 10))(
                bar_func(op_iter)
            )

        return tmp

    return aprun


def _find_regex_dir():
    """Try common locations for regex pattern files."""
    candidates = [
        os.path.join("..", "..", "data", "regex"),
        os.path.join("..", "..", "source", "regex"),
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "regex_patterns.json")):
            return d
    return candidates[0]


class Preprocessor:
    def __init__(
        self, id_list, n_jobs, xml_dir=None, json_dir=None, output_dir=None, regex_dir=None
    ):
        self.id_list = id_list
        self.n_jobs = n_jobs
        self.xml_dir = xml_dir
        self.json_dir = json_dir
        self.output_dir = output_dir or "../../data/preprocessed_data/clintra/"
        self.regex_dir = regex_dir or _find_regex_dir()

    def preprocess_clinical_trials_text(self):
        regex_path = os.path.join(self.regex_dir, "regex_patterns.json")
        exceptions_path = os.path.join(self.regex_dir, "exception_regex_patterns.json")
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        X = parallel_runner(
            joblib.delayed(eic_text_preprocessing)(
                [_id],
                regex_path=regex_path,
                exceptions_path=exceptions_path,
                xml_dir=self.xml_dir,
                json_dir=self.json_dir,
                output_path=self.output_dir,
            )
            for _id in self.id_list
        )
        results = [x for x in X if x is not None]
        if results:
            return pd.concat(results).reset_index(drop=True)
        print("Warning: no trials were successfully preprocessed.")
        return pd.DataFrame()

    def preprocess_patient_clinical_notes(self):
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        X = parallel_runner(
            joblib.delayed(tokenize_clinical_note)([_id]) for _id in self.id_list
        )
        return pd.concat(X).reset_index(drop=True)


def discover_ids(input_dir, input_type):
    """Discover trial IDs from a directory of XML or JSON files."""
    ids = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if not os.path.isfile(full):
            continue
        name, ext = os.path.splitext(f)
        if input_type == "xml" and ext == ".xml":
            ids.append(name)
        elif input_type == "json" and ext == ".json":
            ids.append(name)
        elif input_type == "auto" and ext in (".xml", ".json"):
            ids.append(name)
    return sorted(set(ids))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess clinical trial eligibility criteria into TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # From XML files (original ClinicalTrials.gov data)
  python preprocessing.py --input-dir ../../data/trials_xmls --input-type xml

  # From JSON files (output of jsonify.py, e.g. Excel-derived)
  python preprocessing.py --input-dir ../../data/trials_jsons --input-type json

  # Custom output directory and parallelism
  python preprocessing.py --input-dir ../../data/trials_jsons --input-type json \\
      --output-dir ../../data/preprocessed_data/clintra/ --n-jobs 4
""",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing trial files (XML or JSON)",
    )
    parser.add_argument(
        "--input-type",
        choices=["xml", "json", "auto"],
        default="auto",
        help="Type of input files: xml, json, or auto-detect (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/preprocessed_data/clintra/",
        help="Directory to write preprocessed TSV files (default: ../../data/preprocessed_data/clintra/)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--regex-dir",
        default=None,
        help="Directory containing regex_patterns.json and exception_regex_patterns.json. "
        "Auto-detected if not specified.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trial_ids = discover_ids(args.input_dir, args.input_type)
    if not trial_ids:
        print(f"No {args.input_type} files found in {args.input_dir}")
        raise SystemExit(1)

    print(f"Found {len(trial_ids)} trial(s) in {args.input_dir}")

    xml_dir = args.input_dir if args.input_type in ("xml", "auto") else None
    json_dir = args.input_dir if args.input_type in ("json", "auto") else None

    preprocessor = Preprocessor(
        trial_ids,
        args.n_jobs,
        xml_dir=xml_dir,
        json_dir=json_dir,
        output_dir=args.output_dir,
        regex_dir=args.regex_dir,
    )
    preprocessor.preprocess_clinical_trials_text()
    print("Done. TSVs written to:", args.output_dir)
