#!/usr/bin/env python3
"""Batch-evaluate matcher experiment outputs from a manifest or directory list.

Ground truth can be a single JSON, a recursive directory of JSONs (--ground-truth),
or a flat directory with one JSON file per patient (--truth-dir, top-level *.json only).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from evaluate_ranked_trials import (
    PatientGroundTruth,
    compute_threshold_match_stats,
    evaluate,
    load_ground_truth,
    load_predictions,
    load_predictions_above_score,
)


def _load_ground_truth_compat(
    path: Path, *, recursive_directory: bool
) -> List[PatientGroundTruth]:
    """Call load_ground_truth with recursive_directory when supported (newer evaluate_ranked_trials).

    Older copies of evaluate_ranked_trials.py omit that parameter; for a flat directory
    (--truth-dir) we load each top-level *.json separately so subfolders are not scanned.
    """
    sig = inspect.signature(load_ground_truth)
    if "recursive_directory" in sig.parameters:
        return load_ground_truth(path, recursive_directory=recursive_directory)
    if path.is_dir() and not recursive_directory:
        merged: List[PatientGroundTruth] = []
        for fp in sorted(path.glob("*.json")):
            merged.extend(load_ground_truth(fp))
        return merged
    return load_ground_truth(path)


# This file lives at scripts/evaluation/ — repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_MANIFEST = DEFAULT_RESULTS_ROOT / "matcher_experiment_manifest.json"
DEFAULT_PREDICTION_FILE = "ranked_trials.json"
DEFAULT_MIN_PRED_SCORE = 0.5
DEFAULT_EXAMPLE_EXCLUDE_STEMS = ("v2_family", "v2_cohort")


def _normalize_json_stem(name: str) -> str:
    n = str(name).strip()
    if n.lower().endswith(".json"):
        return Path(n).stem.lower()
    return n.lower()


def _discover_example_json_stems(
    example_dir: Path, exclude_names: Sequence[str]
) -> List[str]:
    """Top-level example/*.json stems, excluding cohort/family fixtures."""
    exclude_lower = {_normalize_json_stem(x) for x in exclude_names if str(x).strip()}
    stems: List[str] = []
    if not example_dir.is_dir():
        return stems
    for path in sorted(example_dir.glob("*.json")):
        stem = path.stem
        if _normalize_json_stem(stem) in exclude_lower:
            continue
        stems.append(stem)
    return stems


def _allowed_stem_match_set(stems: Sequence[str]) -> Set[str]:
    """Keys to match against GT patient_id / aliases (folder names, basenames)."""
    keys: Set[str] = set()
    for s in stems:
        sl = _normalize_json_stem(s)
        if not sl:
            continue
        keys.add(sl)
        keys.add(f"{sl}.json")
    return keys


def _gt_matches_example_stem(gt: PatientGroundTruth, allowed: Set[str]) -> bool:
    def matches_token(token: str) -> bool:
        t = str(token).strip()
        if not t:
            return False
        tl = t.lower()
        if tl in allowed:
            return True
        return _normalize_json_stem(t) in allowed

    if matches_token(gt.patient_id):
        return True
    for alias in gt.aliases:
        if matches_token(alias):
            return True
        try:
            if matches_token(Path(alias).name):
                return True
        except Exception:
            pass
    return False


def _filter_gt_by_example_json(
    gt_patients: Sequence[PatientGroundTruth],
    example_dir: Path,
    exclude_names: Sequence[str],
) -> tuple[List[PatientGroundTruth], List[str]]:
    stems = _discover_example_json_stems(example_dir, exclude_names)
    allowed = _allowed_stem_match_set(stems)
    kept = [gt for gt in gt_patients if _gt_matches_example_stem(gt, allowed)]
    return kept, stems


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "NA"


def _merge_eval_and_threshold_rows(
    per_patient: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    thr_by_id = {row["patient_id"]: row for row in threshold_rows}
    merged: List[Dict[str, Any]] = []
    for row in per_patient:
        pid = row.get("patient_id")
        extra = thr_by_id.get(pid, {})
        out = dict(row)
        for key, val in extra.items():
            if key != "patient_id":
                out[key] = val
        merged.append(out)
    return merged


def _print_per_patient_table(experiment_name: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n--- Per-patient metrics: {experiment_name} ---")
    header = (
        f"{'patient_id':36} {'gt_rel':>7} {'pred':>6} {'mtch':>5} "
        f"{'R@3':>8} {'R@5':>8} {'MRR@3':>8} {'MRR@5':>8} "
        f"{'>thr':>5} {'m@thr':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row.get('patient_id', ''))[:36]:36} "
            f"{int(row.get('ground_truth_relevant_count', 0)):>7} "
            f"{int(row.get('predicted_count', 0)):>6} "
            f"{int(row.get('matched_predicted_count', 0)):>5} "
            f"{_fmt_float(row.get('overall_recall@3')):>8} "
            f"{_fmt_float(row.get('overall_recall@5')):>8} "
            f"{_fmt_float(row.get('mrr@3')):>8} "
            f"{_fmt_float(row.get('mrr@5')):>8} "
            f"{int(row.get('predicted_count_above_threshold', 0)):>5} "
            f"{int(row.get('matched_predicted_count_above_threshold', 0)):>5}"
        )


def _load_manifest_dirs(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("experiments", [])
    result: List[Dict[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        output_dir = str(item.get("output_dir", "")).strip()
        if name and output_dir:
            result.append({"name": name, "output_dir": output_dir})
    return result


def _resolve_patient_json(root: Path, patient_file: str) -> Path:
    direct = root / patient_file
    if direct.exists():
        return direct
    return root / Path(patient_file).stem / patient_file


def _iter_prediction_roots(
    manifest: str, results_root: Path, dirs: Sequence[str]
) -> List[Dict[str, str]]:
    if dirs:
        return [
            {"name": dir_name, "output_dir": str((results_root / dir_name).resolve())}
            for dir_name in dirs
        ]
    manifest_path = Path(manifest).resolve() if str(manifest).strip() else None
    if manifest_path is not None and manifest_path.is_file():
        return _load_manifest_dirs(manifest_path)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple matcher output directories in one pass."
    )
    parser.add_argument(
        "--ground-truth",
        default="",
        help=(
            "Ground-truth JSON file, or a directory of JSON files (recursive). "
            "Omit if you pass --truth-dir."
        ),
    )
    parser.add_argument(
        "--truth-dir",
        default="",
        help=(
            "Directory of per-patient ground-truth JSON files: only immediate *.json "
            "(not subfolders). Each file is one object with patient_id or id and "
            "matched_trials (NCT/CTR trial ids). Overrides --ground-truth when set."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=(
            "Experiment manifest JSON from scripts/run_pred/run_matcher_experiments.py "
            f"(default: {DEFAULT_MANIFEST})"
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Root directory for experiment outputs (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help=(
            "Subdirectory names under --results-root (each holds per-patient prediction folders). "
            "If omitted, defaults to the stem of each top-level *.json under --example-dir "
            f"(same rules as --example-exclude; default excludes: {' '.join(DEFAULT_EXAMPLE_EXCLUDE_STEMS)}). "
            "If that list is empty, falls back to --manifest when the file exists."
        ),
    )
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help=(
            "When set, do not default --dirs from --example-dir; use --manifest if --dirs is empty."
        ),
    )
    parser.add_argument(
        "--prediction-file",
        default=DEFAULT_PREDICTION_FILE,
        help=f"Prediction filename under each patient folder (default: {DEFAULT_PREDICTION_FILE})",
    )
    parser.add_argument(
        "--patient-file",
        default="",
        help="Optional patient JSON filename for single-patient GT workflows.",
    )
    parser.add_argument(
        "--min-pred-score",
        type=float,
        default=DEFAULT_MIN_PRED_SCORE,
        help=f"Threshold for matched-rate summary (default: {DEFAULT_MIN_PRED_SCORE})",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--example-json-only",
        action="store_true",
        help=(
            "Restrict evaluation to GT patients that align with a top-level *.json "
            "under --example-dir (see --example-exclude)."
        ),
    )
    parser.add_argument(
        "--example-dir",
        default=str(REPO_ROOT / "example"),
        help=(
            "Directory of example phenopacket *.json files. Used for --example-json-only and "
            "as the default for --dirs (patient folder names = each file's stem) when --dirs is omitted."
        ),
    )
    parser.add_argument(
        "--example-exclude",
        nargs="*",
        default=list(DEFAULT_EXAMPLE_EXCLUDE_STEMS),
        metavar="STEM_OR_NAME",
        help=(
            "Basenames or stems omitted from the example scan "
            f"(default: {' '.join(DEFAULT_EXAMPLE_EXCLUDE_STEMS)})."
        ),
    )
    args = parser.parse_args()

    truth_dir_raw = str(args.truth_dir).strip()
    gt_raw = str(args.ground_truth).strip()
    if truth_dir_raw:
        gt_path = Path(truth_dir_raw).resolve()
        if not gt_path.is_dir():
            print(
                f"error: --truth-dir must be an existing directory: {gt_path}",
                file=sys.stderr,
            )
            return 2
        truth_json_files = sorted(gt_path.glob("*.json"))
        if not truth_json_files:
            print(
                f"error: --truth-dir has no top-level *.json ground-truth files: {gt_path}",
                file=sys.stderr,
            )
            return 2
        gt_patients = _load_ground_truth_compat(gt_path, recursive_directory=False)
        if not gt_patients:
            print(
                "error: --truth-dir JSON files did not yield any patients "
                "(each file needs a top-level object with patient_id or id).",
                file=sys.stderr,
            )
            return 2
        gt_source_meta = {
            "mode": "truth_dir",
            "path": str(gt_path),
            "ground_truth_json_files": [p.name for p in truth_json_files],
        }
        print(
            f"[truth-dir] {gt_path} — loaded {len(gt_patients)} patient(s) from "
            f"{len(truth_json_files)} file(s): {', '.join(p.name for p in truth_json_files)}",
            file=sys.stderr,
        )
    elif gt_raw:
        gt_path = Path(gt_raw).resolve()
        gt_patients = _load_ground_truth_compat(gt_path, recursive_directory=True)
        gt_source_meta = {"mode": "ground_truth", "path": str(gt_path)}
    else:
        print(
            "error: provide --truth-dir (per-patient JSONs in one folder) or --ground-truth.",
            file=sys.stderr,
        )
        return 2

    results_root = Path(args.results_root).resolve()
    example_filter_meta: Optional[Dict[str, Any]] = None
    if args.example_json_only:
        ex_dir = Path(args.example_dir).resolve()
        filtered, stems = _filter_gt_by_example_json(
            gt_patients, ex_dir, args.example_exclude
        )
        example_filter_meta = {
            "example_dir": str(ex_dir),
            "included_stems": stems,
            "exclude": list(args.example_exclude),
            "ground_truth_patients_before": len(gt_patients),
            "ground_truth_patients_after": len(filtered),
        }
        gt_patients = filtered
        if not gt_patients:
            print(
                "error: --example-json-only removed all ground-truth patients "
                "(check patient_id / patient_file vs example/*.json stems).",
                file=sys.stderr,
            )
            return 2
        print(
            f"[example-json-only] using {len(gt_patients)} GT patient(s) "
            f"matching stems: {', '.join(stems)}",
            file=sys.stderr,
        )

    ex_dir_resolved = Path(args.example_dir).resolve()
    dirs_eff: List[str] = list(args.dirs)
    if not dirs_eff and not args.from_manifest:
        dirs_eff = _discover_example_json_stems(ex_dir_resolved, args.example_exclude)
        if dirs_eff:
            print(
                f"[dirs] default under {results_root}: {', '.join(dirs_eff)} "
                f"(from {ex_dir_resolved} top-level *.json stems)",
                file=sys.stderr,
            )
    prediction_roots = _iter_prediction_roots(
        manifest=args.manifest,
        results_root=results_root,
        dirs=dirs_eff,
    )
    if not prediction_roots:
        print(
            "error: no experiment roots to evaluate "
            "(pass --dirs <name> under --results-root, or a valid --manifest file).",
            file=sys.stderr,
        )
        return 2

    results: List[Dict[str, Any]] = []
    for item in prediction_roots:
        name = item["name"]
        root = Path(item["output_dir"]).resolve()
        entry: Dict[str, Any] = {"name": name, "predictions_dir": str(root)}

        if not root.exists():
            entry["status"] = "missing_dir"
            results.append(entry)
            continue

        prediction_source = root
        if args.patient_file:
            prediction_source = _resolve_patient_json(root, args.patient_file)
            if not prediction_source.exists():
                entry["status"] = "missing_patient_prediction"
                entry["prediction_source"] = str(prediction_source)
                results.append(entry)
                continue

        pred_map = load_predictions(prediction_source, args.prediction_file)
        summary, per_patient, missing = evaluate(gt_patients, pred_map)
        pred_map_above = load_predictions_above_score(
            prediction_source, args.prediction_file, args.min_pred_score
        )
        threshold_summary, thr_per_patient, thr_missing = compute_threshold_match_stats(
            gt_patients, pred_map_above
        )
        per_patient_full = _merge_eval_and_threshold_rows(per_patient, thr_per_patient)

        entry["status"] = "ok"
        entry["prediction_source"] = str(prediction_source)
        entry["summary"] = summary
        entry["per_patient"] = per_patient_full
        entry["missing_predictions"] = missing
        entry["missing_predictions_threshold"] = thr_missing
        entry["evaluated_patients"] = len(per_patient)
        entry["predicted_count_above_threshold"] = int(
            threshold_summary.get("predicted_count_above_threshold", 0)
        )
        entry["matched_count_above_threshold"] = int(
            threshold_summary.get("matched_predicted_above_threshold", 0)
        )
        entry["matched_rate_above_threshold"] = float(
            threshold_summary.get("matched_rate_above_threshold", 0.0)
        )
        results.append(entry)

    print("-" * 180)
    print(
        f"{'name':32} {'status':12} {'matched':>8} {'pred':>8} {'rate':>8} "
        f"{'R@3':>8} {'R@5':>8} {'MRR@3':>8} {'MRR@5':>8} {'MAR@3':>8} {'MAR@5':>8}"
    )
    print("-" * 180)
    for item in results:
        if item["status"] != "ok":
            print(
                f"{item['name'][:32]:32} {item['status']:12} "
                f"{'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8}"
            )
            continue

        summary = item["summary"]
        print(
            f"{item['name'][:32]:32} {item['status']:12} "
            f"{int(item.get('matched_count_above_threshold', 0)):>8} "
            f"{int(item.get('predicted_count_above_threshold', 0)):>8} "
            f"{_fmt_float(item.get('matched_rate_above_threshold', 0.0)):>8} "
            f"{_fmt_float(summary.get('overall_recall@3')):>8} "
            f"{_fmt_float(summary.get('overall_recall@5')):>8} "
            f"{_fmt_float(summary.get('mrr@3')):>8} "
            f"{_fmt_float(summary.get('mrr@5')):>8} "
            f"{_fmt_float(summary.get('mean_average_rank@3')):>8} "
            f"{_fmt_float(summary.get('mean_average_rank@5')):>8}"
        )
        _print_per_patient_table(item["name"], item["per_patient"])

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_setup = {
            "dirs_cli": list(args.dirs),
            "from_manifest": bool(args.from_manifest),
            "dirs_effective": [item["name"] for item in prediction_roots],
            "example_dir": str(ex_dir_resolved),
        }
        payload = {
            "ground_truth": str(gt_path),
            "ground_truth_source": gt_source_meta,
            "manifest": args.manifest,
            "results_root": str(results_root),
            "prediction_setup": pred_setup,
            "prediction_file": args.prediction_file,
            "patient_file": args.patient_file,
            "min_pred_score": args.min_pred_score,
            "example_json_only": bool(args.example_json_only),
            "example_filter": example_filter_meta,
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("-" * 180)
        print(f"Saved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
