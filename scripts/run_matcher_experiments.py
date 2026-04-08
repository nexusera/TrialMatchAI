#!/usr/bin/env python3
"""Run Matcher.main across a fixed set of experiment variants.

This script focuses on the experiment matrix discussed in the repo notes:

- baseline
- skip1
- skip1v + phi-4 with/without LoRA
- skip1v + Qwen3-{4B,8B,14B,32B} without LoRA
- skip1vr + phi-4 with/without LoRA
- skip1vr + Qwen3-{4B,8B,14B,32B} without LoRA

It also supports pre-populating per-patient output folders with cached
artifacts such as keywords.json / nct_ids.txt / top_trials.txt so later
stages can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_PATIENTS_DIR = REPO_ROOT / "example_raw"
DEFAULT_TRIALS_JSON_FOLDER = REPO_ROOT / "data" / "processed_trials.200"
DEFAULT_OUTPUT_PREFIX = "some_trials_result_sim_thr0.3"
DEFAULT_ES_INDEX_TRIALS = "clinical_trials_200"
DEFAULT_ES_INDEX_TRIALS_ELIGIBILITY = "trials_eligibility_200"
DEFAULT_HF_HOME = Path("~/.cache/huggingface").expanduser()
DEFAULT_CUDA_VISIBLE_DEVICES = "7"
DEFAULT_MATCHER_MODULE = "Matcher.main"
DEFAULT_REQUIRED_PREFILL_FILES = (
    "keywords.json",
    "nct_ids.txt",
    "first_level_scores.json",
    "top_trials.txt",
)

QWEN_MODELS: Dict[str, str] = {
    "qwen3_4b": "/data/ocean/model/Qwen/Qwen3-4B/",
    "qwen3_8b": "/data/ocean/model/Qwen/Qwen3-8B/",
    "qwen3_14b": "/data/ocean/model/qwen3-14b/",
    "qwen3_32b": "/data/ocean/model/Qwen/Qwen3-32B/",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    skip_first_level: bool = False
    second_level_search_mode: Optional[str] = None
    resume_from_second_level: bool = False
    base_model: Optional[str] = None
    cot_adapter_path: Optional[str] = None


EXPERIMENTS: Sequence[Experiment] = (
    Experiment(name="baseline"),
    Experiment(name="skip1", skip_first_level=True),
    Experiment(
        name="skip1v_phi4",
        skip_first_level=True,
        second_level_search_mode="all_rerank",
    ),
    Experiment(
        name="skip1v_phi4_wo_lora",
        skip_first_level=True,
        second_level_search_mode="all_rerank",
        cot_adapter_path="",
    ),
    Experiment(
        name="skip1vr_phi4",
        resume_from_second_level=True,
    ),
    Experiment(
        name="skip1vr_phi4_wo_lora",
        resume_from_second_level=True,
        cot_adapter_path="",
    ),
    *(Experiment(
        name=f"skip1v_{alias}",
        skip_first_level=True,
        second_level_search_mode="all_rerank",
        base_model=model_path,
        cot_adapter_path="",
    ) for alias, model_path in QWEN_MODELS.items()),
    *(Experiment(
        name=f"skip1vr_{alias}",
        resume_from_second_level=True,
        base_model=model_path,
        cot_adapter_path="",
    ) for alias, model_path in QWEN_MODELS.items()),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run TrialMatchAI experiment variants with stable naming."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Optional subset of experiment names to run. Default: run all built-in experiments.",
    )
    parser.add_argument(
        "--patients-dir",
        default=str(DEFAULT_PATIENTS_DIR),
        help=f"Patient phenopacket directory (default: {DEFAULT_PATIENTS_DIR})",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Root directory containing experiment outputs (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output folder prefix before suffixes (default: {DEFAULT_OUTPUT_PREFIX})",
    )
    parser.add_argument(
        "--trials-json-folder",
        default=str(DEFAULT_TRIALS_JSON_FOLDER),
        help=f"Trials JSON folder passed to Matcher.main (default: {DEFAULT_TRIALS_JSON_FOLDER})",
    )
    parser.add_argument(
        "--es-index-trials",
        default=DEFAULT_ES_INDEX_TRIALS,
        help=f"Trial index name (default: {DEFAULT_ES_INDEX_TRIALS})",
    )
    parser.add_argument(
        "--es-index-trials-eligibility",
        default=DEFAULT_ES_INDEX_TRIALS_ELIGIBILITY,
        help=f"Eligibility index name (default: {DEFAULT_ES_INDEX_TRIALS_ELIGIBILITY})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--vector-score-threshold",
        type=float,
        default=0.3,
        help="Vector similarity threshold.",
    )
    parser.add_argument(
        "--max-trials-rag",
        type=int,
        default=1000,
        help="Maximum trials sent into RAG.",
    )
    parser.add_argument(
        "--artifact-source-dir",
        default="",
        help=(
            "Optional base output directory to copy cached per-patient artifacts from "
            "before each experiment runs."
        ),
    )
    parser.add_argument(
        "--artifact-source-dir-skip1vr",
        default="",
        help=(
            "Optional override source directory used only for skip1vr experiments. "
            "Useful when top_trials.txt should come from a skip1v run."
        ),
    )
    parser.add_argument(
        "--prefill-files",
        nargs="*",
        default=list(DEFAULT_REQUIRED_PREFILL_FILES),
        help="Artifact filenames copied into each patient output folder when available.",
    )
    parser.add_argument(
        "--force-prefill",
        action="store_true",
        help="Overwrite existing cached artifacts in target output folders.",
    )
    parser.add_argument(
        "--require-prefill-for-resume",
        action="store_true",
        help="Fail skip1vr experiments if required prefill artifacts are missing.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=f"Python executable used to launch Matcher.main (default: {sys.executable})",
    )
    parser.add_argument(
        "--matcher-module",
        default=DEFAULT_MATCHER_MODULE,
        help=f"Python module for the matcher entrypoint (default: {DEFAULT_MATCHER_MODULE})",
    )
    parser.add_argument(
        "--workdir",
        default=str(SOURCE_DIR),
        help=f"Working directory used to launch Matcher.main (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help=f"CUDA_VISIBLE_DEVICES value (default: {DEFAULT_CUDA_VISIBLE_DEVICES})",
    )
    parser.add_argument(
        "--hf-home",
        default=str(DEFAULT_HF_HOME),
        help=f"HF_HOME value (default: {DEFAULT_HF_HOME})",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF_TOKEN value passed to subprocesses. Default: read from current environment.",
    )
    parser.add_argument(
        "--transformers-offline",
        default="1",
        help="TRANSFORMERS_OFFLINE value (default: 1)",
    )
    parser.add_argument(
        "--hf-hub-offline",
        default="1",
        help="HF_HUB_OFFLINE value (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and manifest without running experiments.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one experiment fails.",
    )
    parser.add_argument(
        "--manifest-name",
        default="matcher_experiment_manifest.json",
        help="Manifest filename written under --results-root.",
    )
    return parser.parse_args()


def _select_experiments(requested: Sequence[str]) -> List[Experiment]:
    if not requested:
        return list(EXPERIMENTS)

    by_name = {exp.name: exp for exp in EXPERIMENTS}
    missing = [name for name in requested if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise SystemExit(
            f"Unknown experiment(s): {', '.join(missing)}\nAvailable: {available}"
        )
    return [by_name[name] for name in requested]


def _output_dir_name(prefix: str, experiment: Experiment) -> str:
    if experiment.name == "baseline":
        return prefix
    return f"{prefix}_{experiment.name}"


def _iter_patient_ids(patients_dir: Path) -> List[str]:
    if not patients_dir.exists():
        raise SystemExit(f"Patients dir not found: {patients_dir}")
    patient_files = sorted(p for p in patients_dir.iterdir() if p.suffix == ".json")
    if not patient_files:
        raise SystemExit(f"No patient JSON files found in: {patients_dir}")
    return [p.stem for p in patient_files]


def _copy_prefill_artifacts(
    patient_ids: Sequence[str],
    source_root: Path,
    target_root: Path,
    filenames: Sequence[str],
    force: bool,
) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    for patient_id in patient_ids:
        src_dir = source_root / patient_id
        dst_dir = target_root / patient_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src_path = src_dir / filename
            dst_path = dst_dir / filename
            if not src_path.exists():
                continue
            if dst_path.exists() and not force:
                continue
            shutil.copy2(src_path, dst_path)
            copied.append(
                {
                    "patient_id": patient_id,
                    "file": filename,
                    "source": str(src_path),
                    "target": str(dst_path),
                }
            )
    return copied


def _has_resume_artifacts(
    patient_ids: Sequence[str], target_root: Path, filenames: Sequence[str]
) -> List[Dict[str, object]]:
    missing: List[Dict[str, object]] = []
    for patient_id in patient_ids:
        patient_missing = [
            name for name in filenames if not (target_root / patient_id / name).exists()
        ]
        if patient_missing:
            missing.append({"patient_id": patient_id, "missing_files": patient_missing})
    return missing


def _build_command(
    args: argparse.Namespace, experiment: Experiment, output_dir: Path
) -> List[str]:
    cmd = [
        args.python,
        "-m",
        args.matcher_module,
        "--patients-dir",
        str(Path(args.patients_dir)),
        "--output-dir",
        str(output_dir),
        "--es-index-trials",
        args.es_index_trials,
        "--es-index-trials-eligibility",
        args.es_index_trials_eligibility,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--vector-score-threshold",
        str(args.vector_score_threshold),
        "--trials-json-folder",
        str(Path(args.trials_json_folder)),
        "--max-trials-rag",
        str(args.max_trials_rag),
    ]
    if experiment.skip_first_level:
        cmd.append("--skip-first-level")
    if experiment.second_level_search_mode:
        cmd.extend(["--second-level-search-mode", experiment.second_level_search_mode])
    if experiment.resume_from_second_level:
        cmd.append("--resume-from-second-level")
    if experiment.base_model:
        cmd.extend(["--base-model", experiment.base_model])
    if experiment.cot_adapter_path is not None:
        cmd.extend(["--cot-adapter-path", experiment.cot_adapter_path])
    return cmd


def _build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    env["TRANSFORMERS_OFFLINE"] = args.transformers_offline
    env["HF_HUB_OFFLINE"] = args.hf_hub_offline
    env["HF_TOKEN"] = args.hf_token
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def main() -> int:
    args = _parse_args()
    experiments = _select_experiments(args.experiments)
    results_root = Path(args.results_root).resolve()
    patients_dir = Path(args.patients_dir).resolve()
    workdir = Path(args.workdir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    patient_ids: List[str] = []
    if patients_dir.exists():
        patient_ids = _iter_patient_ids(patients_dir)
    elif not args.dry_run:
        raise SystemExit(f"Patients dir not found: {patients_dir}")

    manifest: Dict[str, object] = {
        "patients_dir": str(patients_dir),
        "results_root": str(results_root),
        "output_prefix": args.output_prefix,
        "trials_json_folder": str(Path(args.trials_json_folder).resolve()),
        "prefill_files": list(args.prefill_files),
        "experiments": [],
    }

    env = _build_env(args)
    overall_exit_code = 0

    for experiment in experiments:
        output_dir = results_root / _output_dir_name(args.output_prefix, experiment)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefill_source = args.artifact_source_dir
        if experiment.resume_from_second_level and args.artifact_source_dir_skip1vr:
            prefill_source = args.artifact_source_dir_skip1vr

        copied_files: List[Dict[str, str]] = []
        if prefill_source:
            copied_files = _copy_prefill_artifacts(
                patient_ids=patient_ids,
                source_root=Path(prefill_source).resolve(),
                target_root=output_dir,
                filenames=args.prefill_files,
                force=args.force_prefill,
            )

        missing_resume_files: List[Dict[str, object]] = []
        if experiment.resume_from_second_level:
            missing_resume_files = _has_resume_artifacts(
                patient_ids=patient_ids,
                target_root=output_dir,
                filenames=("keywords.json", "top_trials.txt"),
            )
            if missing_resume_files and args.require_prefill_for_resume:
                status = "missing_resume_artifacts"
                entry = {
                    "name": experiment.name,
                    "output_dir": str(output_dir),
                    "status": status,
                    "copied_files": copied_files,
                    "missing_resume_files": missing_resume_files,
                    "command": _build_command(args, experiment, output_dir),
                }
                cast_list = manifest["experiments"]
                assert isinstance(cast_list, list)
                cast_list.append(entry)
                overall_exit_code = 1
                if args.stop_on_error:
                    break
                continue

        command = _build_command(args, experiment, output_dir)
        entry = {
            "name": experiment.name,
            "output_dir": str(output_dir),
            "status": "dry_run" if args.dry_run else "pending",
            "copied_files": copied_files,
            "missing_resume_files": missing_resume_files,
            "command": command,
        }

        cast_list = manifest["experiments"]
        assert isinstance(cast_list, list)
        cast_list.append(entry)

        print("=" * 120)
        print(f"Experiment: {experiment.name}")
        print(f"Output dir : {output_dir}")
        print(f"Command    : {' '.join(command)}")
        if copied_files:
            print(f"Prefill    : copied {len(copied_files)} files")
        if missing_resume_files:
            print(f"Resume prefill missing: {json.dumps(missing_resume_files, ensure_ascii=False)}")

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=workdir, env=env, check=False)
        entry["returncode"] = completed.returncode
        entry["status"] = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            overall_exit_code = completed.returncode
            if args.stop_on_error:
                break

    manifest_path = results_root / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 120)
    print(f"Saved manifest: {manifest_path}")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
