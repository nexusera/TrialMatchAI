#!/usr/bin/env python3
"""Shared batch evaluation across prediction experiment dirs (one GT source per run)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypedDict

from evaluate_ranked_trials import (
    compute_threshold_match_stats,
    evaluate,
    load_ground_truth,
    load_predictions,
    load_predictions_above_score,
)

# This file lives at scripts/evaluation/ — repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT_BASE = REPO_ROOT / "scripts" / "results"
DEFAULT_PRED_BASE = REPO_ROOT / "results"
DEFAULT_PRED_FILE = "ranked_trials.json"
DEFAULT_MIN_PRED_SCORE = 0.5

PRED_DIRS_BASELINE: List[str] = [
    "some_trials_result_sim_thr0.3",
    "some_trials_result_sim_thr0.3_skip1v",
    "some_trials_result_sim_thr0.3_skip1v_qwen3_14b",
    "some_trials_result_sim_thr0.3_skip1v_qwen3_27b",
    "some_trials_result_sim_thr0.3_skip1v_qwen3_4b",
    "some_trials_result_sim_thr0.3_skip1v_qwen3_8b",
    "some_trials_result_sim_thr0.3_skip1vr_phi4",
    "some_trials_result_sim_thr0.3_skip1vr_phi4_wo_lora",
    "some_trials_result_sim_thr0.3_skip1vr_qwen3_14b",
    "some_trials_result_sim_thr0.3_skip1vr_qwen3_32b",
    "some_trials_result_sim_thr0.3_skip1vr_qwen3_4b",
    "some_trials_result_sim_thr0.3_skip1vr_qwen3_8b",
]


class Preset(TypedDict):
    label: str
    patient_subdir: str
    patient_file: str
    ground_truth_dir: str
    pred_dirs: List[str]


PRESETS: Dict[str, Preset] = {
    "breast001": {
        "label": "patient-breast-001",
        "patient_subdir": "patient-breast-001",
        "patient_file": "patient-breast-001.json",
        "ground_truth_dir": "qwen_patient_trial_matches_subset2",
        "pred_dirs": list(PRED_DIRS_BASELINE),
    },
    "lung001": {
        "label": "patient-lung-001",
        "patient_subdir": "patient-lung-001",
        "patient_file": "patient-lung-001.json",
        "ground_truth_dir": "qwen_patient_trial_matches_subset2",
        "pred_dirs": list(PRED_DIRS_BASELINE),
    },
    "patient67890": {
        "label": "patient67890",
        "patient_subdir": "phenopacket",
        "patient_file": "patient67890.json",
        "ground_truth_dir": "qwen_patient_trial_matches_subset_qwen122ba10",
        "pred_dirs": PRED_DIRS_BASELINE
        + [
            "qwen_patient_trial_matches_subset_qwen4b",
            "qwen_patient_trial_matches_subset_qwen8b",
            "qwen_patient_trial_matches_subset_qwen14b",
            "qwen_patient_trial_matches_subset_qwen27b",
        ],
    },
}


def _fmt_float(v: Any) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "NA"


def _resolve_prediction_dir(root: Path, patient_subdir: str) -> Path:
    with_patient = root / patient_subdir
    if with_patient.exists():
        return with_patient
    return root


def _resolve_prediction_root(
    pred_base: Path, gt_base: Path, pred_dir_name: str
) -> Path:
    candidates = [
        Path(pred_dir_name),
        REPO_ROOT / pred_dir_name,
        pred_base / pred_dir_name,
        gt_base / pred_dir_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return pred_base / pred_dir_name


def _resolve_prediction_source(
    pred_root: Path, patient_subdir: str, patient_file: str
) -> Path:
    patient_json = pred_root / patient_file
    if patient_json.exists():
        return patient_json
    return _resolve_prediction_dir(pred_root, patient_subdir)


def main_with_preset(preset_key: str) -> int:
    if preset_key not in PRESETS:
        raise SystemExit(f"Unknown preset {preset_key!r}; expected one of {sorted(PRESETS)}")
    preset = PRESETS[preset_key]
    label = preset["label"]

    parser = argparse.ArgumentParser(
        description=f"Batch evaluate {label} using one GT source across many prediction experiments."
    )
    parser.add_argument(
        "--gt-base",
        default=str(DEFAULT_GT_BASE),
        help=f"Base directory containing GT folders (default: {DEFAULT_GT_BASE})",
    )
    parser.add_argument(
        "--pred-base",
        default=str(DEFAULT_PRED_BASE),
        help=f"Base directory containing prediction folders (default: {DEFAULT_PRED_BASE})",
    )
    parser.add_argument(
        "--patient-file",
        default=preset["patient_file"],
        help=f"Patient GT filename inside the GT folder (default: {preset['patient_file']})",
    )
    parser.add_argument(
        "--patient-subdir",
        default=preset["patient_subdir"],
        help=(
            "Patient subdirectory under each prediction folder "
            f"(default: {preset['patient_subdir']})"
        ),
    )
    parser.add_argument(
        "--prediction-file",
        default=DEFAULT_PRED_FILE,
        help=f"Prediction filename (default: {DEFAULT_PRED_FILE})",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=preset["ground_truth_dir"],
        help=(
            "Single GT directory under --gt-base "
            f"(default: {preset['ground_truth_dir']})"
        ),
    )
    parser.add_argument(
        "--pred-dirs",
        nargs="*",
        default=None,
        help="Prediction directories under --pred-base. Default: preset list.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path for batch results.",
    )
    parser.add_argument(
        "--min-pred-score",
        type=float,
        default=DEFAULT_MIN_PRED_SCORE,
        help=(
            "Only count a prediction as matched when Score >= this threshold "
            "and the trial is in the GT matched_trials list."
        ),
    )
    args = parser.parse_args()
    pred_dirs: Sequence[str] = (
        args.pred_dirs if args.pred_dirs is not None else preset["pred_dirs"]
    )

    gt_base = Path(args.gt_base)
    pred_base = Path(args.pred_base)
    results: List[Dict[str, Any]] = []
    gt_dir_name = args.ground_truth_dir
    gt_path = gt_base / gt_dir_name / args.patient_file

    for pred_dir_name in pred_dirs:
        pred_root = _resolve_prediction_root(pred_base, gt_base, pred_dir_name)
        pred_dir = _resolve_prediction_source(
            pred_root, args.patient_subdir, args.patient_file
        )
        entry: Dict[str, Any] = {
            "name": pred_dir_name,
            "ground_truth": str(gt_path),
            "predictions_dir": str(pred_dir),
            "ground_truth_dir": gt_dir_name,
            "prediction_dir": pred_dir_name,
        }

        if not gt_path.exists():
            entry["status"] = "missing_gt"
            entry["missing_path"] = str(gt_path)
            results.append(entry)
            continue
        if not pred_dir.exists():
            entry["status"] = "missing_pred"
            entry["missing_path"] = str(pred_dir)
            results.append(entry)
            continue

        gt_patients = load_ground_truth(gt_path)
        pred_map = load_predictions(pred_dir, args.prediction_file)
        summary, per_patient, missing = evaluate(gt_patients, pred_map)
        pred_map_above = load_predictions_above_score(
            pred_dir, args.prediction_file, args.min_pred_score
        )
        threshold_summary, _, _ = compute_threshold_match_stats(
            gt_patients, pred_map_above
        )

        entry["status"] = "ok"
        entry["summary"] = summary
        entry["missing_predictions"] = missing
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

    print(f"Batch Evaluation: {label}")
    print("-" * 226)
    print(
        f"{'prediction_dir':58} {'gt_dir':38} {'status':12} {'matched':>8} {'pred':>8} {'rate':>8} {'R@3':>8} {'R@5':>8} "
        f"{'MRR@3':>8} {'MRR@5':>8} {'MAR@3':>8} {'MAR@5':>8}"
    )
    print("-" * 226)
    for item in results:
        gt_dir = item.get("ground_truth_dir", "")
        pred_dir_name = item.get("prediction_dir", item.get("name", ""))
        if item["status"] != "ok":
            print(
                f"{pred_dir_name[:58]:58} {gt_dir[:38]:38} {item['status']:12} "
                f"{'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8}"
            )
            if item.get("missing_path"):
                print(f"  missing_path: {item['missing_path']}")
            continue

        s = item["summary"]
        print(
            f"{pred_dir_name[:58]:58} {gt_dir[:38]:38} {item['status']:12} "
            f"{int(item.get('matched_count_above_threshold', 0)):>8} "
            f"{int(item.get('predicted_count_above_threshold', 0)):>8} "
            f"{_fmt_float(item.get('matched_rate_above_threshold', 0.0)):>8} "
            f"{_fmt_float(s.get('overall_recall@3')):>8} "
            f"{_fmt_float(s.get('overall_recall@5')):>8} "
            f"{_fmt_float(s.get('mrr@3')):>8} "
            f"{_fmt_float(s.get('mrr@5')):>8} "
            f"{_fmt_float(s.get('mean_average_rank@3')):>8} "
            f"{_fmt_float(s.get('mean_average_rank@5')):>8}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preset": preset_key,
            "gt_base": str(gt_base),
            "pred_base": str(pred_base),
            "patient_file": args.patient_file,
            "patient_subdir": args.patient_subdir,
            "prediction_file": args.prediction_file,
            "ground_truth_dir": args.ground_truth_dir,
            "pred_dirs": list(pred_dirs),
            "min_pred_score": args.min_pred_score,
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("-" * 226)
        print(f"Saved batch report: {out_path}")

    return 0
