#!/usr/bin/env python3
"""Batch-evaluate patient-lung-001 across multiple prediction directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluate_ranked_trials import evaluate, load_ground_truth, load_predictions


DEFAULT_GT = "results/qwen_patient_trial_matches_subset2/patient-lung-001.json"
DEFAULT_PRED_BASE = "../results"
DEFAULT_PATIENT_SUBDIR = "patient-lung-001"
DEFAULT_PRED_FILE = "ranaked_trials.json"

DEFAULT_DIRS = [
    "some_trials_result_sim_thr0.3",
    "some_trials_result_sim_thr0.3_skip1",
    "some_trials_result_sim_thr0.3_skip1_wo_cot_lora",
    "some_trials_result_sim_thr0.3_skip1_wo_reranker_lora",
    "some_trials_result_sim_thr0.3_skip1_wo_both_lora",
    "some_trials_result_sim_thr0.3_wo_both_lora",
    "some_trials_result_sim_thr0.3_wo_cot_lora",
    "some_trials_result_sim_thr0.3_wo_reranker_lora",
]


def _resolve_prediction_dir(base: Path, dir_name: str, patient_subdir: str) -> Path:
    root = base / dir_name
    with_patient = root / patient_subdir
    if with_patient.exists():
        return with_patient
    return root


def _fmt_float(v: Any) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "NA"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch evaluate patient-lung-001 across multiple prediction directories."
    )
    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GT,
        help=f"Ground-truth JSON path (default: {DEFAULT_GT})",
    )
    parser.add_argument(
        "--pred-base",
        default=DEFAULT_PRED_BASE,
        help=f"Base directory containing prediction folders (default: {DEFAULT_PRED_BASE})",
    )
    parser.add_argument(
        "--patient-subdir",
        default=DEFAULT_PATIENT_SUBDIR,
        help=f"Patient subdirectory under each prediction folder (default: {DEFAULT_PATIENT_SUBDIR})",
    )
    parser.add_argument(
        "--prediction-file",
        default=DEFAULT_PRED_FILE,
        help=f"Prediction filename (default: {DEFAULT_PRED_FILE})",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=DEFAULT_DIRS,
        help="Prediction folder names under --pred-base. If omitted, built-in list is used.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path for batch results.",
    )
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    pred_base = Path(args.pred_base)
    gt_patients = load_ground_truth(gt_path)

    results: List[Dict[str, Any]] = []
    for dir_name in args.dirs:
        pred_dir = _resolve_prediction_dir(pred_base, dir_name, args.patient_subdir)
        entry: Dict[str, Any] = {
            "name": dir_name,
            "predictions_dir": str(pred_dir),
        }

        if not pred_dir.exists():
            entry["status"] = "missing_dir"
            results.append(entry)
            continue

        pred_map = load_predictions(pred_dir, args.prediction_file)
        summary, per_patient, missing = evaluate(gt_patients, pred_map)
        entry["status"] = "ok"
        entry["summary"] = summary
        entry["missing_predictions"] = missing
        entry["evaluated_patients"] = len(per_patient)
        results.append(entry)

    print("Batch Evaluation: patient-lung-001")
    print("-" * 110)
    print(
        f"{'name':44} {'status':10} {'R@3':>8} {'R@5':>8} "
        f"{'MRR@3':>8} {'MRR@5':>8} {'MAR@3':>8} {'MAR@5':>8}"
    )
    print("-" * 110)
    for item in results:
        if item["status"] != "ok":
            print(
                f"{item['name'][:44]:44} {item['status']:10} "
                f"{'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8}"
            )
            continue
        s = item["summary"]
        print(
            f"{item['name'][:44]:44} {item['status']:10} "
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
            "ground_truth": str(gt_path),
            "pred_base": str(pred_base),
            "prediction_file": args.prediction_file,
            "patient_subdir": args.patient_subdir,
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("-" * 110)
        print(f"Saved batch report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

