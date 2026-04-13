#!/usr/bin/env python3
"""Summarize result JSONs produced by scripts/evaluation/match_patients_trials_qwen.py.

Usage:
    python scripts/stat_match_results_qwen.py
    python scripts/stat_match_results_qwen.py results/qwen_patient_trial_matches
    python scripts/stat_match_results_qwen.py results/qwen_patient_trial_matches --top 20
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_RESULTS_DIR = "results/qwen_patient_trial_matches"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def summarize_file(data: Dict[str, Any]) -> Dict[str, Any]:
    matched_trials = data.get("matched_trials") or []
    all_assessments = data.get("all_assessments") or []

    matched_scores = [
        as_float(item.get("match_score"))
        for item in matched_trials
        if isinstance(item, dict)
    ]
    all_scores = [
        as_float(item.get("match_score"))
        for item in all_assessments
        if isinstance(item, dict)
    ]

    return {
        "patient_id": str(data.get("patient_id") or ""),
        "complete": bool(data.get("complete")),
        "trials_evaluated": int(data.get("trials_evaluated") or 0),
        "trials_assessed": int(data.get("trials_assessed") or 0),
        "matched_trial_count": int(data.get("matched_trial_count") or 0),
        "matched_scores": matched_scores,
        "all_scores": all_scores,
        "matched_trials": matched_trials,
        "all_assessments": all_assessments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen patient-trial matching result JSON files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help=f"Result JSON file or directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top items to show for rankings/counters (default: 10).",
    )
    args = parser.parse_args()

    target = Path(args.path)
    try:
        json_files = list_json_files(target)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not json_files:
        print(f"No JSON files found in {target}.", file=sys.stderr)
        return 1

    parsed: List[Dict[str, Any]] = []
    errors = 0

    complete_count = 0
    total_trials_evaluated = 0
    total_trials_assessed = 0
    total_matched_trials = 0
    matched_patients = 0

    matched_score_counter: List[float] = []
    all_score_counter: List[float] = []
    missing_info_counter: Counter[str] = Counter()
    conflicts_counter: Counter[str] = Counter()
    matched_trial_id_counter: Counter[str] = Counter()
    incomplete_patients: List[str] = []
    patient_match_counts: List[tuple[str, int]] = []

    for path in json_files:
        try:
            data = load_json(path)
            item = summarize_file(data)
        except Exception as exc:
            print(f"  Error reading {path.name}: {exc}", file=sys.stderr)
            errors += 1
            continue

        parsed.append(item)
        patient_id = item["patient_id"] or path.stem
        patient_match_counts.append((patient_id, item["matched_trial_count"]))

        if item["complete"]:
            complete_count += 1
        else:
            incomplete_patients.append(patient_id)

        total_trials_evaluated += item["trials_evaluated"]
        total_trials_assessed += item["trials_assessed"]
        total_matched_trials += item["matched_trial_count"]
        if item["matched_trial_count"] > 0:
            matched_patients += 1

        matched_score_counter.extend(item["matched_scores"])
        all_score_counter.extend(item["all_scores"])

        for trial in item["matched_trials"]:
            if not isinstance(trial, dict):
                continue
            tid = str(trial.get("trial_id") or "").strip()
            if tid:
                matched_trial_id_counter[tid] += 1
            for text in trial.get("missing_information") or []:
                if str(text).strip():
                    missing_info_counter[str(text).strip()] += 1
            for text in trial.get("conflicts") or []:
                if str(text).strip():
                    conflicts_counter[str(text).strip()] += 1

    ok_count = len(parsed)
    if ok_count == 0:
        print("No valid result JSONs could be parsed.", file=sys.stderr)
        return 1

    patient_match_counts.sort(key=lambda x: (-x[1], x[0]))

    print(f"\n{'=' * 78}")
    print(f"  Qwen Match Result Statistics — {target}")
    print(f"  Files scanned: {len(json_files)}  |  Parsed OK: {ok_count}  |  Errors: {errors}")
    print(f"{'=' * 78}\n")

    print("  Overall")
    print(f"  Patients complete:      {complete_count}/{ok_count} ({100 * complete_count / ok_count:.1f}%)")
    print(f"  Patients with matches:  {matched_patients}/{ok_count} ({100 * matched_patients / ok_count:.1f}%)")
    print(f"  Trials evaluated total: {total_trials_evaluated}")
    print(f"  Trials assessed total:  {total_trials_assessed}")
    print(f"  Matched trials total:   {total_matched_trials}")
    print(f"  Avg matches/patient:    {total_matched_trials / ok_count:.2f}")
    print(f"  Median matches/patient: {median([count for _, count in patient_match_counts]):.2f}")

    coverage = (
        100 * total_trials_assessed / total_trials_evaluated
        if total_trials_evaluated else 0.0
    )
    print(f"  Assessment coverage:    {coverage:.1f}%")

    print("\n  Scores")
    print(f"  Avg score (all):        {avg(all_score_counter):.4f}")
    print(f"  Median score (all):     {median(all_score_counter):.4f}")
    print(f"  Avg score (matched):    {avg(matched_score_counter):.4f}")
    print(f"  Median score (matched): {median(matched_score_counter):.4f}")

    print(f"\n  Top {min(args.top, len(patient_match_counts))} patients by matched_trial_count")
    for patient_id, count in patient_match_counts[: args.top]:
        print(f"    {patient_id:<20} {count}")

    if incomplete_patients:
        print(f"\n  Incomplete patients ({len(incomplete_patients)}):")
        for patient_id in incomplete_patients[: args.top]:
            print(f"    {patient_id}")
        if len(incomplete_patients) > args.top:
            print(f"    ... and {len(incomplete_patients) - args.top} more")

    if matched_trial_id_counter:
        print(f"\n  Most frequently matched trial IDs (top {args.top})")
        for trial_id, count in matched_trial_id_counter.most_common(args.top):
            print(f"    {trial_id:<20} {count}")

    if missing_info_counter:
        print(f"\n  Most common missing_information items (top {args.top})")
        for text, count in missing_info_counter.most_common(args.top):
            print(f"    {text:<40} {count}")

    if conflicts_counter:
        print(f"\n  Most common conflicts items (top {args.top})")
        for text, count in conflicts_counter.most_common(args.top):
            print(f"    {text:<40} {count}")
    else:
        print("\n  Most common conflicts items")
        print("    None")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

