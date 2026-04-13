#!/usr/bin/env python3
"""Shrink Qwen match result JSONs while keeping the same schema as the matcher.

Each input file (from scripts/evaluation/match_patients_trials_qwen.py) keeps keys such as
patient_id, patient_file, trials_evaluated, trials_assessed, complete,
matched_trial_count, matched_trials, and all_assessments — only the number of
trials in all_assessments is reduced (up to N matched + M unmatched per file),
in the same order as in the original all_assessments.

Example:
  python scripts/gt_generation/subset_qwen_match_result.py \\
    --n-matched 30 --n-unmatched 30 \\
    --output-dir results/qwen_patient_trial_matches_subset

  python scripts/gt_generation/subset_qwen_match_result.py \\
    results/qwen_patient_trial_matches/patient-001.json \\
    -o results/patient-001_subset.json
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


DEFAULT_INPUT = "results/qwen_patient_trial_matches"
DEFAULT_OUTPUT_DIR = "results/qwen_patient_trial_matches_subset"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")


def is_api_failure(a: Dict[str, Any]) -> bool:
    return bool(a.get("api_error"))


def is_matched(a: Dict[str, Any], min_score: float) -> bool:
    return bool(a.get("matched")) and float(a.get("match_score", 0.0)) >= min_score


def partition_assessments(
    raw: List[Any],
    min_score: float,
    skip_api_failures: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for a in raw:
        if not isinstance(a, dict):
            continue
        tid = str(a.get("trial_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        if skip_api_failures and is_api_failure(a):
            continue
        if is_matched(a, min_score):
            matched.append(a)
        else:
            unmatched.append(a)

    return matched, unmatched


def pick_matched(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    rows = sorted(
        rows,
        key=lambda x: float(x.get("match_score", 0.0)),
        reverse=True,
    )
    return rows[:n]


def pick_unmatched(
    rows: List[Dict[str, Any]],
    n: int,
    strategy: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not rows or n <= 0:
        return []

    if strategy == "random":
        return rng.sample(rows, min(n, len(rows)))

    if strategy == "high_score":
        rows = sorted(
            rows,
            key=lambda x: float(x.get("match_score", 0.0)),
            reverse=True,
        )
        return rows[:n]

    rows = sorted(
        rows,
        key=lambda x: float(x.get("match_score", 0.0)),
    )
    return rows[:n]


def build_keep_ids(
    matched_pick: List[Dict[str, Any]],
    unmatched_pick: List[Dict[str, Any]],
) -> Set[str]:
    ids: Set[str] = set()
    for a in matched_pick + unmatched_pick:
        tid = str(a.get("trial_id") or "").strip()
        if tid:
            ids.add(tid)
    return ids


def subset_all_assessments_in_order(
    raw: List[Any],
    keep_ids: Set[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not keep_ids:
        return out
    pending = set(keep_ids)
    for a in raw:
        if not isinstance(a, dict):
            continue
        tid = str(a.get("trial_id") or "").strip()
        if tid in pending:
            out.append(copy.deepcopy(a))
            pending.discard(tid)
        if not pending:
            break
    return out


def recompute_matcher_fields(
    assessments: List[Dict[str, Any]],
    min_match_score: float,
) -> tuple[List[Dict[str, Any]], int]:
    matched_trials = [
        item
        for item in assessments
        if bool(item.get("matched"))
        and float(item.get("match_score", 0.0)) >= min_match_score
    ]
    matched_trials.sort(
        key=lambda x: float(x.get("match_score", 0.0)),
        reverse=True,
    )
    return matched_trials, len(matched_trials)


def subset_patient_record(
    data: Dict[str, Any],
    min_match_score: float,
    n_matched: int,
    n_unmatched: int,
    unmatched_strategy: str,
    rng: random.Random,
    skip_api_failures: bool,
) -> Dict[str, Any]:
    raw = data.get("all_assessments")
    if not isinstance(raw, list):
        return copy.deepcopy(data)

    matched_pool, unmatched_pool = partition_assessments(
        raw, min_match_score, skip_api_failures
    )
    m_pick = pick_matched(matched_pool, n_matched)
    u_pick = pick_unmatched(unmatched_pool, n_unmatched, unmatched_strategy, rng)
    keep_ids = build_keep_ids(m_pick, u_pick)
    new_assessments = subset_all_assessments_in_order(raw, keep_ids)

    out = copy.deepcopy(data)
    out["all_assessments"] = new_assessments
    matched_trials, count = recompute_matcher_fields(new_assessments, min_match_score)
    out["matched_trials"] = matched_trials
    out["matched_trial_count"] = count
    out["trials_assessed"] = len(new_assessments)
    out["trials_evaluated"] = len(new_assessments)
    out["complete"] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Subset trials per file; output keeps match_patients_trials_qwen JSON shape."
    )
    p.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Result JSON file or directory (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file (only when input is a single .json file).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory when input is a directory "
        f"(default: {DEFAULT_OUTPUT_DIR} if input is a directory).",
    )
    p.add_argument(
        "--n-matched",
        type=int,
        default=20,
        help="Max matched trials to keep per patient file (default: 20).",
    )
    p.add_argument(
        "--n-unmatched",
        type=int,
        default=20,
        help="Max unmatched trials to keep per patient file (default: 20).",
    )
    p.add_argument(
        "--min-match-score",
        type=float,
        default=0.5,
        help="Same threshold as match_patients_trials_qwen --min-match-score (default: 0.5).",
    )
    p.add_argument(
        "--unmatched-strategy",
        choices=("high_score", "low_score", "random"),
        default="high_score",
        help="How to choose unmatched trials (default: high_score).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed when --unmatched-strategy random.",
    )
    p.add_argument(
        "--include-api-failures",
        action="store_true",
        help="Include API-failure rows in the unmatched pool (otherwise dropped).",
    )
    args = p.parse_args()

    input_path = Path(args.input_path)
    try:
        files = list_json_files(input_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not files:
        print(f"No JSON files found in {input_path}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    if input_path.is_file():
        out_file = Path(args.output) if args.output else input_path.with_name(
            f"{input_path.stem}_subset.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = load_json(input_path)
        except Exception as exc:
            print(f"Error reading {input_path}: {exc}", file=sys.stderr)
            return 1
        out = subset_patient_record(
            data,
            args.min_match_score,
            args.n_matched,
            args.n_unmatched,
            args.unmatched_strategy,
            rng,
            skip_api_failures=not args.include_api_failures,
        )
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_file} ({out['trials_assessed']} trials)")
        return 0

    # Directory input: write one JSON per source file
    out_dir = Path(args.output_dir or args.output or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output and not args.output_dir:
        # -o was used; if it looks like a file path, reject for dir input
        op = Path(args.output)
        if op.suffix.lower() == ".json":
            print(
                "Input is a directory: use --output-dir DIR (not a single .json).",
                file=sys.stderr,
            )
            return 1

    written = 0
    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            print(f"Skip {path.name}: {exc}", file=sys.stderr)
            continue
        out = subset_patient_record(
            data,
            args.min_match_score,
            args.n_matched,
            args.n_unmatched,
            args.unmatched_strategy,
            rng,
            skip_api_failures=not args.include_api_failures,
        )
        dest = out_dir / path.name
        with dest.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        written += 1

    print(f"Wrote {written} file(s) under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

