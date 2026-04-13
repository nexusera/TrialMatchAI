#!/usr/bin/env python3
"""Extract matched trial IDs from scripts/evaluation/match_patients_trials_qwen.py JSON outputs.

Each result file is typically ``<patient_id>.json`` with ``matched_trials`` (already
score-filtered at save time) and optional ``all_assessments``.

Usage:
  python scripts/extract_matched_trial_ids_from_qwen_results.py results/qwen_patient_trial_matches
  python scripts/extract_matched_trial_ids_from_qwen_results.py results/qwen_patient_trial_matches/P001.json
  python scripts/extract_matched_trial_ids_from_qwen_results.py --source all_assessments --min-score 0.5 dir/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trial_id_from_item(item: Dict[str, Any]) -> Optional[str]:
    for key in ("trial_id", "nct_id", "TrialID", "trialId"):
        raw = item.get(key)
        if raw is None:
            continue
        tid = str(raw).strip()
        if tid:
            return tid.upper()
    return None


def _ids_from_matched_trials(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        tid = _trial_id_from_item(item)
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def _ids_from_all_assessments(
    items: List[Any], min_score: float
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("matched")):
            continue
        try:
            score = float(item.get("match_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            continue
        tid = _trial_id_from_item(item)
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def extract_from_record(
    data: Dict[str, Any],
    *,
    source: str,
    min_score: float,
) -> Tuple[str, List[str]]:
    pid = str(data.get("patient_id") or "").strip() or "?"

    if source == "matched_trials":
        raw = data.get("matched_trials")
        if not isinstance(raw, list):
            return pid, []
        return pid, _ids_from_matched_trials(raw)

    raw = data.get("all_assessments")
    if not isinstance(raw, list):
        return pid, []
    return pid, _ids_from_all_assessments(raw, min_score)


def _iter_result_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.glob("*.json")):
        if p.is_file():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print matched trial IDs from Qwen matcher JSON outputs."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Result JSON file or directory of *.json patient results.",
    )
    parser.add_argument(
        "--source",
        choices=("matched_trials", "all_assessments"),
        default="matched_trials",
        help="Take IDs from matched_trials (default) or re-filter all_assessments.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="With --source all_assessments: minimum match_score (default: 0.5).",
    )
    parser.add_argument(
        "--with-patient",
        action="store_true",
        help="Prefix each line with patient_id<TAB> (default for directories).",
    )
    parser.add_argument(
        "--no-with-patient",
        action="store_true",
        help="Only print trial IDs, even when reading a directory.",
    )
    args = parser.parse_args()
    root = args.path.resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 1

    files = list(_iter_result_files(root))
    if not files:
        print(f"No JSON files under: {root}", file=sys.stderr)
        return 1

    multi = len(files) > 1
    with_patient = args.with_patient or (multi and not args.no_with_patient)

    for path in files:
        try:
            raw = _load_json(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(raw, dict):
            print(f"Skip {path}: expected JSON object", file=sys.stderr)
            continue
        pid, ids = extract_from_record(
            raw, source=args.source, min_score=args.min_score
        )
        for tid in ids:
            if with_patient:
                print(f"{pid}\t{tid}")
            else:
                print(tid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
