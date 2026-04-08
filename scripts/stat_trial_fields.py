#!/usr/bin/env python3
"""Report field-level completeness statistics for trial JSON files.

Works on both raw JSONs (from jsonify.py) and processed/embedded JSONs
(from prepare_trials.py).

Usage:
    python scripts/stat_trial_fields.py data/custom/trials_jsons
    python scripts/stat_trial_fields.py data/custom/processed_trials --show-ids gender minimum_age
    python scripts/stat_trial_fields.py data/processed_trials --top 30
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


CORE_FIELDS = [
    "nct_id",
    "brief_title",
    "official_title",
    "brief_summary",
    "detailed_description",
    "condition",
    "eligibility_criteria",
    "overall_status",
    "phase",
    "study_type",
    "gender",
    "minimum_age",
    "maximum_age",
    "sponsor",
    "intervention",
    "location",
    "start_date",
    "completion_date",
    "reference",
]

VECTOR_FIELDS = [
    "brief_title_vector",
    "brief_summary_vector",
    "condition_vector",
    "eligibility_criteria_vector",
]

EXTRA_FIELDS = [
    "ecog_range",
]


def classify_value(val: Any) -> str:
    """Classify a field value as 'present', 'null', 'empty', or 'zero_vector'."""
    if val is None:
        return "null"
    if isinstance(val, str) and not val.strip():
        return "empty"
    if isinstance(val, list):
        if len(val) == 0:
            return "empty"
        # Only check for zero-vectors on numeric lists (embeddings)
        if val and isinstance(val[0], (int, float)):
            if all(v == 0.0 for v in val):
                return "zero_vector"
    return "present"


def analyze_file(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    all_fields = CORE_FIELDS + VECTOR_FIELDS + EXTRA_FIELDS
    for field in all_fields:
        if field not in data:
            result[field] = "missing"
        else:
            result[field] = classify_value(data[field])

    # Also detect any extra unknown fields
    known = set(all_fields)
    for key in data:
        if key not in known:
            result[key] = classify_value(data[key])

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report field completeness statistics for trial JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        help="Directory containing trial JSON files to analyze.",
    )
    parser.add_argument(
        "--show-ids",
        nargs="+",
        metavar="FIELD",
        help="List trial IDs that are missing/null for these fields.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many sample IDs to show per field (default: 10).",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        return 1

    json_files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".json")
    if not json_files:
        print(f"No JSON files found in {directory}.", file=sys.stderr)
        return 1

    # Analyze all files
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_ids: Dict[str, List[str]] = defaultdict(list)
    total = len(json_files)
    errors = 0

    for path in json_files:
        try:
            result = analyze_file(path)
        except Exception as exc:
            print(f"  Error reading {path.name}: {exc}", file=sys.stderr)
            errors += 1
            continue

        trial_id = path.stem
        for field, status in result.items():
            stats[field][status] += 1
            if status in ("missing", "null", "empty", "zero_vector"):
                missing_ids[field].append(trial_id)

    # Determine which fields to report
    report_fields = []
    has_vectors = any(f in stats for f in VECTOR_FIELDS)

    for f in CORE_FIELDS:
        if f in stats:
            report_fields.append(f)
    if has_vectors:
        for f in VECTOR_FIELDS:
            if f in stats:
                report_fields.append(f)
    for f in EXTRA_FIELDS:
        if f in stats:
            report_fields.append(f)

    # Print summary
    ok_count = total - errors
    print(f"\n{'=' * 78}")
    print(f"  Trial JSON Field Statistics — {directory}")
    print(f"  Files scanned: {total}  |  Parsed OK: {ok_count}  |  Errors: {errors}")
    print(f"{'=' * 78}\n")

    hdr = f"  {'Field':<30s} {'Present':>8s} {'Null':>8s} {'Empty':>8s} {'Missing':>8s} {'ZeroVec':>8s}"
    print(hdr)
    print(f"  {'-' * 72}")

    for field in report_fields:
        s = stats[field]
        present = s.get("present", 0)
        null = s.get("null", 0)
        empty = s.get("empty", 0)
        missing = s.get("missing", 0)
        zero_vec = s.get("zero_vector", 0)
        bad = null + empty + missing + zero_vec

        marker = " " if bad == 0 else "*"
        pct = f"({100 * present / ok_count:.0f}%)" if ok_count else ""

        print(
            f"{marker} {'  ' if field in VECTOR_FIELDS else ''}"
            f"{field:<30s} {present:>7d} {pct:>5s}"
            f" {null:>7d} {empty:>7d} {missing:>7d} {zero_vec:>7d}"
        )

    # Completeness score
    key_fields = ["brief_title", "condition", "eligibility_criteria",
                   "gender", "minimum_age", "maximum_age", "phase"]
    key_present = sum(stats[f].get("present", 0) for f in key_fields if f in stats)
    key_total = sum(
        sum(stats[f].values()) for f in key_fields if f in stats
    )
    pct = 100 * key_present / key_total if key_total else 0

    print(f"\n  Key-field completeness: {key_present}/{key_total} ({pct:.1f}%)")
    print(f"  Key fields: {', '.join(key_fields)}")

    # Show IDs for requested fields
    if args.show_ids:
        print(f"\n{'=' * 78}")
        print("  Trials with missing/null/empty values:")
        print(f"{'=' * 78}")
        for field in args.show_ids:
            ids = missing_ids.get(field, [])
            if not ids:
                print(f"\n  {field}: all present")
            else:
                shown = ids[: args.top]
                extra = f"  ... and {len(ids) - len(shown)} more" if len(ids) > len(shown) else ""
                print(f"\n  {field} ({len(ids)} trials):")
                for tid in shown:
                    print(f"    - {tid}")
                if extra:
                    print(extra)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

