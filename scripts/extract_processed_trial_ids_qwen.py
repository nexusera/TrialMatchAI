#!/usr/bin/env python3
"""Extract trial IDs processed by scripts/evaluation/match_patients_trials_qwen.py.

By default, this script reads all JSON files in `results/qwen_patient_trial_matches`,
collects `trial_id` values from `all_assessments`, de-duplicates them, and writes one
trial ID per line.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


DEFAULT_INPUT = "results/qwen_patient_trial_matches"
DEFAULT_OUTPUT = "results/qwen_processed_trial_ids.txt"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")


def extract_ids_from_assessments(assessments: Iterable[Any]) -> Set[str]:
    ids: Set[str] = set()
    for item in assessments:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("trial_id") or "").strip()
        if tid:
            ids.add(tid)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract processed Qwen trial IDs into a text file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Result JSON file or directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output txt file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output)

    try:
        files = list_json_files(input_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not files:
        print(f"No JSON files found in {input_path}", file=sys.stderr)
        return 1

    all_ids: Set[str] = set()
    parsed_files = 0
    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            print(f"Skip {path.name}: {exc}", file=sys.stderr)
            continue

        assessments = data.get("all_assessments")
        if not isinstance(assessments, list):
            # Fallback: some files may only include matched_trials.
            assessments = data.get("matched_trials") or []
        all_ids.update(extract_ids_from_assessments(assessments))
        parsed_files += 1

    sorted_ids = sorted(all_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted_ids) + ("\n" if sorted_ids else ""), encoding="utf-8")

    print(f"Parsed files: {parsed_files}/{len(files)}")
    print(f"Unique trial IDs: {len(sorted_ids)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

