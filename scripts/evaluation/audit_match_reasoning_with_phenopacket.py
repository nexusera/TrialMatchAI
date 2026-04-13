#!/usr/bin/env python3
"""Audit Qwen match CoT criterion justifications against the source phenopacket.

Reads per-trial result JSON (or one aggregate patient JSON with ``all_assessments``),
calls an OpenAI-compatible Qwen endpoint, and asks the model whether each criterion's
reasoning is grounded in verbatim phenopacket text (with JSON-path style locations).

Outputs aggregate mismatch rate: count where ``aligned_with_phenopacket`` is false
over all criteria audited (optionally plus substring verification of quotes).

Examples::

  python scripts/evaluation/audit_match_reasoning_with_phenopacket.py \\
    --patient-results-dir ../../TrialMatchAI-exp/results5/.../covid19 \\
    --phenopacket example/covid19.json

  python scripts/evaluation/audit_match_reasoning_with_phenopacket.py \\
    --results-root ../../TrialMatchAI-exp/results5/.../ \\
    --phenopackets-dir example \\
    --max-result-files 5

  # One file from match_patients_trials_qwen (all_assessments + patient_file)
  python scripts/evaluation/audit_match_reasoning_with_phenopacket.py \\
    --aggregate-patient-json outputs/covid19.json \\
    --max-assessments 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Run as ``python scripts/evaluation/this.py`` — import sibling match_patients_trials_qwen.
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import match_patients_trials_qwen as mptq  # noqa: E402


def _get_item(d: Dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _iter_criteria_rows(cot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten inclusion/exclusion rows into list with kind + original index."""
    rows: List[Dict[str, Any]] = []
    inc = cot.get("Inclusion_Criteria_Evaluation") or cot.get(
        "inclusion_criteria_evaluation"
    )
    if isinstance(inc, list):
        for i, item in enumerate(inc):
            if isinstance(item, dict):
                rows.append({"kind": "inclusion", "index": i, "row": item})
    exc = cot.get("Exclusion_Criteria_Evaluation") or cot.get(
        "exclusion_criteria_evaluation"
    )
    if isinstance(exc, list):
        for i, item in enumerate(exc):
            if isinstance(item, dict):
                rows.append({"kind": "exclusion", "index": i, "row": item})
    return rows


def _reasoning_text(row: Dict[str, Any]) -> str:
    return _get_item(
        row,
        "Justification",
        "justification",
        "Reasoning",
        "reasoning",
        "Rationale",
        "rationale",
    )


def _evidence_field(row: Dict[str, Any]) -> str:
    return _get_item(row, "Evidence", "evidence")


def flatten_phenopacket_lines(obj: Any, prefix: str = "") -> List[str]:
    """Human-readable paths for auditing (JSON-pointer-like)."""
    lines: List[str] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            v = obj[k]
            p = f"{prefix}/{k}" if prefix else str(k)
            lines.extend(flatten_phenopacket_lines(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}/{i}"
            lines.extend(flatten_phenopacket_lines(v, p))
    else:
        if obj is None:
            return lines
        text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
        if str(text).strip():
            lines.append(f"{prefix} = {text}")
    return lines


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assessments_from_file(path: Path) -> List[Dict[str, Any]]:
    """Return one or more trial assessments from a results JSON file."""
    data = load_json(path)
    if isinstance(data.get("all_assessments"), list):
        out: List[Dict[str, Any]] = []
        for a in data["all_assessments"]:
            if isinstance(a, dict):
                tid = a.get("trial_id")
                if tid:
                    out.append(a)
        return out
    if isinstance(data, dict) and (
        data.get("cot_eligibility")
        or data.get("trial_id")
        or re.match(r"^NCT\d+$", path.stem, re.I)
    ):
        blob = dict(data)
        blob.setdefault("trial_id", data.get("trial_id") or path.stem)
        return [blob]
    return []


def resolve_phenopacket_path(
    patient_id: str,
    explicit: Optional[Path],
    phenopackets_dir: Optional[Path],
) -> Path:
    if explicit is not None:
        return explicit
    if phenopackets_dir is None:
        raise ValueError(
            "Need --phenopacket or --phenopackets-dir to locate phenopacket JSON."
        )
    cand = phenopackets_dir / f"{patient_id}.json"
    if not cand.is_file():
        raise FileNotFoundError(f"Phenopacket not found for patient_id={patient_id}: {cand}")
    return cand


def build_audit_prompt(
    phenopacket_compact: str,
    phenopacket_flat: str,
    trial_id: str,
    criteria_payload: List[Dict[str, Any]],
    *,
    max_compact_chars: int,
    max_flat_chars: int,
) -> Tuple[str, str]:
    system = (
        "You are a strict clinical informatics auditor. "
        "You must compare each eligibility criterion assessment (criterion text, classification, "
        "and written reasoning) against ONLY the phenopacket JSON provided. "
        "Output exactly one JSON object; no markdown or extra text. "
        "A criterion is aligned_with_phenopacket=true only if every factual claim in the reasoning "
        "that is presented as patient fact is directly supported by verbatim text in the phenopacket "
        "(same wording or clear exact substring from a string field). "
        "If the reasoning invents facts, misquotes, or cites information not present, set false. "
        "If the reasoning correctly states that information is missing/unclear in the phenopacket, "
        "that is aligned (true); use empty verbatim_quotes and explain. "
        "For each criterion, give json_pointer paths (RFC 6901, use / separators) pointing to "
        "the smallest object/array element that contains the supporting quote, and include "
        "verbatim_quotes copied exactly from the phenopacket for the evidence chain. "
        "You MUST return exactly one audit object per input row, with matching kind and index."
    )
    compact_block = phenopacket_compact
    if max_compact_chars > 0 and len(compact_block) > max_compact_chars:
        compact_block = (
            f"[OMITTED: compact JSON length {len(phenopacket_compact)} > "
            f"{max_compact_chars}; rely on flattened lines below.]\n"
        )
    flat_use = phenopacket_flat
    if max_flat_chars > 0 and len(flat_use) > max_flat_chars:
        flat_use = (
            flat_use[:max_flat_chars] + "\n... [truncated flat phenopacket]\n"
        )
    user_parts = [
        f"trial_id: {trial_id}",
        "",
        "### phenopacket (compact JSON)",
        compact_block,
        "",
        "### phenopacket (flattened leaf lines; same data, for easier path checking)",
        flat_use,
        "",
        "### criteria to audit (from matcher output; do not trust unsourced claims in reasoning)",
        json.dumps(criteria_payload, ensure_ascii=False, indent=2),
        "",
        "Return JSON with this shape:",
        json.dumps(
            {
                "audits": [
                    {
                        "kind": "inclusion|exclusion",
                        "index": 0,
                        "aligned_with_phenopacket": True,
                        "classification_plausible_given_phenopacket": True,
                        "json_pointers": ["/subject/sex"],
                        "verbatim_quotes": ["MALE"],
                        "explanation": "short",
                    }
                ]
            },
            indent=2,
        ),
    ]
    return system, "\n".join(user_parts)


def quote_verified_in_phenopacket(
    quotes: List[str],
    phenopacket_raw: str,
    *,
    normalize_ws: bool = False,
) -> Tuple[bool, List[str]]:
    """Heuristic: each non-empty quote must appear as substring in raw JSON text."""
    missing: List[str] = []
    raw = phenopacket_raw
    if normalize_ws:
        raw = re.sub(r"\s+", " ", raw)
    for q in quotes:
        q = (q or "").strip()
        if not q:
            continue
        probe = re.sub(r"\s+", " ", q) if normalize_ws else q
        if probe not in raw:
            missing.append(q[:200] + ("..." if len(q) > 200 else ""))
    ok = len(missing) == 0
    return ok, missing


def _normalize_audit_key(a: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    k = a.get("kind")
    if not isinstance(k, str):
        return None
    k = k.strip().lower()
    if k not in ("inclusion", "exclusion"):
        return None
    try:
        idx = int(a.get("index"))
    except (TypeError, ValueError):
        return None
    return k, idx


def _align_audits_to_criteria(
    expected: List[Dict[str, Any]], audits: List[Any]
) -> List[Optional[Dict[str, Any]]]:
    """Map each expected criterion row to an audit dict (or None if missing).

    Prefer (kind, index) keys from the model; otherwise consume unused rows in
    response order. Never assign the same audit dict to two criteria.
    """
    ordered: List[Dict[str, Any]] = [
        a for a in (audits or []) if isinstance(a, dict)
    ]
    by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for a in ordered:
        key = _normalize_audit_key(a)
        if key is not None:
            by_key[key] = a

    used: set[int] = set()
    seq_i = 0
    out: List[Optional[Dict[str, Any]]] = []

    def _next_sequential() -> Optional[Dict[str, Any]]:
        nonlocal seq_i
        while seq_i < len(ordered):
            cand = ordered[seq_i]
            seq_i += 1
            if id(cand) not in used:
                return cand
        return None

    for exp in expected:
        k = exp["kind"]
        idx = exp["index"]
        hit = by_key.get((k, idx))
        if hit is not None and id(hit) not in used:
            out.append(hit)
            used.add(id(hit))
            continue
        nxt = _next_sequential()
        out.append(nxt)
        if nxt is not None:
            used.add(id(nxt))

    return out


def audit_one_trial(
    client: mptq.OpenAICompatClient,
    trial_id: str,
    cot: Dict[str, Any],
    phenopacket: Dict[str, Any],
    *,
    max_tokens: int,
    max_compact_chars: int,
    max_flat_chars: int,
    normalize_quote_ws: bool,
) -> Dict[str, Any]:
    rows = _iter_criteria_rows(cot)
    criteria_payload = []
    for r in rows:
        item = r["row"]
        criteria_payload.append(
            {
                "kind": r["kind"],
                "index": r["index"],
                "criterion": _get_item(item, "Criterion", "criterion"),
                "classification": _get_item(item, "Classification", "classification"),
                "reasoning": _reasoning_text(item),
                "original_evidence_field": _evidence_field(item),
            }
        )
    compact = json.dumps(phenopacket, ensure_ascii=False)
    flat = "\n".join(flatten_phenopacket_lines(phenopacket))
    system, user = build_audit_prompt(
        compact,
        flat,
        trial_id,
        criteria_payload,
        max_compact_chars=max_compact_chars,
        max_flat_chars=max_flat_chars,
    )
    parsed = client.chat_json(system, user, max_tokens=max_tokens)
    audits_raw = parsed.get("audits")
    if not isinstance(audits_raw, list):
        raise ValueError(f"Unexpected API JSON (no audits list): keys={list(parsed.keys())}")

    aligned = _align_audits_to_criteria(criteria_payload, audits_raw)

    verified_rows: List[Dict[str, Any]] = []
    for exp, a in zip(criteria_payload, aligned):
        if a is None:
            verified_rows.append(
                {
                    "kind": exp["kind"],
                    "index": exp["index"],
                    "aligned_with_phenopacket": False,
                    "classification_plausible_given_phenopacket": False,
                    "json_pointers": [],
                    "verbatim_quotes": [],
                    "explanation": "auditor returned no row for this criterion",
                    "verbatim_quote_substring_match": False,
                    "missing_substrings": [],
                }
            )
            continue
        quotes = a.get("verbatim_quotes") or []
        if not isinstance(quotes, list):
            quotes = []
        qlist = [str(x) for x in quotes]
        ok_sub, missing = quote_verified_in_phenopacket(
            qlist, compact, normalize_ws=normalize_quote_ws
        )
        merged = dict(a)
        merged.setdefault("kind", exp["kind"])
        merged.setdefault("index", exp["index"])
        merged["verbatim_quote_substring_match"] = ok_sub
        merged["missing_substrings"] = missing
        verified_rows.append(merged)

    return {
        "trial_id": trial_id,
        "criteria_count": len(criteria_payload),
        "audits": verified_rows,
        "raw_response_keys": list(parsed.keys()),
        "prompt_meta": {
            "phenopacket_compact_chars": len(compact),
            "phenopacket_flat_chars": len(flat),
            "max_compact_chars": max_compact_chars,
            "max_flat_chars": max_flat_chars,
        },
    }


def _call_with_retries(
    fn,
    *,
    max_retries: int,
    backoff: float,
    max_backoff: float,
):
    delay = backoff
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_backoff) if max_backoff > 0 else delay * 2
    assert last_exc is not None
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit matcher criterion reasoning against phenopacket via Qwen API."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--patient-results-dir",
        type=Path,
        help="Directory of per-trial JSON files for one patient (e.g. .../covid19/).",
    )
    src.add_argument(
        "--results-root",
        type=Path,
        help="Parent directory containing one subdirectory per patient with NCT*.json files.",
    )
    src.add_argument(
        "--aggregate-patient-json",
        type=Path,
        help=(
            "Single patient output from match_patients_trials_qwen (all_assessments). "
            "Uses patient_file as phenopacket when that path exists."
        ),
    )
    parser.add_argument(
        "--phenopacket",
        type=Path,
        help="Explicit phenopacket JSON path (when using --patient-results-dir).",
    )
    parser.add_argument(
        "--phenopackets-dir",
        type=Path,
        help="Directory of {patient_id}.json phenopackets (with --results-root).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="",
        help="Override patient id for phenopacket resolution (default: results folder name).",
    )
    parser.add_argument("--base-url", default=mptq.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=mptq.DEFAULT_API_KEY)
    parser.add_argument("--model", default=mptq.DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--provider-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--max-result-files",
        type=int,
        default=0,
        help="Process at most this many per-trial JSON paths (0 = no limit). Ignored for --aggregate-patient-json.",
    )
    parser.add_argument(
        "--max-assessments",
        type=int,
        default=0,
        help="Cap total API audits across all inputs (each trial's criteria = 1 request). 0 = no limit.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--retry-max-backoff-seconds", type=float, default=60.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full audit report JSON to this path.",
    )
    parser.add_argument(
        "--require-quote-substring",
        action="store_true",
        help="Also count as mismatch when verbatim_quotes are not substrings of phenopacket JSON.",
    )
    parser.add_argument(
        "--max-phenopacket-compact-chars",
        type=int,
        default=120_000,
        help=(
            "Omit full compact JSON from the prompt when longer than this (0 = never omit). "
            "Flat lines still carry the same information."
        ),
    )
    parser.add_argument(
        "--max-phenopacket-flat-chars",
        type=int,
        default=200_000,
        help="Truncate flattened phenopacket block in the prompt (0 = no truncation).",
    )
    parser.add_argument(
        "--normalize-quote-whitespace",
        action="store_true",
        help="When verifying verbatim_quotes, collapse whitespace in both quote and phenopacket JSON.",
    )
    args = parser.parse_args()

    if args.results_root and not args.phenopackets_dir:
        parser.error("--results-root requires --phenopackets-dir (e.g. example/).")
    if args.patient_results_dir and not (args.phenopacket or args.phenopackets_dir):
        parser.error(
            "--patient-results-dir requires --phenopacket or --phenopackets-dir."
        )
    client = mptq.OpenAICompatClient(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        provider_timeout_seconds=args.provider_timeout_seconds,
    )

    jobs: List[Tuple[str, Path, Path]] = []
    # (patient_id, trial_result_path, phenopacket_path)

    if args.patient_results_dir:
        pdir = args.patient_results_dir.resolve()
        pid = args.patient_id or pdir.name
        pp_path = resolve_phenopacket_path(
            pid,
            args.phenopacket.resolve() if args.phenopacket else None,
            args.phenopackets_dir.resolve() if args.phenopackets_dir else None,
        )
        files = sorted(p for p in pdir.glob("*.json") if p.is_file())
        for p in files:
            jobs.append((pid, p, pp_path))
    elif args.results_root:
        root = args.results_root.resolve()
        if not root.is_dir():
            parser.error(f"--results-root is not a directory: {root}")
        for sub in sorted(d for d in root.iterdir() if d.is_dir()):
            pid = args.patient_id or sub.name
            try:
                pp_path = resolve_phenopacket_path(
                    pid,
                    None,
                    args.phenopackets_dir.resolve() if args.phenopackets_dir else None,
                )
            except FileNotFoundError:
                print(f"[skip] no phenopacket for patient_id={pid}", file=sys.stderr)
                continue
            for p in sorted(sub.glob("*.json")):
                jobs.append((pid, p, pp_path))
    else:
        agg = args.aggregate_patient_json.resolve()
        if not agg.is_file():
            parser.error(f"--aggregate-patient-json not found: {agg}")
        agg_data = load_json(agg)
        pid = args.patient_id or str(
            agg_data.get("patient_id") or mptq.patient_id_from_path(agg, agg_data)
        )
        pp_path: Optional[Path] = None
        pf = agg_data.get("patient_file")
        if isinstance(pf, str) and pf.strip():
            cand = Path(pf).expanduser()
            if cand.is_file():
                pp_path = cand.resolve()
        if pp_path is None:
            try:
                pp_path = resolve_phenopacket_path(
                    pid,
                    args.phenopacket.resolve() if args.phenopacket else None,
                    args.phenopackets_dir.resolve() if args.phenopackets_dir else None,
                )
            except (FileNotFoundError, ValueError) as exc:
                parser.error(
                    "Could not resolve phenopacket for aggregate file: "
                    f"set --phenopacket or --phenopackets-dir, or valid patient_file. ({exc})"
                )
        jobs.append((pid, agg, pp_path))

    if args.max_result_files > 0:
        jobs = jobs[: args.max_result_files]

    total_criteria = 0
    total_mismatch_model = 0
    total_mismatch_quote = 0
    total_skipped_no_cot = 0
    report: Dict[str, Any] = {
        "jobs": [],
        "totals": {},
    }

    phenopacket_cache: Dict[str, Dict[str, Any]] = {}
    assessments_budget = (
        args.max_assessments if args.max_assessments > 0 else None
    )

    for patient_id, trial_path, pp_path in jobs:
        key = str(pp_path)
        if key not in phenopacket_cache:
            phenopacket_cache[key] = load_json(pp_path)

        phenopacket = phenopacket_cache[key]
        for assessment in assessments_from_file(trial_path):
            if assessments_budget is not None and assessments_budget <= 0:
                break
            if assessment.get("api_error"):
                continue
            cot = assessment.get("cot_eligibility")
            if not isinstance(cot, dict):
                total_skipped_no_cot += 1
                continue
            trial_id = str(assessment.get("trial_id") or trial_path.stem)
            n_crit = len(_iter_criteria_rows(cot))
            if n_crit == 0:
                total_skipped_no_cot += 1
                continue

            if assessments_budget is not None:
                assessments_budget -= 1

            def _run():
                return audit_one_trial(
                    client,
                    trial_id,
                    cot,
                    phenopacket,
                    max_tokens=args.max_tokens,
                    max_compact_chars=args.max_phenopacket_compact_chars,
                    max_flat_chars=args.max_phenopacket_flat_chars,
                    normalize_quote_ws=args.normalize_quote_whitespace,
                )

            try:
                audit = _call_with_retries(
                    _run,
                    max_retries=args.max_retries,
                    backoff=args.retry_backoff_seconds,
                    max_backoff=args.retry_max_backoff_seconds,
                )
            except Exception as exc:
                print(f"[error] {patient_id} {trial_id}: {exc}", file=sys.stderr)
                report["jobs"].append(
                    {
                        "patient_id": patient_id,
                        "trial_id": trial_id,
                        "trial_path": str(trial_path),
                        "error": str(exc),
                    }
                )
                continue

            mism_model = 0
            mism_quote = 0
            for row in audit.get("audits", []):
                if not row.get("aligned_with_phenopacket", True):
                    mism_model += 1
                if args.require_quote_substring and not row.get(
                    "verbatim_quote_substring_match", True
                ):
                    mism_quote += 1

            total_criteria += n_crit
            total_mismatch_model += mism_model
            if args.require_quote_substring:
                total_mismatch_quote += mism_quote

            entry = {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "trial_path": str(trial_path),
                "phenopacket": str(pp_path),
                "criteria_count": n_crit,
                "audits_returned": len(audit.get("audits") or []),
                "mismatch_count_model_aligned_flag": mism_model,
                "audit": audit,
            }
            if args.require_quote_substring:
                entry["mismatch_count_quote_substring"] = mism_quote
            report["jobs"].append(entry)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    denom = total_criteria if total_criteria else 0
    report["totals"] = {
        "criteria_audited": total_criteria,
        "mismatch_count_aligned_flag": total_mismatch_model,
        "mismatch_rate_aligned_flag": (total_mismatch_model / denom) if denom else None,
        "skipped_no_cot": total_skipped_no_cot,
    }
    if args.require_quote_substring:
        report["totals"]["mismatch_count_quote_substring"] = total_mismatch_quote
        report["totals"]["mismatch_rate_quote_substring"] = (
            (total_mismatch_quote / denom) if denom else None
        )

    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    if args.out_json:
        mptq.dump_json(args.out_json, report)
        print(f"Wrote {args.out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
