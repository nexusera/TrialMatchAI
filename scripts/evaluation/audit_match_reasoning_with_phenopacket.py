#!/usr/bin/env python3
"""Audit Qwen match CoT criterion justifications against the source phenopacket.

Reads per-trial JSON in several layouts:

- **CoT-only file** (common under ``<patient_dir>/NCTxxxxxx.json``): the root object is exactly
  the eligibility JSON, e.g. only ``Inclusion_Criteria_Evaluation`` and
  ``Exclusion_Criteria_Evaluation`` (plus optional ``Recap`` / ``Final Decision``). Trial id
  is taken from the filename stem.

- **Patient results directory** (``--patient-results-dir`` / each folder under ``--results-root``):
  by default, **only** patients folders that contain ``ranked_trials.json`` are processed: the
  script schedules ``{trial_id}.json`` for each listed id **in that order** (any id prefix).
  If there is **no** ``ranked_trials.json``, or it has no parseable trial ids, that folder is
  **skipped** (no audit). Use ``--no-ranked-trials-json`` to ignore the ranked file and instead
  schedule all non-metadata ``*.json`` in the folder (sorted by name).

  With ``--results-root``: if the path you pass **directly** contains ``ranked_trials.json``, only
  that folder is used (subdirectories are not scanned as separate patients). If it does not, each
  immediate child directory (except ``__pycache__`` and dot-folders) is treated as a patient folder.
  **Relative paths depend on your current working directory** when you run the script.

- **Wrapped assessment**: same CoT nested under ``cot_eligibility`` (``match_patients_trials_qwen.py``
  with ``--use-cot-reasoning``), or Matcher-style top-level lists next to other keys.

- **Aggregate patient file**: top-level ``all_assessments`` list (one ``<patient_id>.json``).
  Those rows must come from ``match_patients_trials_qwen.py`` **with** ``--use-cot-reasoning``
  so each item includes a ``cot_eligibility`` object (or top-level inclusion/exclusion lists).
  Plain (non-CoT) match JSON has no per-criterion text to audit.

calls an OpenAI-compatible Qwen endpoint. For each row it passes **criterion** (trial
text), **justification** (CoT reasoning), and **classification** together with the
phenopacket, and asks whether (1) patient facts claimed in the justification are grounded
in the phenopacket, and (2) the eligibility **classification** is reasonable given the
**criterion wording** plus those facts.

Outputs aggregate mismatch rates: default count where ``aligned_with_phenopacket`` is
false; totals also report failures of ``classification_plausible_given_phenopacket`` and
either flag (optionally plus substring verification of quotes).

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
import urllib.request
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


def _inclusion_criteria_list(d: Dict[str, Any]) -> Optional[List[Any]]:
    """Return the inclusion list if present (including empty ``[]``); else None.

    Do not use ``a or b`` here: ``[]`` is falsy and would incorrectly fall through.
    """
    for k in ("Inclusion_Criteria_Evaluation", "inclusion_criteria_evaluation"):
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, list):
            return v
    return None


def _exclusion_criteria_list(d: Dict[str, Any]) -> Optional[List[Any]]:
    for k in ("Exclusion_Criteria_Evaluation", "exclusion_criteria_evaluation"):
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, list):
            return v
    return None


def _looks_like_cot_eligibility_shape(d: Dict[str, Any]) -> bool:
    """True if *d* carries matcher-style inclusion/exclusion lists (any list, even empty)."""
    return _inclusion_criteria_list(d) is not None or _exclusion_criteria_list(
        d
    ) is not None


def _normalize_cot_eligibility_value(ce: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ce, dict):
        return ce
    if isinstance(ce, str) and ce.strip():
        try:
            parsed = json.loads(ce)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _find_nested_cot_dict(
    root: Dict[str, Any], *, max_depth: int = 8
) -> Optional[Dict[str, Any]]:
    """Depth-first search for a dict that carries CoT lists (non-empty rows preferred)."""
    best_empty: Optional[Dict[str, Any]] = None
    stack: List[Tuple[Dict[str, Any], int]] = [(root, 0)]
    seen: set[int] = set()
    while stack:
        d, depth = stack.pop()
        i = id(d)
        if i in seen or not isinstance(d, dict):
            continue
        seen.add(i)
        if _iter_criteria_rows(d):
            return d
        if _looks_like_cot_eligibility_shape(d):
            best_empty = d
        if depth >= max_depth:
            continue
        for v in d.values():
            if isinstance(v, dict):
                stack.append((v, depth + 1))
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        stack.append((it, depth + 1))
    return best_empty


def extract_cot_dict_from_assessment(assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """CoT may live in ``cot_eligibility`` (qwen script) or at top level (Matcher per-trial JSON)."""
    candidates: List[Dict[str, Any]] = []
    ce_d = _normalize_cot_eligibility_value(assessment.get("cot_eligibility"))
    if ce_d is not None:
        candidates.append(ce_d)
    if _looks_like_cot_eligibility_shape(assessment):
        candidates.append(assessment)

    seen_ids = {id(c) for c in candidates}
    nested = _find_nested_cot_dict(assessment)
    if nested is not None and id(nested) not in seen_ids:
        candidates.append(nested)

    for cand in candidates:
        if _iter_criteria_rows(cand):
            return cand
    for cand in candidates:
        if _looks_like_cot_eligibility_shape(cand):
            return cand
    return None


def _iter_criteria_rows(cot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten inclusion/exclusion rows into list with kind + original index."""
    rows: List[Dict[str, Any]] = []
    inc = _inclusion_criteria_list(cot)
    if inc is not None:
        for i, item in enumerate(inc):
            if isinstance(item, dict):
                rows.append({"kind": "inclusion", "index": i, "row": item})
    exc = _exclusion_criteria_list(cot)
    if exc is not None:
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


def _trial_id_from_assessment_row(a: Dict[str, Any]) -> str:
    """trial_id in matcher output; some pipelines only set nct_id."""
    tid = a.get("trial_id") or a.get("nct_id") or a.get("trialId")
    if tid is not None and str(tid).strip():
        return str(tid).strip()
    return ""


def assessments_from_file(path: Path) -> List[Dict[str, Any]]:
    """Return one or more trial assessments from a results JSON file.

    CoT-only roots (lists at top level) become a single pseudo-assessment; ``trial_id`` is
    set from ``trial_id`` / ``nct_id`` in JSON or from ``NCT*`` in the file stem.
    """
    data = load_json(path)
    if isinstance(data.get("all_assessments"), list):
        out: List[Dict[str, Any]] = []
        for i, a in enumerate(data["all_assessments"]):
            if not isinstance(a, dict):
                continue
            tid = _trial_id_from_assessment_row(a)
            blob = dict(a)
            if not tid:
                if extract_cot_dict_from_assessment(a) is not None:
                    tid = f"all_assessments[{i}]"
                else:
                    continue
            blob.setdefault("trial_id", tid)
            out.append(blob)
        return out
    stem_nct = re.search(r"(NCT\d+)", path.stem, flags=re.I)
    stem_nct_s = stem_nct.group(1).upper() if stem_nct else ""
    if isinstance(data, dict) and (
        data.get("cot_eligibility")
        or data.get("trial_id")
        or data.get("nct_id")
        or re.match(r"^NCT\d+$", path.stem, re.I)
        or _looks_like_cot_eligibility_shape(data)
    ):
        blob = dict(data)
        blob.setdefault(
            "trial_id",
            str(
                data.get("trial_id")
                or data.get("nct_id")
                or stem_nct_s
                or path.stem
            ),
        )
        return [blob]
    return []


def count_audit_runs_scheduled(
    jobs: List[Tuple[str, Path, Path]], max_assessments: int
) -> int:
    """How many ``audit_one_trial`` API runs the main loop will perform (same skips / budget)."""
    budget = max_assessments if max_assessments > 0 else None
    n = 0
    for _patient_id, trial_path, _pp_path in jobs:
        extracted = assessments_from_file(trial_path)
        for assessment in extracted:
            if budget is not None and budget <= 0:
                return n
            if assessment.get("api_error"):
                continue
            cot = extract_cot_dict_from_assessment(assessment)
            if cot is None:
                continue
            if not _iter_criteria_rows(cot):
                continue
            if budget is not None:
                budget -= 1
            n += 1
    return n


def _phenopacket_stem(patient_id: str) -> str:
    s = str(patient_id).strip()
    if s.lower().endswith(".json"):
        return s[: -len(".json")]
    return s


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
    cand = phenopackets_dir / f"{_phenopacket_stem(patient_id)}.json"
    if not cand.is_file():
        raise FileNotFoundError(f"Phenopacket not found for patient_id={patient_id}: {cand}")
    return cand


# JSON files in a patient folder that are not per-trial matcher outputs.
_METADATA_RESULT_JSON_STEMS = frozenset(
    {
        "ranked_trials",
        "rankedtrials",
        "second_level_trial_scores",
    }
)


def _json_stem_is_metadata(stem: str) -> bool:
    return stem.lower().replace("-", "_") in _METADATA_RESULT_JSON_STEMS


def _trial_id_from_ranked_value(raw: Any) -> Optional[str]:
    """Normalize a trial id from ranked_trials for filename lookup (any prefix: NCT, CTR, ChiCTR, …)."""
    s = str(raw).strip()
    if s.lower().endswith(".json"):
        s = s[:-5].strip()
    if not s or len(s) > 200:
        return None
    if any(c in s for c in ("/", "\\", ":", "\0")):
        return None
    if s in (".", ".."):
        return None
    return s.upper()


def _patient_result_subdirs(root: Path) -> List[Path]:
    """Immediate child dirs that could be per-patient output folders (noise removed)."""
    skip = frozenset({"__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"})
    out: List[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name in skip:
            continue
        out.append(d)
    return sorted(out, key=lambda p: p.name.lower())


def _find_ranked_trials_json_in_dir(pdir: Path) -> Optional[Path]:
    """Matcher / evaluate_ranked_trials layout: ``ranked_trials.json`` (stem match, case-insensitive)."""
    for p in pdir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        stem = p.stem.lower().replace("-", "_")
        if stem in ("ranked_trials", "rankedtrials"):
            return p
    return None


def _trial_ids_from_ranked_json(path: Path) -> List[str]:
    """Ordered trial IDs from ``RankedTrials`` / ``ranked_trials`` (etc.), deduplicated.

    IDs are not restricted to NCT/CTR; any non-empty safe string from the list is kept.
    """
    raw = load_json(path)
    ranked: List[Any]
    if isinstance(raw, dict):
        ranked = []
        for k in ("RankedTrials", "ranked_trials", "ranked", "trials"):
            v = raw.get(k)
            if isinstance(v, list):
                ranked = v
                break
    elif isinstance(raw, list):
        ranked = raw
    else:
        ranked = []

    ids: List[str] = []
    seen: set[str] = set()
    if not ranked:
        return ids

    first = ranked[0]
    if isinstance(first, str):
        for s in ranked:
            u = _trial_id_from_ranked_value(s)
            if u and u not in seen:
                seen.add(u)
                ids.append(u)
        return ids

    if isinstance(first, dict):
        for r in ranked:
            if not isinstance(r, dict):
                continue
            nid: Optional[str] = None
            for key in (
                "TrialID",
                "trial_id",
                "trialId",
                "nct_id",
                "nctId",
                "id",
                "corpus_id",
                "corpusId",
                "docid",
                "doc_id",
            ):
                if key in r and r[key] is not None:
                    nid = _trial_id_from_ranked_value(r[key])
                    break
            if nid and nid not in seen:
                seen.add(nid)
                ids.append(nid)
        return ids

    if isinstance(first, (list, tuple)):
        for item in ranked:
            if not item:
                continue
            nid = _trial_id_from_ranked_value(item[0])
            if nid and nid not in seen:
                seen.add(nid)
                ids.append(nid)
    return ids


def _resolve_per_trial_json(pdir: Path, trial_id: str) -> Optional[Path]:
    """Return path to existing ``{trial_id}.json`` (case-insensitive stem on disk)."""
    want = trial_id.upper()
    direct = pdir / f"{want}.json"
    if direct.is_file():
        return direct
    for p in pdir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        if p.stem.upper() == want:
            return p
    return None


def collect_per_trial_json_jobs_for_patient_dir(
    pdir: Path,
    pid: str,
    pp_path: Path,
    *,
    use_ranked_trials_json: bool,
) -> Tuple[List[Tuple[str, Path, Path]], Dict[str, Any]]:
    """Build (patient_id, trial_json_path, phenopacket_path) jobs for one patient folder.

    When *use_ranked_trials_json* is true (default): require ``ranked_trials.json``. If it is
    missing, or contains no parseable trial ids, return **no jobs**. Otherwise enqueue
    ``{trial_id}.json`` for each listed id in order.

    When *use_ranked_trials_json* is false (``--no-ranked-trials-json``): enqueue every
    ``*.json`` except known metadata files (sorted by path name).
    """
    meta: Dict[str, Any] = {
        "patient_id": pid,
        "ranked_trials_json": None,
        "trial_ids_from_ranked": 0,
        "missing_per_trial_json": [],
        "skipped_no_ranked_trials_json": False,
        "skipped_ranked_empty_or_unparsed": False,
    }
    jobs: List[Tuple[str, Path, Path]] = []

    if use_ranked_trials_json:
        ranked_path = _find_ranked_trials_json_in_dir(pdir)
        if ranked_path is None:
            meta["skipped_no_ranked_trials_json"] = True
            print(
                f"[skip] {pid}: no ranked_trials.json under {pdir}",
                file=sys.stderr,
            )
            return [], meta
        meta["ranked_trials_json"] = str(ranked_path)
        tids = _trial_ids_from_ranked_json(ranked_path)
        meta["trial_ids_from_ranked"] = len(tids)
        if not tids:
            meta["skipped_ranked_empty_or_unparsed"] = True
            print(
                f"[skip] {pid}: {ranked_path.name} has no parseable trial ids",
                file=sys.stderr,
            )
            return [], meta
        for tid in tids:
            p = _resolve_per_trial_json(pdir, tid)
            if p is None:
                meta["missing_per_trial_json"].append(tid)
                continue
            jobs.append((pid, p.resolve(), pp_path))
        return jobs, meta

    for p in sorted(pdir.glob("*.json")):
        if not p.is_file() or _json_stem_is_metadata(p.stem):
            continue
        jobs.append((pid, p.resolve(), pp_path))

    return jobs, meta


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
        "Each input row has: kind/index, criterion (verbatim trial inclusion or exclusion text), "
        "justification (the model's written reasoning for this row), classification (e.g. PASS/FAIL/UNKNOWN), "
        "and optionally evidence_from_matcher. Use ONLY the phenopacket JSON provided—do not use outside knowledge. "
        "Output exactly one JSON object; no markdown or extra text. "
        "\n"
        "1) aligned_with_phenopacket: true only if every factual claim in the **justification** that is "
        "presented as patient fact is directly supported by verbatim text in the phenopacket "
        "(same wording or clear exact substring from a string field). "
        "If the justification invents facts, misquotes, or cites information not present, set false. "
        "If it correctly states that information is missing/unclear in the phenopacket, set true; "
        "use empty verbatim_quotes and explain. "
        "\n"
        "2) classification_plausible_given_phenopacket: true only if the **classification** is a reasonable "
        "eligibility conclusion when you jointly consider (a) the **criterion** wording, (b) what the "
        "phenopacket actually says, and (c) whether the **justification** coherently connects (a) to (b). "
        "The justification should address the requirement expressed in the criterion, not unrelated facts. "
        "If the classification contradicts the phenopacket or ignores an explicit fact that decides the criterion, "
        "set false. "
        "\n"
        "For each row, give json_pointer paths (RFC 6901, / separators) to the smallest object/array element "
        "that contains supporting quotes, and verbatim_quotes copied exactly from the phenopacket. "
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
        "### criteria to audit (criterion + justification + classification from matcher; "
        "do not trust unsourced patient claims in justification)",
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
                        "explanation": "How criterion+justification relate to phenopacket; cite gaps if any.",
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
    max_criteria_per_api_call: int = 40,
) -> Dict[str, Any]:
    rows = _iter_criteria_rows(cot)
    criteria_payload = []
    for r in rows:
        item = r["row"]
        justification = _reasoning_text(item)
        criteria_payload.append(
            {
                "kind": r["kind"],
                "index": r["index"],
                "criterion": _get_item(item, "Criterion", "criterion"),
                "justification": justification,
                "classification": _get_item(item, "Classification", "classification"),
                "evidence_from_matcher": _evidence_field(item),
            }
        )
    compact = json.dumps(phenopacket, ensure_ascii=False)
    flat = "\n".join(flatten_phenopacket_lines(phenopacket))

    limit = max_criteria_per_api_call
    if limit <= 0 or len(criteria_payload) <= limit:
        chunks = [criteria_payload]
    else:
        chunks = [
            criteria_payload[i : i + limit]
            for i in range(0, len(criteria_payload), limit)
        ]

    merged_audits: List[Any] = []
    last_keys: List[str] = []
    for chunk in chunks:
        system, user = build_audit_prompt(
            compact,
            flat,
            trial_id,
            chunk,
            max_compact_chars=max_compact_chars,
            max_flat_chars=max_flat_chars,
        )
        parsed = client.chat_json(system, user, max_tokens=max_tokens)
        last_keys = list(parsed.keys())
        audits_raw = parsed.get("audits")
        if not isinstance(audits_raw, list):
            raise ValueError(
                f"Unexpected API JSON (no audits list): keys={last_keys}"
            )
        merged_audits.extend(audits_raw)

    aligned = _align_audits_to_criteria(criteria_payload, merged_audits)

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
                    "audit_reasonable_overall": False,
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
        merged.setdefault("aligned_with_phenopacket", False)
        merged.setdefault("classification_plausible_given_phenopacket", False)
        merged["verbatim_quote_substring_match"] = ok_sub
        merged["missing_substrings"] = missing
        merged["audit_reasonable_overall"] = bool(
            merged.get("aligned_with_phenopacket")
            and merged.get("classification_plausible_given_phenopacket")
        )
        verified_rows.append(merged)

    return {
        "trial_id": trial_id,
        "criteria_count": len(criteria_payload),
        "audits": verified_rows,
        "raw_response_keys": last_keys,
        "api_batch_count": len(chunks),
        "prompt_meta": {
            "phenopacket_compact_chars": len(compact),
            "phenopacket_flat_chars": len(flat),
            "max_compact_chars": max_compact_chars,
            "max_flat_chars": max_flat_chars,
            "max_criteria_per_api_call": limit,
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


def _verify_openai_compat_via_chat(client: mptq.OpenAICompatClient) -> None:
    """Probe chat/completions when GET /models is not supported."""
    client.chat_json(
        "Reply with a single JSON object only.",
        'Respond with exactly: {"api_check": true}',
        max_tokens=64,
    )


def verify_openai_compat_api_available(client: mptq.OpenAICompatClient) -> None:
    """Raise RuntimeError if the OpenAI-compatible API is not reachable or rejects the request.

    Tries GET ``{base_url}/models`` first (common for OpenAI-compatible servers). If that returns
    404/405, falls back to a minimal ``chat_json`` round-trip on the same base URL the audit uses.
    """
    base = client.base_url.rstrip("/")
    models_url = f"{base}/models"
    headers = {"Authorization": f"Bearer {client.api_key}"}
    timeout = min(30, max(5, int(client.timeout)))
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                print(f"[info] API check OK (GET {models_url})", file=sys.stderr)
                return
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405):
            print(
                f"[info] GET /models not supported (HTTP {exc.code}); "
                "probing chat/completions…",
                file=sys.stderr,
            )
            try:
                _verify_openai_compat_via_chat(client)
            except urllib.error.HTTPError as e2:
                raise RuntimeError(mptq.format_error(e2)) from e2
            except urllib.error.URLError as e2:
                raise RuntimeError(mptq.format_error(e2)) from e2
            except (TimeoutError, OSError, json.JSONDecodeError, ValueError) as e2:
                raise RuntimeError(f"{type(e2).__name__}: {e2}") from e2
            print(
                f"[info] API check OK (chat/completions at {client.base_url})",
                file=sys.stderr,
            )
            return
        raise RuntimeError(mptq.format_error(exc)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"API unreachable at {base}: {type(exc).__name__}: {exc}"
        ) from exc

    raise RuntimeError(f"API check failed: unexpected response from {models_url}")


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
        help=(
            "Directory that either (A) contains ranked_trials.json itself (single patient folder), "
            "or (B) is a parent whose immediate subfolders are patients (each with ranked_trials.json). "
            "Relative paths are resolved from the shell cwd. Ignores __pycache__ and dot-dirs as patients."
        ),
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
        "--max-criteria-per-api-call",
        type=int,
        default=40,
        help=(
            "When a trial has more criterion rows than this, run multiple auditor API calls "
            "(smaller JSON per response, avoids truncated/invalid JSON). 0 = one call with all rows."
        ),
    )
    parser.add_argument(
        "--no-ranked-trials-json",
        action="store_true",
        help=(
            "Do not require ranked_trials.json: schedule all non-metadata *.json in each patient "
            "folder (sorted by name). Default is to skip folders without ranked_trials.json."
        ),
    )
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
        "--no-progress",
        action="store_true",
        help="Do not print progress lines to stderr (default: show job and API-audit progress).",
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
    try:
        verify_openai_compat_api_available(client)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    jobs: List[Tuple[str, Path, Path]] = []
    # (patient_id, trial_result_path, phenopacket_path)
    patient_dir_metas: List[Dict[str, Any]] = []

    if args.patient_results_dir:
        pdir = args.patient_results_dir.resolve()
        pid = args.patient_id or pdir.name
        pp_path = resolve_phenopacket_path(
            pid,
            args.phenopacket.resolve() if args.phenopacket else None,
            args.phenopackets_dir.resolve() if args.phenopackets_dir else None,
        )
        job_list, meta = collect_per_trial_json_jobs_for_patient_dir(
            pdir,
            pid,
            pp_path,
            use_ranked_trials_json=not args.no_ranked_trials_json,
        )
        patient_dir_metas.append(meta)
        if meta.get("ranked_trials_json") and meta.get("trial_ids_from_ranked"):
            n_miss = len(meta.get("missing_per_trial_json") or [])
            print(
                f"[info] {pid}: ranked list → {len(job_list)} trial JSON(s) scheduled "
                f"({meta['trial_ids_from_ranked']} ids, {n_miss} missing on disk)",
                file=sys.stderr,
            )
        if meta.get("missing_per_trial_json"):
            miss = meta["missing_per_trial_json"]
            tail = "..." if len(miss) > 15 else ""
            print(
                f"[warn] {pid}: ranked list has no matching JSON for: {miss[:15]}{tail}",
                file=sys.stderr,
            )
        jobs.extend(job_list)
    elif args.results_root:
        root = args.results_root.resolve()
        print(
            "[info] --results-root: "
            f"argv={args.results_root!s} → absolute={root}  (process cwd={Path.cwd()})",
            file=sys.stderr,
        )
        if not root.is_dir():
            parser.error(f"--results-root is not a directory: {root}")
        patient_subdirs = _patient_result_subdirs(root)
        json_at_root = sorted(p for p in root.glob("*.json") if p.is_file())
        use_ranked_for_layout = not args.no_ranked_trials_json

        def _enqueue_patient_jsons(folder: Path, pid: str) -> None:
            try:
                pp = resolve_phenopacket_path(
                    pid,
                    None,
                    args.phenopackets_dir.resolve() if args.phenopackets_dir else None,
                )
            except FileNotFoundError:
                print(f"[skip] no phenopacket for patient_id={pid}", file=sys.stderr)
                return
            job_list, meta = collect_per_trial_json_jobs_for_patient_dir(
                folder,
                pid,
                pp,
                use_ranked_trials_json=not args.no_ranked_trials_json,
            )
            patient_dir_metas.append(meta)
            if meta.get("ranked_trials_json") and meta.get("trial_ids_from_ranked"):
                n_miss = len(meta.get("missing_per_trial_json") or [])
                print(
                    f"[info] {pid}: ranked list → {len(job_list)} trial JSON(s) scheduled "
                    f"({meta['trial_ids_from_ranked']} ids, {n_miss} missing on disk)",
                    file=sys.stderr,
                )
            if meta.get("missing_per_trial_json"):
                miss = meta["missing_per_trial_json"]
                tail = "..." if len(miss) > 15 else ""
                print(
                    f"[warn] {pid}: ranked list has no matching JSON for: {miss[:15]}{tail}",
                    file=sys.stderr,
                )
            jobs.extend(job_list)

        # If this directory itself is a patient folder (ranked file here), audit it only —
        # do not treat child folders (e.g. .git, logs) as separate patients.
        if use_ranked_for_layout and _find_ranked_trials_json_in_dir(root) is not None:
            pid = args.patient_id or root.name
            _enqueue_patient_jsons(root, pid)
        elif patient_subdirs:
            for sub in patient_subdirs:
                pid = args.patient_id or sub.name
                _enqueue_patient_jsons(sub, pid)
        elif json_at_root:
            pid = args.patient_id or root.name
            _enqueue_patient_jsons(root, pid)
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

    if args.results_root and not jobs:
        print(
            "[hint] No per-trial jobs were scheduled. --results-root was resolved to:\n"
            f"  {args.results_root.resolve()}\n"
            "Expected: either (1) a folder that directly contains ranked_trials.json and "
            "per-trial JSONs (one patient), or (2) a parent folder whose *subfolders* are "
            "patients (each subfolder with ranked_trials.json). "
            "Relative paths are relative to your shell cwd. "
            "For a single patient directory, you can use --patient-results-dir instead.",
            file=sys.stderr,
        )

    ranked_missing_json = sum(
        len(m.get("missing_per_trial_json") or []) for m in patient_dir_metas
    )

    total_criteria = 0
    total_mismatch_model = 0
    total_mismatch_classification = 0
    total_mismatch_either = 0
    total_mismatch_quote = 0
    total_skipped_missing_cot = 0
    total_skipped_explicit_non_cot = 0
    total_skipped_no_cot_eligibility_key = 0
    total_skipped_empty_criteria = 0
    total_skipped_api_error = 0
    total_assessments_seen = 0
    total_unparsed_files = 0
    report: Dict[str, Any] = {
        "jobs": [],
        "totals": {},
        "patient_dir_meta": patient_dir_metas,
        "scheduling": {
            "trial_json_jobs_after_cap": len(jobs),
            "ranked_list_missing_per_trial_json": ranked_missing_json,
        },
    }

    phenopacket_cache: Dict[str, Dict[str, Any]] = {}
    assessments_budget = (
        args.max_assessments if args.max_assessments > 0 else None
    )

    show_progress = not args.no_progress
    planned_audit_runs = 0
    if show_progress and jobs:
        planned_audit_runs = count_audit_runs_scheduled(jobs, args.max_assessments)
        cap_msg = (
            f", cap={args.max_assessments} assessments"
            if args.max_assessments > 0
            else ""
        )
        print(
            f"[progress] {len(jobs)} result file(s) queued, "
            f"{planned_audit_runs} CoT audit API run(s) planned{cap_msg}",
            file=sys.stderr,
            flush=True,
        )

    audit_run_i = 0
    n_jobs = len(jobs)
    for job_i, (patient_id, trial_path, pp_path) in enumerate(jobs, 1):
        if show_progress and n_jobs:
            print(
                f"[progress] job {job_i}/{n_jobs} patient={patient_id} file={trial_path.name}",
                file=sys.stderr,
                flush=True,
            )
        key = str(pp_path)
        if key not in phenopacket_cache:
            phenopacket_cache[key] = load_json(pp_path)

        phenopacket = phenopacket_cache[key]
        extracted = assessments_from_file(trial_path)
        if not extracted:
            total_unparsed_files += 1
        for assessment in extracted:
            total_assessments_seen += 1
            if assessments_budget is not None and assessments_budget <= 0:
                break
            if assessment.get("api_error"):
                total_skipped_api_error += 1
                continue
            cot = extract_cot_dict_from_assessment(assessment)
            if cot is None:
                total_skipped_missing_cot += 1
                if "cot_eligibility" not in assessment:
                    total_skipped_no_cot_eligibility_key += 1
                if assessment.get("use_cot_reasoning") is False:
                    total_skipped_explicit_non_cot += 1
                continue
            trial_id = str(
                assessment.get("trial_id")
                or assessment.get("nct_id")
                or trial_path.stem
            )
            n_crit = len(_iter_criteria_rows(cot))
            if n_crit == 0:
                total_skipped_empty_criteria += 1
                continue

            if assessments_budget is not None:
                assessments_budget -= 1

            audit_run_i += 1
            if show_progress:
                if planned_audit_runs > 0:
                    frac = f"{audit_run_i}/{planned_audit_runs}"
                else:
                    frac = str(audit_run_i)
                print(
                    f"[progress] API audit {frac} trial={trial_id} patient={patient_id} "
                    f"criteria={n_crit}",
                    file=sys.stderr,
                    flush=True,
                )

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
                    max_criteria_per_api_call=args.max_criteria_per_api_call,
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

            mism_aligned = 0
            mism_class_plausible = 0
            mism_either = 0
            mism_quote = 0
            for row in audit.get("audits", []):
                ok_a = row.get("aligned_with_phenopacket", True)
                ok_c = row.get("classification_plausible_given_phenopacket", True)
                if not ok_a:
                    mism_aligned += 1
                if not ok_c:
                    mism_class_plausible += 1
                if not ok_a or not ok_c:
                    mism_either += 1
                if args.require_quote_substring and not row.get(
                    "verbatim_quote_substring_match", True
                ):
                    mism_quote += 1

            total_criteria += n_crit
            total_mismatch_model += mism_aligned
            total_mismatch_classification += mism_class_plausible
            total_mismatch_either += mism_either
            if args.require_quote_substring:
                total_mismatch_quote += mism_quote

            entry = {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "trial_path": str(trial_path),
                "phenopacket": str(pp_path),
                "criteria_count": n_crit,
                "audits_returned": len(audit.get("audits") or []),
                "mismatch_count_aligned_flag": mism_aligned,
                "mismatch_count_classification_plausible_flag": mism_class_plausible,
                "mismatch_count_either_audit_flag": mism_either,
                "audit": audit,
            }
            if args.require_quote_substring:
                entry["mismatch_count_quote_substring"] = mism_quote
            report["jobs"].append(entry)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    if show_progress and jobs:
        print(
            f"[progress] finished {audit_run_i} audit run(s), {len(report['jobs'])} report row(s)",
            file=sys.stderr,
            flush=True,
        )

    denom = total_criteria if total_criteria else 0
    skipped_no_ranked = sum(
        1 for m in patient_dir_metas if m.get("skipped_no_ranked_trials_json")
    )
    skipped_ranked_unparsed = sum(
        1 for m in patient_dir_metas if m.get("skipped_ranked_empty_or_unparsed")
    )
    report["totals"] = {
        "result_paths_scheduled": len(jobs),
        "patient_dirs_skipped_no_ranked_trials_json": skipped_no_ranked,
        "patient_dirs_skipped_ranked_unparseable": skipped_ranked_unparsed,
        "assessments_seen": total_assessments_seen,
        "skipped_api_error": total_skipped_api_error,
        "result_paths_yielding_zero_assessments": total_unparsed_files,
        "criteria_audited": total_criteria,
        "mismatch_count_aligned_flag": total_mismatch_model,
        "mismatch_rate_aligned_flag": (total_mismatch_model / denom) if denom else None,
        "mismatch_count_classification_plausible_flag": total_mismatch_classification,
        "mismatch_rate_classification_plausible_flag": (
            (total_mismatch_classification / denom) if denom else None
        ),
        "mismatch_count_either_audit_flag": total_mismatch_either,
        "mismatch_rate_either_audit_flag": (
            (total_mismatch_either / denom) if denom else None
        ),
        "skipped_missing_cot_structure": total_skipped_missing_cot,
        "skipped_assessments_use_cot_reasoning_false": total_skipped_explicit_non_cot,
        "skipped_assessments_without_cot_eligibility_key": total_skipped_no_cot_eligibility_key,
        "skipped_cot_zero_criteria_rows": total_skipped_empty_criteria,
        "skipped_no_cot": total_skipped_missing_cot + total_skipped_empty_criteria,
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
