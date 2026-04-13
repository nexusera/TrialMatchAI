#!/usr/bin/env python3
"""Report field-level completeness statistics for trial JSON files.

Works on both raw JSONs (from jsonify.py) and processed/embedded JSONs
(from prepare_trials.py).

Usage:
    python scripts/stat_trial_fields.py data/custom/trials_jsons
    python scripts/stat_trial_fields.py data/custom/processed_trials --show-ids gender minimum_age
    python scripts/stat_trial_fields.py data/processed_trials --top 30
    python scripts/stat_trial_fields.py data/custom/trials_jsons --gene-criteria
    python scripts/stat_trial_fields.py data/custom/trials_jsons --gene-criteria --gene-include-summary
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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

# --- Gene / molecular eligibility heuristics (eligibility_criteria text) ---

_GENE_SYMBOLS = (
    "EGFR ALK ROS1 RET BRAF KRAS NRAS HRAS PIK3CA PTEN AKT1 AKT2 AKT3 "
    "MET NTRK1 NTRK2 NTRK3 FGFR1 FGFR2 FGFR3 FGFR4 ERBB2 HER2 BRCA1 BRCA2 "
    "PALB2 ATM CHEK2 CDK4 CDK6 MDM2 MDM4 STK11 TP53 NF1 NF2 SMARCB1 "
    "IDH1 IDH2 TERT ABL1 PDGFRA KIT FLT3 JAK2 STAT3 BCL2 MYC CCND1 "
    "AR ER ESR1 PGR ARID1A MSI TMB CTLA4 PDCD1 PD1 PDL1 CD274 LAG3 TIM3 "
    "HAVCR2".split()
)
_GENE_RE = re.compile(
    r"\b(" + "|".join(re.escape(g) for g in dict.fromkeys(_GENE_SYMBOLS)) + r")\b",
    re.IGNORECASE,
)

# (id, human label, regex) — categories are not mutually exclusive.
_GENE_CATEGORY_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    (
        "mutation_variant",
        "mutation / variant / alteration language",
        re.compile(
            r"\b(mutation|mutations|mutant|variant|variants|alteration|alterations|"
            r"somatic|germline|pathogenic|likely\s+pathogenic|VUS|"
            r"loss[\s-]of[\s-]function|gain[\s-]of[\s-]function)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wild_type",
        "wild-type / WT requirement",
        re.compile(r"\bwild[\s-]type\b|\bWT\b(?!\s+loss)", re.IGNORECASE),
    ),
    (
        "fusion_rearrangement",
        "fusion / rearrangement / translocation",
        re.compile(
            r"\b(fusion|fusions|rearrangement|rearrangements|translocation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "amplification_cn",
        "amplification / copy number",
        re.compile(
            r"\b(amplification|amplified|copy\s+number|gene\s+copy|CNV)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "testing_ngs",
        "NGS / comprehensive genomic / molecular profiling",
        re.compile(
            r"\b(NGS|next[\s-]generation\s+sequencing|whole\s+exome|WES|WGS|"
            r"comprehensive\s+genomic|tumor\s+profiling|molecular\s+profiling|"
            r"genomic\s+profiling|CGP|FoundationOne|F1CDx|MSK[\s-]?IMPACT)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "testing_ihc_fish_pcr",
        "IHC / FISH / PCR style assays",
        re.compile(
            r"\b(IHC|immunohistochemistry|immunohistochemical|FISH|"
            r"fluorescence\s+in\s+situ|PCR|RT[\s-]?PCR|ddPCR)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "biomarker_pd_l1",
        "PD-L1 / PD-1 checkpoint context",
        re.compile(r"PD[\s-]?L1|\bPD1\b|\bPD-1\b|\bCD274\b", re.IGNORECASE),
    ),
    (
        "biomarker_msi_tmb",
        "MSI / TMB / dMMR",
        re.compile(
            r"\b(MSI[\s-]?H|MSI\s+high|dMMR|deficient\s+MMR|"
            r"microsatellite|TMB|tumor\s+mutational\s+burden)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "her2_hormone",
        "HER2 / hormone receptor (often eligibility)",
        re.compile(
            r"\b(HER2|ERBB2|hormone\s+receptor|ER\s*\+|PR\s*\+|"
            r"estrogen\s+receptor|progesterone\s+receptor|triple[\s-]negative)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "brca_hrd",
        "BRCA / HRD / PARP pathway",
        re.compile(
            r"\b(BRCA1|BRCA2|HRD|homologous\s+recombination|PARP)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "genetic_screening",
        "genetic testing / carrier / family history",
        re.compile(
            r"\b(genetic\s+test|genetic\s+testing|germline\s+test|carrier|"
            r"family\s+history\s+of.*(cancer|mutation))\b",
            re.IGNORECASE,
        ),
    ),
]


def _trial_text_for_gene_scan(data: Dict[str, Any], include_summary: bool) -> str:
    parts: List[str] = []
    ec = data.get("eligibility_criteria")
    if isinstance(ec, str) and ec.strip():
        parts.append(ec)
    if include_summary:
        for key in ("brief_summary", "official_title", "brief_title"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v)
    return "\n\n".join(parts)


def analyze_gene_criteria_text(text: str) -> Tuple[bool, Set[str], Dict[str, int]]:
    """Return (any_signal, category_ids, gene_symbol_hit_counts_upper)."""
    if not text or not text.strip():
        return False, set(), {}

    cats: Set[str] = set()
    for cat_id, _label, pat in _GENE_CATEGORY_PATTERNS:
        if pat.search(text):
            cats.add(cat_id)

    gene_counts: Dict[str, int] = defaultdict(int)
    for m in _GENE_RE.finditer(text):
        sym = m.group(0).upper()
        if sym == "MET":
            start = max(0, m.start() - 48)
            end = min(len(text), m.end() + 48)
            window = text[start:end]
            if not re.search(
                r"(?:c-)?MET(?:\s+(?:exon|amplification|mutation|gene|positive|negative|status|"
                r"overexpression|inhibitor|pathway|skipping))|(?:\bMET\s*[/,])|(?:[/,]\s*MET\b)",
                window,
                re.IGNORECASE,
            ):
                continue
        gene_counts[sym] += 1

    has_symbols = bool(gene_counts)
    has_signal = bool(cats) or has_symbols
    return has_signal, cats, dict(gene_counts)


def run_gene_criteria_report(
    directory: Path,
    *,
    include_summary: bool,
    show_ids_top: int,
    list_ids: bool,
) -> None:
    json_files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".json")
    total = len(json_files)
    with_signal = 0
    category_trial_counts: Dict[str, int] = defaultdict(int)
    gene_doc_freq: Dict[str, int] = defaultdict(int)
    gene_total_hits: Dict[str, int] = defaultdict(int)
    trial_ids_signal: List[str] = []

    for path in json_files:
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  Error reading {path.name}: {exc}", file=sys.stderr)
            continue
        text = _trial_text_for_gene_scan(data, include_summary)
        has_sig, cats, genes = analyze_gene_criteria_text(text)
        if not has_sig:
            continue
        with_signal += 1
        tid = path.stem
        trial_ids_signal.append(tid)
        for c in cats:
            category_trial_counts[c] += 1
        for g, n in genes.items():
            gene_doc_freq[g] += 1
            gene_total_hits[g] += n

    label_by_id = {cid: lab for cid, lab, _ in _GENE_CATEGORY_PATTERNS}

    print(f"\n{'=' * 78}")
    scope = "eligibility_criteria" + (
        " + brief_title/official_title/brief_summary" if include_summary else ""
    )
    print(f"  Gene / molecular-related text — {directory}")
    print(f"  Scope: {scope}")
    print(f"  Trials scanned: {total}  |  With gene/molecular signals: {with_signal}")
    if total:
        print(f"  Share: {100.0 * with_signal / total:.2f}%")
    print(f"{'=' * 78}\n")

    print("  Category (trial-level; one trial can match multiple):")
    ranked_cats = sorted(category_trial_counts.items(), key=lambda x: -x[1])
    for cid, cnt in ranked_cats:
        lab = label_by_id.get(cid, cid)
        pct = 100.0 * cnt / total if total else 0.0
        print(f"    {cnt:6d}  ({pct:5.1f}% of all trials)  {cid}: {lab}")
    if not ranked_cats:
        print("    (no category regex matches; trials may only hit gene symbols)")

    print("\n  Gene / biomarker symbols (trial-level doc frequency, top by #trials):")
    ranked_genes = sorted(gene_doc_freq.items(), key=lambda x: (-x[1], x[0]))[:40]
    for g, n_docs in ranked_genes:
        hits = gene_total_hits[g]
        pct = 100.0 * n_docs / total if total else 0.0
        print(f"    {n_docs:6d} trials ({pct:5.1f}%)  {g}  (token hits in matched trials: {hits})")
    if not ranked_genes:
        print("    (none)")

    if list_ids and trial_ids_signal:
        print(f"\n  Trial IDs with any signal ({len(trial_ids_signal)}), first {show_ids_top}:")
        for tid in trial_ids_signal[:show_ids_top]:
            print(f"    - {tid}")
        if len(trial_ids_signal) > show_ids_top:
            print(f"    ... and {len(trial_ids_signal) - show_ids_top} more")


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


def analyze_payload(data: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    all_fields = CORE_FIELDS + VECTOR_FIELDS + EXTRA_FIELDS
    for field in all_fields:
        if field not in data:
            result[field] = "missing"
        else:
            result[field] = classify_value(data[field])

    known = set(all_fields)
    for key in data:
        if key not in known:
            result[key] = classify_value(data[key])

    return result


def analyze_file(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return analyze_payload(data)


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
    parser.add_argument(
        "--gene-criteria",
        action="store_true",
        help="Also scan eligibility text for gene/molecular patterns (see --gene-include-summary).",
    )
    parser.add_argument(
        "--gene-criteria-only",
        action="store_true",
        help="Only run gene/molecular eligibility statistics (skip field completeness table).",
    )
    parser.add_argument(
        "--gene-include-summary",
        action="store_true",
        help="Include brief_title, official_title, brief_summary in gene scan (default: eligibility_criteria only).",
    )
    parser.add_argument(
        "--gene-list-ids",
        action="store_true",
        help="List sample trial IDs that matched any gene/molecular signal.",
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

    if args.gene_criteria_only:
        run_gene_criteria_report(
            directory,
            include_summary=args.gene_include_summary,
            show_ids_top=args.top,
            list_ids=args.gene_list_ids,
        )
        return 0

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

    if args.gene_criteria:
        run_gene_criteria_report(
            directory,
            include_summary=args.gene_include_summary,
            show_ids_top=args.top,
            list_ids=args.gene_list_ids,
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

