#!/usr/bin/env python3
"""Prepare & embed raw clinical-trial JSONs for Elasticsearch indexing.

Handles both English (ClinicalTrials.gov) and Chinese (CHICTR) trial data.
When structured metadata is missing (common for Chinese registries), the script
extracts age, gender, and other fields from the eligibility criteria text and
synthesises a brief_summary from available fields.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dateutil.parser
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Sentence Embedder (batched, FP16-aware)
# ---------------------------------------------------------------------------

class SentenceEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading embedding model %s → %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Converted embedding model to FP16")
        self.model.eval()
        self.max_length = max_length
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.model.config.hidden_size
        return self._dim

    def _mean_pool(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * expanded, dim=1)
        counts = torch.clamp(expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of non-empty strings; returns list of float vectors."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        vecs = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=1)
        return vecs.cpu().tolist()

    def embed_single(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        return self.embed_batch([text])[0]


# ---------------------------------------------------------------------------
#  Text / field helpers
# ---------------------------------------------------------------------------

def preprocess_text(t: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", t).strip() if t else None


def to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return dateutil.parser.parse(s).date().isoformat()
    except Exception:
        return None


def age_to_years(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return round(float(s), 2)
    s = str(s).strip()
    if not s:
        return None
    m = re.search(r"([\d.]+)", s)
    if not m:
        return None
    v = float(m.group(1))
    low = s.lower()
    if "year" in low or "周岁" in s or "岁" in s:
        y = v
    elif "month" in low or "月" in s:
        y = v / 12.0
    elif "week" in low or ("周" in s and "周岁" not in s):
        y = v / 52.0
    elif "day" in low or "天" in s or "日" in s:
        y = v / 365.0
    else:
        y = v
    return round(y, 2)


def _flatten_text(val: Any) -> Optional[str]:
    """Turn a string, list-of-strings, or None into a single clean string."""
    if val is None:
        return None
    if isinstance(val, list):
        joined = " ".join(str(v) for v in val if v)
        return preprocess_text(joined) if joined.strip() else None
    return preprocess_text(str(val))


# ---------------------------------------------------------------------------
#  Criteria-based extraction (age, gender, ECOG)
# ---------------------------------------------------------------------------

_AGE_RANGE_PATTERNS = [
    re.compile(r"(?:≥|>=)\s*(\d+)\s*(?:周岁|岁).*?(?:≤|<=)\s*(\d+)\s*(?:周岁|岁)"),
    re.compile(r"(\d+)\s*[-~～至到]\s*(\d+)\s*(?:周岁|岁)"),
    re.compile(r"(?:age|aged?)\s*(?:≥|>=)\s*(\d+).*?(?:≤|<=)\s*(\d+)", re.I),
    re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s+years", re.I),
]
_MIN_AGE_ONLY = [
    re.compile(r"(?:≥|>=)\s*(\d+)\s*(?:周岁|岁|years?)", re.I),
    re.compile(r"(?:age|年龄)\s*(?:≥|>=)\s*(\d+)", re.I),
]
_MAX_AGE_ONLY = [
    re.compile(r"(?:≤|<=)\s*(\d+)\s*(?:周岁|岁|years?)", re.I),
    re.compile(r"(?:age|年龄)\s*(?:≤|<=)\s*(\d+)", re.I),
]


def extract_ages_from_criteria(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    for pat in _AGE_RANGE_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1)), float(m.group(2))
    min_age = max_age = None
    for pat in _MIN_AGE_ONLY:
        m = pat.search(text)
        if m:
            min_age = float(m.group(1))
            break
    for pat in _MAX_AGE_ONLY:
        m = pat.search(text)
        if m:
            max_age = float(m.group(1))
            break
    return min_age, max_age


_GENDER_ALL = re.compile(
    r"男女不限|男女均可|不限性别|不限男女|gender\s*[:：]?\s*(?:all|both)", re.I
)
_GENDER_MALE = re.compile(
    r"(?:仅限|限于|仅招募?)男性|(?:male\s+only|men\s+only)", re.I
)
_GENDER_FEMALE = re.compile(
    r"(?:仅限|限于|仅招募?)女性|(?:female\s+only|women\s+only)", re.I
)


def extract_gender_from_criteria(text: str) -> Optional[str]:
    if not text:
        return None
    if _GENDER_ALL.search(text):
        return "All"
    if _GENDER_MALE.search(text):
        return "Male"
    if _GENDER_FEMALE.search(text):
        return "Female"
    return None


_ECOG_PAT = re.compile(r"ECOG[^0-9]{0,20}(\d)\s*[-~～至到]\s*(\d)", re.I)
_ECOG_SINGLE = re.compile(r"ECOG[^0-9]{0,20}(?:≤|<=)\s*(\d)", re.I)


def extract_ecog_from_criteria(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ECOG_PAT.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _ECOG_SINGLE.search(text)
    if m:
        return f"0-{m.group(1)}"
    return None


# ---------------------------------------------------------------------------
#  Normalization (phase, status)
# ---------------------------------------------------------------------------

_PHASE_MAP = {
    "phase1": "Phase 1",
    "phase 1": "Phase 1",
    "i期": "Phase 1",
    "ⅰ期": "Phase 1",
    "phase2": "Phase 2",
    "phase 2": "Phase 2",
    "ii期": "Phase 2",
    "ⅱ期": "Phase 2",
    "phase3": "Phase 3",
    "phase 3": "Phase 3",
    "iii期": "Phase 3",
    "ⅲ期": "Phase 3",
    "phase4": "Phase 4",
    "phase 4": "Phase 4",
    "iv期": "Phase 4",
    "ⅳ期": "Phase 4",
    "phase 1/phase 2": "Phase 1/Phase 2",
    "phase 2/phase 3": "Phase 2/Phase 3",
}

_STATUS_MAP = {
    "进行中": "Recruiting",
    "招募中": "Recruiting",
    "尚未招募": "Not yet recruiting",
    "已完成": "Completed",
    "已终止": "Terminated",
    "暂停": "Suspended",
    "主动终止": "Withdrawn",
}


def normalize_phase(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _PHASE_MAP.get(raw.lower().strip(), raw)


def normalize_status(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _STATUS_MAP.get(raw.strip(), raw)


# ---------------------------------------------------------------------------
#  Brief-summary synthesis
# ---------------------------------------------------------------------------

def synthesize_brief_summary(doc: dict) -> Optional[str]:
    """Build a brief summary from available fields when brief_summary is missing."""
    parts: List[str] = []
    for key in ("brief_title", "official_title"):
        val = doc.get(key)
        if val and str(val).strip():
            parts.append(str(val).strip())
            break
    cond = doc.get("condition")
    if isinstance(cond, list):
        cond = "；".join(str(c) for c in cond if c)
    if cond and str(cond).strip():
        parts.append(str(cond).strip())
    desc = doc.get("detailed_description")
    if desc and str(desc).strip():
        parts.append(str(desc).strip()[:500])
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------------------
#  Core: embed_and_prepare
# ---------------------------------------------------------------------------

_TEXT_FIELDS = [
    ("brief_title", "brief_title_vector"),
    ("brief_summary", "brief_summary_vector"),
    ("condition", "condition_vector"),
    ("eligibility_criteria", "eligibility_criteria_vector"),
]

_SIMPLE_PASS = [
    "overall_status", "phase", "study_type", "gender",
    "official_title", "detailed_description", "sponsor",
]

_NESTED_PASS = ["intervention", "location", "reference"]


def embed_and_prepare(doc: dict, embedder: SentenceEmbedder) -> dict:
    out: Dict[str, Any] = {"nct_id": doc["nct_id"]}

    # --- Synthesize brief_summary when missing ---
    if not doc.get("brief_summary"):
        synth = synthesize_brief_summary(doc)
        if synth:
            doc["brief_summary"] = synth

    # --- Collect texts to embed (batch) ---
    embed_items: List[Tuple[str, str, str]] = []  # (field, vec_name, clean_text)
    for field, vec_name in _TEXT_FIELDS:
        txt = _flatten_text(doc.get(field))
        if txt:
            embed_items.append((field, vec_name, txt))

    # --- Batch embed ---
    if embed_items:
        texts = [t for _, _, t in embed_items]
        vectors = embedder.embed_batch(texts)
        for (field, vec_name, txt), vec in zip(embed_items, vectors):
            out[field] = txt
            out[vec_name] = vec

    # --- Simple passthroughs (skip None) ---
    for key in _SIMPLE_PASS:
        val = doc.get(key)
        if val is not None:
            out[key] = val

    # --- Normalize phase & status ---
    if "phase" in out:
        out["phase"] = normalize_phase(out["phase"])
    if "overall_status" in out:
        out["overall_status"] = normalize_status(out["overall_status"])

    # --- Gender fallback from criteria ---
    if not out.get("gender"):
        g = extract_gender_from_criteria(doc.get("eligibility_criteria") or "")
        if g:
            out["gender"] = g

    # --- Dates ---
    for key in ("start_date", "completion_date"):
        iso = to_iso_date(doc.get(key))
        if iso:
            out[key] = iso

    # --- Ages: structured first, then criteria fallback ---
    for key in ("minimum_age", "maximum_age"):
        val = doc.get(key)
        if val is not None:
            yrs = age_to_years(val)
            if yrs is not None:
                out[key] = yrs

    if "minimum_age" not in out or "maximum_age" not in out:
        criteria = doc.get("eligibility_criteria") or ""
        min_age, max_age = extract_ages_from_criteria(criteria)
        if min_age is not None and "minimum_age" not in out:
            out["minimum_age"] = min_age
        if max_age is not None and "maximum_age" not in out:
            out["maximum_age"] = max_age

    # --- ECOG score (bonus metadata) ---
    ecog = extract_ecog_from_criteria(doc.get("eligibility_criteria") or "")
    if ecog:
        out["ecog_range"] = ecog

    # --- Nested passthroughs ---
    for key in _NESTED_PASS:
        val = doc.get(key)
        if val:
            out[key] = val

    return out


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

_EXPECTED_VECTOR_FIELDS = {v for _, v in _TEXT_FIELDS}


def validate_output(out: dict, nct_id: str) -> List[str]:
    """Return a list of warning messages for missing important fields."""
    warnings_list: List[str] = []
    present_vecs = [k for k in _EXPECTED_VECTOR_FIELDS if k in out]
    if len(present_vecs) < 2:
        warnings_list.append(
            f"Only {len(present_vecs)}/4 vector fields present: {present_vecs}"
        )
    for key in ("gender", "minimum_age", "maximum_age"):
        if key not in out:
            warnings_list.append(f"Missing {key}")
    return warnings_list


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Prepare & embed clinical trial JSONs for ES indexing."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids-file", help="File with one trial ID per line")
    src.add_argument(
        "--scan",
        action="store_true",
        help="Auto-scan --source-folder for all .json files instead of using --ids-file",
    )
    p.add_argument("--source-folder", required=True, help="Raw trial JSONs directory")
    p.add_argument(
        "--processed-folder",
        default="processed_docs",
        help="Output directory for embedded JSONs",
    )
    p.add_argument("--model-name", default="BAAI/bge-m3", help="Sentence-transformer model")
    p.add_argument("--device", default=None, help="Device: 'cpu', 'cuda', 'cuda:1', …")
    p.add_argument("--max-length", type=int, default=512, help="Max token length for embedding")
    p.add_argument("--force", action="store_true", help="Re-process even if output exists")
    p.add_argument("--batch-size", type=int, default=1, help="Reserved for future batch-file support")
    args = p.parse_args()

    source = Path(args.source_folder)
    processed = Path(args.processed_folder)
    processed.mkdir(parents=True, exist_ok=True)

    # --- Resolve trial IDs ---
    if args.scan:
        ids = sorted(
            p.stem for p in source.iterdir() if p.suffix.lower() == ".json"
        )
        logger.info("Scanned %d JSON files from %s", len(ids), source)
    else:
        with open(args.ids_file, encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d IDs from %s", len(ids), args.ids_file)

    if not ids:
        logger.error("No trial IDs to process.")
        return 1

    embedder = SentenceEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
    )

    counts = {"processed": 0, "skipped": 0, "missing": 0, "failed": 0}
    field_warnings: Dict[str, int] = {}
    t0 = time.time()

    for i, nct_id in enumerate(ids, start=1):
        out_path = processed / f"{nct_id}.json"

        # --- Skip if already done ---
        if out_path.exists() and not args.force:
            counts["skipped"] += 1
            continue

        in_path = source / f"{nct_id}.json"
        if not in_path.exists():
            logger.warning("[%d/%d] Missing raw JSON: %s", i, len(ids), nct_id)
            counts["missing"] += 1
            continue

        try:
            with in_path.open(encoding="utf-8") as f:
                doc = json.load(f)
            doc.setdefault("nct_id", nct_id)

            result = embed_and_prepare(doc, embedder)

            # Validate & warn
            warns = validate_output(result, nct_id)
            for w in warns:
                field_warnings[w] = field_warnings.get(w, 0) + 1

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            counts["processed"] += 1
            if i % 20 == 0 or i == len(ids):
                elapsed = time.time() - t0
                rate = counts["processed"] / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] %s  (%.1f trials/min)",
                    i, len(ids), nct_id, rate * 60,
                )
        except Exception:
            logger.exception("[%d/%d] Failed to process %s", i, len(ids), nct_id)
            counts["failed"] += 1

    elapsed = time.time() - t0
    logger.info(
        "\nDone in %.1fs — %d processed, %d skipped, %d missing, %d failed",
        elapsed, counts["processed"], counts["skipped"],
        counts["missing"], counts["failed"],
    )
    if field_warnings:
        logger.info("Field-level warnings across processed trials:")
        for msg, cnt in sorted(field_warnings.items(), key=lambda x: -x[1]):
            logger.info("  %4d × %s", cnt, msg)

    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

