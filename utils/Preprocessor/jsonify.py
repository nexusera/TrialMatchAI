import xml.etree.ElementTree as ET
import json
import os
import re
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

EXCEL_TO_JSON_COLUMN_MAP = {
    "登记号": "nct_id",
    "试验题目": "brief_title",
    "适应症": "condition",
    "试验状态": "overall_status",
    "首次公示日期": "start_date",
    "归一化分期": "phase",
    "申办方": "sponsor",
    "药物名称": "intervention",
    "主要研究机构": "primary_institution",
    "全部研究中心": "location",
    "研究中心数量": "num_centers",
    "原始入选标准": "inclusion_criteria",
    "原始排除标准": "exclusion_criteria",
    "试验分期": "phase_raw",
}


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {}

    data["nct_id"] = root.findtext("id_info/nct_id")
    data["brief_title"] = root.findtext("brief_title")
    data["official_title"] = root.findtext("official_title")
    data["brief_summary"] = root.findtext("brief_summary/textblock")
    data["detailed_description"] = root.findtext("detailed_description/textblock")
    data["overall_status"] = root.findtext("overall_status")
    data["start_date"] = root.findtext("start_date")
    data["completion_date"] = root.findtext("completion_date")
    data["phase"] = root.findtext("phase")
    data["study_type"] = root.findtext("study_type")

    data["condition"] = [cond.text for cond in root.findall("condition")]

    data["intervention"] = []
    for intervention in root.findall("intervention"):
        data["intervention"].append(
            {
                "intervention_type": intervention.findtext("intervention_type"),
                "intervention_name": intervention.findtext("intervention_name"),
            }
        )

    data["gender"] = root.findtext("eligibility/gender")
    data["minimum_age"] = root.findtext("eligibility/minimum_age")
    data["maximum_age"] = root.findtext("eligibility/maximum_age")

    data["eligibility_criteria"] = root.findtext("eligibility/criteria/textblock")

    data["location"] = []
    for location in root.findall("location"):
        city = location.findtext("facility/address/city")
        state = location.findtext("facility/address/state")
        country = location.findtext("facility/address/country")
        location_name = location.findtext("facility/name")
        location_address = ", ".join(filter(None, [city, state, country]))
        data["location"].append(
            {"location_name": location_name, "location_address": location_address}
        )

    data["reference"] = []
    for ref in root.findall("reference"):
        data["reference"].append(
            {"citation": ref.findtext("citation"), "PMID": ref.findtext("PMID")}
        )

    return data


def parse_excel_row(row):
    """Convert a single Excel row (pandas Series) into the unified JSON schema."""
    data = {}

    data["nct_id"] = _safe_str(row.get("登记号"))
    data["brief_title"] = _safe_str(row.get("试验题目"))
    data["official_title"] = _safe_str(row.get("试验题目"))
    data["overall_status"] = _safe_str(row.get("试验状态"))
    data["start_date"] = _safe_str(row.get("首次公示日期"))
    data["phase"] = _safe_str(row.get("归一化分期"))
    data["phase_raw"] = _safe_str(row.get("试验分期"))
    data["sponsor"] = _safe_str(row.get("申办方"))
    data["num_centers"] = _coerce_int(row.get("研究中心数量"))
    data["primary_institution"] = _safe_str(row.get("主要研究机构"))
    data["inclusion_criteria"] = _safe_str(row.get("原始入选标准"))
    data["exclusion_criteria"] = _safe_str(row.get("原始排除标准"))

    condition_raw = _safe_str(row.get("适应症"))
    data["condition"] = (
        [c.strip() for c in re.split(r"[、;；,，]", condition_raw) if c.strip()]
        if condition_raw
        else []
    )

    filled_intervention = _parse_json_cell(row.get("intervention"))
    drug_raw = _safe_str(row.get("药物名称"))
    if isinstance(filled_intervention, list):
        data["intervention"] = filled_intervention
    else:
        data["intervention"] = (
            [
                {"intervention_type": None, "intervention_name": d.strip()}
                for d in re.split(r"[、;；,，]", drug_raw)
                if d.strip()
            ]
            if drug_raw
            else []
        )

    inclusion = data.get("inclusion_criteria")
    exclusion = data.get("exclusion_criteria")
    criteria_parts = []
    if inclusion:
        criteria_parts.append(f"Inclusion Criteria:\n{inclusion}")
    if exclusion:
        criteria_parts.append(f"Exclusion Criteria:\n{exclusion}")
    data["eligibility_criteria"] = (
        _safe_str(row.get("eligibility_criteria"))
        or "\n\n".join(criteria_parts)
        or None
    )

    centers_raw = _safe_str(row.get("全部研究中心"))
    primary = _safe_str(row.get("主要研究机构"))
    if centers_raw:
        names = [c.strip() for c in re.split(r"[、;；,，]", centers_raw) if c.strip()]
        data["location"] = [{"location_name": n, "location_address": None} for n in names]
    elif primary:
        data["location"] = [{"location_name": primary, "location_address": None}]
    else:
        data["location"] = []

    # --- Extract structured fields from criteria text ---
    criteria = data.get("eligibility_criteria") or ""
    condition_str = _safe_str(row.get("适应症"))

    min_age, max_age = _extract_age_range(criteria)
    data["minimum_age"] = _coerce_float(row.get("minimum_age"))
    if data["minimum_age"] is None:
        data["minimum_age"] = min_age
    data["maximum_age"] = _coerce_float(row.get("maximum_age"))
    if data["maximum_age"] is None:
        data["maximum_age"] = max_age

    data["gender"] = (
        _safe_str(row.get("gender"))
        or _extract_gender(criteria, condition_str)
    )
    data["brief_summary"] = (
        _safe_str(row.get("brief_summary"))
        or _synthesize_summary(data.get("brief_title"), data.get("condition"))
    )
    data["detailed_description"] = _safe_str(row.get("detailed_description"))
    data["completion_date"] = _safe_str(row.get("completion_date"))
    data["study_type"] = _safe_str(row.get("study_type"))
    data["reference"] = _parse_json_cell(row.get("reference")) or []
    data["ecog_range"] = _safe_str(row.get("ecog_range"))

    return data


def _safe_str(val):
    """Return a stripped string or None for NaN / missing values."""
    if val is None:
        return None
    import math
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s else None


def _coerce_float(val):
    s = _safe_str(val)
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _coerce_int(val):
    s = _safe_str(val)
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _parse_json_cell(val):
    s = _safe_str(val)
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
#  Extract structured fields from eligibility criteria text
# ---------------------------------------------------------------------------

_AGE_INCL = r"(?:（含）|（包含）|\(含\))?"
_AGE_UNIT = r"(?:周岁|岁|years?|Years?)"

_AGE_RANGE_RX = [
    # 18岁≤年龄≤75岁 (sandwich)
    re.compile(rf"(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?\s*(?:≤|<=)\s*(?:年龄|age)\s*(?:≤|<=)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?", re.I),
    # ≥18周岁且≤75周岁 / >=18 and <=75
    re.compile(rf"(?:≥|>=)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?.{{0,20}}?(?:≤|<=)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?"),
    # 18-75岁 / 18～65（含）周岁
    re.compile(rf"(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?\s*[-~～至到]\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}"),
    # 年龄18至75岁 / 年龄：18-80周岁
    re.compile(rf"年龄[在为：:]*?\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?\s*[-~～至到]\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}"),
    # between 18 and 75 years
    re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s+years", re.I),
]

_MIN_AGE_RX = [
    re.compile(rf"(?:≥|>=)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}", re.I),
    re.compile(rf"(\d+)\s*{_AGE_INCL}{_AGE_UNIT}\s*(?:及以上|以上)"),
    re.compile(rf"(?:大于|不小于|不低于|(?<!不)超过)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}"),
    re.compile(rf"年满\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}?"),
]

_MAX_AGE_RX = [
    re.compile(rf"(?:≤|<=)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}", re.I),
    re.compile(rf"(\d+)\s*{_AGE_INCL}{_AGE_UNIT}\s*(?:及以下|以下)"),
    re.compile(rf"(?:小于|不超过|不大于|不高于|低于)\s*(\d+)\s*{_AGE_INCL}{_AGE_UNIT}"),
]


def _extract_age_range(text):
    """Return (min_age, max_age) as floats, or (None, None)."""
    if not text:
        return None, None
    for pat in _AGE_RANGE_RX:
        m = pat.search(text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return (lo, hi) if lo <= hi else (hi, lo)
    min_a = max_a = None
    for pat in _MIN_AGE_RX:
        m = pat.search(text)
        if m:
            min_a = float(m.group(1))
            break
    for pat in _MAX_AGE_RX:
        m = pat.search(text)
        if m:
            max_a = float(m.group(1))
            break
    return min_a, max_a


_GENDER_ALL_RX = re.compile(
    r"男女不限|男女均可|不限性别|性别不限|不限男女|无性别限制"
    r"|gender\s*[:：]?\s*(?:all|both)",
    re.I,
)
_GENDER_MALE_RX = re.compile(r"(?:仅[限招]|限于|只招?募?)\s*男性|male\s+only", re.I)
_GENDER_FEMALE_RX = re.compile(r"(?:仅[限招]|限于|只招?募?)\s*女性|female\s+only", re.I)
_FEMALE_HINT_RX = re.compile(r"宫颈|卵巢|子宫|乳腺癌|妊娠|cervical|ovarian|breast\s+cancer", re.I)
_MALE_HINT_RX = re.compile(r"前列腺|睾丸|prostate|testicular", re.I)


def _extract_gender(criteria_text, condition_text=None):
    if not criteria_text:
        return None
    if _GENDER_ALL_RX.search(criteria_text):
        return "All"
    if _GENDER_MALE_RX.search(criteria_text):
        return "Male"
    if _GENDER_FEMALE_RX.search(criteria_text):
        return "Female"
    combined = criteria_text + " " + (condition_text or "")
    if _FEMALE_HINT_RX.search(combined):
        return "Female"
    if _MALE_HINT_RX.search(combined):
        return "Male"
    if re.search(r"[\u4e00-\u9fff]", criteria_text) and len(criteria_text) > 50:
        return "All"
    return None


def _synthesize_summary(title, condition_list):
    parts = []
    if title:
        parts.append(title.strip())
    if condition_list:
        if isinstance(condition_list, list):
            parts.append("；".join(str(c) for c in condition_list if c))
        else:
            parts.append(str(condition_list).strip())
    return " ".join(parts) if parts else None


def convert_to_json(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def process_xml_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]
    if not xml_files:
        logger.warning("No XML files found in %s", input_dir)
        return 0

    count = 0
    for filename in xml_files:
        xml_file = os.path.join(input_dir, filename)
        try:
            data = parse_xml(xml_file)
        except ET.ParseError as e:
            logger.error("Failed to parse %s: %s", filename, e)
            continue

        json_file = os.path.join(output_dir, filename.replace(".xml", ".json"))
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(convert_to_json(data))
        count += 1

    logger.info("Converted %d XML file(s) -> JSON in %s", count, output_dir)
    return count


def process_excel(excel_path, output_dir, sheet_name=0):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    logger.info(
        "Loaded Excel: %d rows, %d columns from %s", len(df), len(df.columns), excel_path
    )

    required = {"登记号"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    count = 0
    for _, row in df.iterrows():
        data = parse_excel_row(row)
        trial_id = data.get("nct_id")
        if not trial_id:
            logger.warning("Skipping row with empty 登记号")
            continue

        safe_id = re.sub(r"[^\w\-.]", "_", trial_id)
        json_file = os.path.join(output_dir, f"{safe_id}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(convert_to_json(data))
        count += 1

    logger.info("Converted %d Excel row(s) -> JSON in %s", count, output_dir)
    return count


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert clinical trial data (XML or Excel) to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Convert a folder of XML files
  python jsonify.py xml --input-dir ../../data/trials_xmls --output-dir ../../data/trials_jsons

  # Convert an Excel (.xlsx) file
  python jsonify.py excel --input-file trials.xlsx --output-dir ../../data/trials_jsons

  # Convert a specific sheet in an Excel file
  python jsonify.py excel --input-file trials.xlsx --output-dir out/ --sheet "Sheet2"
""",
    )
    subparsers = parser.add_subparsers(dest="format", help="Input format")

    xml_parser = subparsers.add_parser("xml", help="Process XML files from a directory")
    xml_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .xml trial files",
    )
    xml_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write .json output files",
    )

    excel_parser = subparsers.add_parser(
        "excel", help="Process an Excel (.xlsx/.xls) file"
    )
    excel_parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the Excel file (.xlsx or .xls)",
    )
    excel_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write .json output files",
    )
    excel_parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index to read (default: first sheet)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.format is None:
        parser.print_help()
        raise SystemExit(1)

    if args.format == "xml":
        process_xml_dir(args.input_dir, args.output_dir)
    elif args.format == "excel":
        sheet = args.sheet
        try:
            sheet = int(sheet)
        except (ValueError, TypeError):
            pass
        process_excel(args.input_file, args.output_dir, sheet_name=sheet)

