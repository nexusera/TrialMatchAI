import re
from datetime import datetime
from typing import Any, Dict, Optional

from dateutil import parser as date_parser


def years_from_iso8601_duration(duration: Optional[str]) -> Optional[int]:
    """Return whole years from an ISO-8601 duration like P58Y1M10D, or None if no year part."""
    if not duration or not isinstance(duration, str):
        return None
    match = re.match(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", duration.strip())
    if not match or not match.group(1):
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def infer_patient_age_years_from_phenopacket(data: Dict[str, Any]) -> Optional[int]:
    """Best-effort patient age for trial eligibility filters from a Phenopacket-like dict."""
    subject = data.get("subject") or {}
    enc = subject.get("timeAtLastEncounter") or {}
    age_block = enc.get("age") or {}
    dur = age_block.get("iso8601duration")
    y = years_from_iso8601_duration(dur)
    if y is not None:
        return y
    dob = subject.get("dateOfBirth")
    if not dob:
        return None
    try:
        dob_dt = date_parser.parse(str(dob), fuzzy=True)
        today = datetime.today()
        age = (
            today.year
            - dob_dt.year
            - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
        )
        return age if age >= 0 else None
    except (ValueError, OverflowError, date_parser.ParserError):
        return None


def parse_temporal(temporal_obj: Optional[Dict]) -> str:
    """Parse complex temporal elements with error handling."""
    if not temporal_obj:
        return "Timing not specified"
    try:
        if "age" in temporal_obj:
            return parse_iso_duration(temporal_obj["age"].get("iso8601duration"))
        if "timestamp" in temporal_obj:
            return datetime.fromisoformat(temporal_obj["timestamp"]).strftime(
                "%Y-%m-%d"
            )
        if "interval" in temporal_obj:
            start = temporal_obj["interval"].get("start", "unknown")
            end = temporal_obj["interval"].get("end", "unknown")
            return f"{start} to {end}"
        return "Timing information available"
    except Exception as e:
        return f"Timing information unavailable: {str(e)}"


def parse_iso_duration(duration: Optional[str]) -> str:
    """Convert ISO8601 duration to a human-readable format."""
    if not duration:
        return "Age unspecified"
    try:
        match = re.match(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", duration)
        parts = []
        if match:
            if match.group(1):
                parts.append(f"{match.group(1)} years")
            if match.group(2):
                parts.append(f"{match.group(2)} months")
            if match.group(3):
                parts.append(f"{match.group(3)} days")
            return " ".join(parts) if parts else duration
        return duration
    except Exception as e:
        return f"Duration parsing failed: {str(e)}"
