"""
metadata_detector.py
---------------------
Extracts borehole-level metadata (job number, borehole ID, date, location,
ground water depth, elevation, coordinates, etc.) from raw page text.

Works on any text format – the patterns cover the two contrasting report
styles shown in the problem statement and many more international variants.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from borehole_parser.field_aliases import METADATA_PATTERNS


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    """Return the first captured group from the first matching pattern."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def extract_metadata(page_text: str) -> Dict[str, Optional[str]]:
    """
    Scan raw page text and extract borehole-level metadata fields.

    Parameters
    ----------
    page_text : str
        All text from one page / document header section.

    Returns
    -------
    dict with keys matching METADATA_PATTERNS plus 'job_no', 'sheet_no'.
    """
    text = page_text or ""

    result: Dict[str, Optional[str]] = {}

    for field_name, patterns in METADATA_PATTERNS.items():
        result[field_name] = _first_match(text, patterns)

    # Job / report number
    result["job_no"] = _first_match(text, [
        r"(?:JOB NO\.?|JOB NUMBER|JOB NO)[:\s]+([A-Z0-9][-A-Z0-9/]*)",
        r"(?:PROJECT NO\.?|PROJECT NUMBER)[:\s]+([A-Z0-9][-A-Z0-9/\-]+)",
    ])

    # Sheet number
    result["sheet_no"] = _first_match(text, [
        r"SHEET\s+(\d+)\s+OF\s+(\d+)",
        r"SHEET[:\s]+(\d+)",
        r"PAGE[:\s]+(\d+)",
    ])

    # Drilling equipment
    result["drilling_equipment"] = _first_match(text, [
        r"(?:DRILLING EQUIPMENT|EQUIPMENT|DRILL RIG)[:\s]+(.+?)(?:\n|$)",
        r"(?:RIG|MACHINE)[:\s]+(.+?)(?:\n|$)",
    ])

    # Driller / inspector
    result["driller"] = _first_match(text, [
        r"(?:DRILLER|DRILLERS)[:\s]+(.+?)(?:\n|$)",
        r"(?:INSPECTOR|SITE ENGINEER)[:\s]+(.+?)(?:\n|$)",
    ])

    # Casing size
    result["casing"] = _first_match(text, [
        r"(?:SIZE.*CASING|CASING SIZE)[:\s]+(.+?)(?:\n|$)",
        r"(?:CASING)[:\s]+(.+?)(?:\n|$)",
    ])

    # Strip trailing noise from each value
    for k, v in result.items():
        if isinstance(v, str):
            result[k] = v.strip().rstrip(".,;:")

    return result


def format_metadata_summary(metadata: Dict[str, Optional[str]]) -> str:
    """Return a human-readable summary of extracted metadata."""
    lines = ["=== Borehole Metadata ==="]
    for key, val in metadata.items():
        if val:
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<30s}: {val}")
    return "\n".join(lines)
