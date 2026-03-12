# -*- coding: utf-8 -*-
"""
Geotechnical Borehole Log Digitization Pipeline
================================================
Extracts structured data from scanned/digital borehole report PDFs using
OCR, regex-based NLP, and rule-based geotechnical logic.

Libraries used
--------------
- os, re        : file system access and pattern matching
- cv2           : image pre-processing (grayscale, thresholding) before OCR
- fitz          : PyMuPDF – open PDF pages, render pixmaps, clip regions
- numpy         : pixel array manipulation
- pandas        : tabular data aggregation and Excel/CSV export
- pytesseract   : Python wrapper for Tesseract OCR engine

Pipeline overview
-----------------
For every PDF in INPUT_DIR:
  1. Open the PDF with PyMuPDF.
  2. For each page:
     a. OCR the full page (300 dpi, Otsu binarisation).
     b. OCR the top header region (350 dpi) for borehole ID and coordinates.
     c. OCR a narrow right-side coordinate region (500 dpi) as a fallback.
     d. Extract metadata: BH ID, Easting, Northing, Elevation.
     e. Parse SPT rows (depth, N-value, blow counts, soil type).
     f. Parse rock-quality indices (RQD, TCR, SCR) and water depth.
  3. Collate rows into a pandas DataFrame and export to Excel + CSV.
"""

import logging
import os
import re

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "input_reports"
OUTPUT_DIR = "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Optional manual metadata override for unreliable OCR pages.
# Keep empty for automatic extraction; add entries only when needed.
# Example: MANUAL_BH_METADATA = {"BH-01": {"Easting": 312345.6, "Northing": 3045678.9}}
MANUAL_BH_METADATA: dict = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

BH_PATTERN = re.compile(r"\b(?:BH|Bore\s*Hole)\s*[-:# ]*0*(\d{1,4})\b", re.I)
EAST_PATTERN = re.compile(
    r"\b(?:E|Easting)\b\s*[:=]?\s*[~\-\u2013\u2014]*\s*([0-9OolISB.,]{5,16})",
    re.I,
)
NORTH_PATTERN = re.compile(
    r"\b(?:N|Northing)\b\s*[:=]?\s*[~\-\u2013\u2014]*\s*([0-9OolISB.,]{6,16})",
    re.I,
)
ELEV_PATTERN = re.compile(
    r"\b(?:Elev(?:ation)?|RL)\s*[:=]?\s*([0-9]{1,6}(?:\.[0-9]+)?)\b", re.I
)
ELEV_DIRECT_PATTERN = re.compile(
    r"\b(?:Elev(?:ation)?|RL)\b\s*[:=]?\s*([0-9OolISB.,]{1,10})",
    re.I,
)
ELEV_BEFORE_E_PATTERN = re.compile(
    r"\b([0-9OolISB.,]{1,10})\s*(?:E|Easting)\s*[:=]",
    re.I,
)
GEO_INV_PATTERN = re.compile(r"\bGeo\.?\s*Inv\.?\s*([0-9OolISB.,]{1,10})", re.I)

SPT_PAIR_PATTERN = re.compile(
    r"\b(\d{1,3})\s*SPT\b[^\n]{0,15}?\b(?:at\s*)?(\d{1,3}(?:\.\d+)?)\s*m\b", re.I
)
LONG_NUM_PATTERN = re.compile(r"\b([0-9OolISB.,]{6,16})\b")
NVALUE_PATTERN = re.compile(r"\bN\s*[:=]?\s*(\d{1,3})\b", re.I)

RQD_PATTERN = re.compile(r"\bRQD\s*[:=]?\s*(\d{1,3})\b", re.I)
TCR_PATTERN = re.compile(r"\bTCR\s*[:=]?\s*(\d{1,3})\b", re.I)
SCR_PATTERN = re.compile(r"\bSCR\s*[:=]?\s*(\d{1,3})\b", re.I)
WATER_PATTERN = re.compile(
    r"\b(?:water\s*(?:level|table|depth)?|w\.?l\.?)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*m\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Soil vocabulary
# ---------------------------------------------------------------------------

SOIL_TERMS = [
    "siltstone/calcisiltite",
    "claystone",
    "shale",
    "limestone",
    "dolomitic limestone",
    "fossiliferous limestone",
    "sandstone/calcarinite",
    "fine/medium/coarse sand",
    "silty/clayey sand/ with gravels/pebbles/cobbles/boulders/gypsum",
    "sandy, clayey silt/ with gravels/pebbles/cobbles/boulders/gypsum",
    "sandy, silty clay/ with gravels/pebbles/cobbles/boulders/gypsum",
    "gravels",
    "gravels with sand/silt/clay/pebbles/cobbles/boulders/gypsum",
    "cemented sand/ with silt/clay/gravels/pebbles/gypsum",
    "lateritic material/ with sand/silt/clay/gravels",
    "gypsum/crystalline gypsum/ with sand/silt/gravels",
    "asphalt/ with sand/silt/gravels/pebbles/cobbles/boulders",
    "silty sand",
    "clayey sand",
    "gravelly sand",
    "sandy gravel",
    "sandy silt",
    "clayey silt",
    "silty clay",
    "sandy clay",
    "topsoil",
    "gravel",
    "sand",
    "silt",
    "clay",
    "rock",
    "fill",
]

PRIMARY_SOIL_TOKEN_MAP = {
    "ASPHALT": "ASPHALT",
    "LATERITIC": "LATERITIC MATERIAL",
    "LATERITE": "LATERITIC MATERIAL",
    "GYPSUM": "GYPSUM",
    "LIMESTONE": "LIMESTONE",
    "SILTSTONE": "SILTSTONE/CALCISILTITE",
    "CALCISILTITE": "SILTSTONE/CALCISILTITE",
    "CLAYSTONE": "CLAYSTONE",
    "SHALE": "SHALE",
    "SANDSTONE": "SANDSTONE/CALCARINITE",
    "CALCARINITE": "SANDSTONE/CALCARINITE",
    "SAND": "SAND",
    "SILT": "SILT",
    "CLAY": "CLAY",
    "GRAVEL": "GRAVEL",
    "GRAVELS": "GRAVEL",
}

NOISE_KEYWORDS = (
    "tel",
    "fax",
    "report ref",
    "document ref",
    "project title",
    "contract",
    "contr.",
    "copyright",
    "email",
    "website",
    "www",
    "astm",
)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "Borehole_ID",
    "Easting",
    "Northing",
    "Elevation_m",
    "Depth_from",
    "Depth_to",
    "Soil_Type",
    "N_value",
    "Blow_Count_Raw",
    "Blow_Count_1",
    "Blow_Count_2",
    "Blow_Count_3",
    "Blow_Count_4",
    "Blow_Count_5",
    "Blow_Count_6",
    "RQD",
    "TCR",
    "SCR",
    "Water_Depth",
    "Notes",
    "Page_No",
    "Source_File",
]

# ---------------------------------------------------------------------------
# Coordinate validation bounds (update for your project area)
# ---------------------------------------------------------------------------

EASTING_MIN = 300_000.0
EASTING_MAX = 330_000.0
NORTHING_MIN = 3_000_000.0
NORTHING_MAX = 3_100_000.0
ELEV_MIN = -100.0
ELEV_MAX = 1_000.0

# Image-clip ratios for selective OCR regions
HEADER_CLIP_RATIO = 0.28
COORD_CLIP_X0 = 0.58
COORD_CLIP_X1 = 0.98
COORD_CLIP_Y0 = 0.07
COORD_CLIP_Y1 = 0.28

REFUSAL_TOKENS = {"R", "REF", "REFUSAL", "RF"}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Collapse whitespace within each line and remove blank lines."""
    text = text.replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def top_lines(text: str, max_lines: int = 18) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[:max_lines])


def clean_note(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" |;,:.-")
    if len(text) > 260:
        text = text[:260].rstrip()
    return text


def normalize_numeric_token(token: str):
    """
    Correct common OCR single-character confusions (O→0, l→1, S→5, B→8)
    and return a float, or None if the token cannot be parsed.
    """
    translate = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "S": "5",
            "B": "8",
            "|": "1",
            ",": ".",
        }
    )
    cleaned = token.translate(translate)
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    if not cleaned:
        return None
    # Keep only the first decimal point.
    if cleaned.count(".") > 1:
        first = cleaned.index(".")
        cleaned = cleaned[: first + 1] + cleaned[first + 1 :].replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_coordinate_value(raw_token: str, coord_type: str):
    """
    Parse a raw OCR token as an Easting ('E') or Northing ('N') coordinate.

    For Easting values the function applies a heuristic two-digit prefix
    repair because OCR frequently misreads the leading digits of 6-digit
    Easting numbers (e.g. 342497 → 312497 when project Eastings start with
    31xxxx).  The repair is skipped for 7-digit Northing-like tokens.
    """
    value = normalize_numeric_token(raw_token)
    if value is None:
        return None

    if coord_type == "E":
        if EASTING_MIN <= value <= EASTING_MAX:
            return value

        token = str(raw_token).translate(
            str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", ",": "."})
        )
        int_part, _dot, frac_part = token.partition(".")
        digits = re.sub(r"\D", "", int_part)
        if len(digits) == 6 and not digits.startswith("30"):
            fixed_int = "31" + digits[-4:]
            frac_digits = re.sub(r"\D", "", frac_part)
            fixed = fixed_int + (("." + frac_digits) if frac_digits else "")
            fixed_val = normalize_numeric_token(fixed)
            if fixed_val is not None and EASTING_MIN <= fixed_val <= EASTING_MAX:
                return fixed_val
        return None

    if coord_type == "N":
        if NORTHING_MIN <= value <= NORTHING_MAX:
            return value
        return None

    return None


def normalize_elevation_value(raw_token: str):
    value = normalize_numeric_token(raw_token)
    if value is None:
        return None
    if ELEV_MIN <= value <= ELEV_MAX:
        return value
    return None


# ---------------------------------------------------------------------------
# Elevation extraction
# ---------------------------------------------------------------------------


def extract_elevation_value(text: str):
    """
    Extract the elevation / RL value from a text block using multiple
    patterns in decreasing confidence order.
    """
    # Pattern 1 – direct label: "Elev: 12.5" or "RL = 8.00"
    m = ELEV_DIRECT_PATTERN.search(text)
    if m:
        value = normalize_elevation_value(m.group(1))
        if value is not None:
            return value

    # Pattern 2 – "Geo. Inv. 12.5"
    m = GEO_INV_PATTERN.search(text)
    if m:
        value = normalize_elevation_value(m.group(1))
        if value is not None:
            return value

    # Pattern 3 – elevation appears before the Easting label
    m = ELEV_BEFORE_E_PATTERN.search(text)
    if m:
        value = normalize_elevation_value(m.group(1))
        if value is not None:
            return value

    # Pattern 4 – multi-line search around lines that contain "elev" or "geo"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, line in enumerate(lines):
        lower = line.lower()
        if "elev" not in lower and "geo" not in lower:
            continue
        block = " ".join(lines[i : min(i + 3, len(lines))])
        for pat in (ELEV_DIRECT_PATTERN, ELEV_BEFORE_E_PATTERN, GEO_INV_PATTERN):
            m = pat.search(block)
            if m:
                value = normalize_elevation_value(m.group(1))
                if value is not None:
                    return value

    return None


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------


def _render_to_bgr(page, dpi: int, clip=None):
    """Render a PyMuPDF page (or clipped region) to a BGR numpy array."""
    pix = page.get_pixmap(dpi=dpi, alpha=False, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def _binarize(bgr_img):
    """Convert a BGR image to a binarized (Otsu threshold) grayscale image."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_page(page) -> str:
    """Full-page OCR at 300 dpi."""
    img = _render_to_bgr(page, dpi=300)
    thresh = _binarize(img)
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    return normalize_text(text)


def ocr_page_header(page) -> str:
    """OCR the top 28 % of the page at 350 dpi (header region)."""
    rect = page.rect
    clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * HEADER_CLIP_RATIO)
    img = _render_to_bgr(page, dpi=350, clip=clip)
    thresh = _binarize(img)
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    return normalize_text(text)


def ocr_page_coord_region(page) -> str:
    """
    OCR a narrow right-side region likely to contain coordinate labels at
    500 dpi.  A restricted character whitelist helps Tesseract avoid
    confusing digit-like symbols.
    """
    rect = page.rect
    clip = fitz.Rect(
        rect.x0 + rect.width * COORD_CLIP_X0,
        rect.y0 + rect.height * COORD_CLIP_Y0,
        rect.x0 + rect.width * COORD_CLIP_X1,
        rect.y0 + rect.height * COORD_CLIP_Y1,
    )
    img = _render_to_bgr(page, dpi=500, clip=clip)
    thresh = _binarize(img)
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6 -c tessedit_char_whitelist=NE:.-0123456789",
    )
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------


def extract_header(text: str):
    """
    Parse borehole ID, Easting, Northing, and Elevation from an OCR text
    block.  Returns a 4-tuple (bh_id, easting, northing, elevation), where
    any item may be None if not found.

    Strategy (each field uses its own ordered fallback chain):

    BH ID    – BH_PATTERN regex
    Easting  – EAST_PATTERN → line-level label scan → proximity inference
    Northing – NORTH_PATTERN → line-level label scan → proximity inference
    Elevation – extract_elevation_value() (multi-pattern)
    """
    bh = easting = northing = None

    m = BH_PATTERN.search(text)
    if m:
        bh = f"BH-{int(m.group(1)):02d}"

    m = EAST_PATTERN.search(text)
    if m:
        easting = normalize_coordinate_value(m.group(1), "E")

    m = NORTH_PATTERN.search(text)
    if m:
        northing = normalize_coordinate_value(m.group(1), "N")

    if easting is None or northing is None:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        n_line_idx = e_line_idx = None

        for idx, line in enumerate(lines):
            lower = line.lower()
            if northing is None and ("n:" in lower or "north" in lower):
                for raw in re.findall(r"[0-9OolISB.,]{6,16}", line):
                    candidate = normalize_coordinate_value(raw, "N")
                    if candidate is not None:
                        northing = candidate
                        n_line_idx = idx
                        break
            if easting is None and ("e:" in lower or "east" in lower):
                for raw in re.findall(r"[0-9OolISB.,]{5,16}", line):
                    candidate = normalize_coordinate_value(raw, "E")
                    if candidate is not None:
                        easting = candidate
                        e_line_idx = idx
                        break

        # Proximity inference: if one coordinate is found, look in adjacent
        # lines for the other.
        if northing is not None and easting is None and n_line_idx is not None:
            for idx in range(n_line_idx + 1, min(n_line_idx + 4, len(lines))):
                for raw in re.findall(r"[0-9OolISB.,]{5,16}", lines[idx]):
                    candidate = normalize_coordinate_value(raw, "E")
                    if candidate is not None:
                        easting = candidate
                        break
                if easting is not None:
                    break

        if easting is not None and northing is None and e_line_idx is not None:
            for idx in range(max(0, e_line_idx - 2), min(e_line_idx + 2, len(lines))):
                for raw in re.findall(r"[0-9OolISB.,]{6,16}", lines[idx]):
                    candidate = normalize_coordinate_value(raw, "N")
                    if candidate is not None:
                        northing = candidate
                        break
                if northing is not None:
                    break

    elev = extract_elevation_value(text)
    return bh, easting, northing, elev


# ---------------------------------------------------------------------------
# Line classification helpers
# ---------------------------------------------------------------------------


def is_noise_line(line: str) -> bool:
    """
    Return True if the line is administrative noise (contact details,
    document references, very short fragments) that should be ignored
    during data extraction.
    """
    lower = line.lower()
    if len(line) < 8:
        return True
    if "spt" in lower:
        return False
    return any(token in lower for token in NOISE_KEYWORDS)


def is_description_update_line(line: str) -> bool:
    """
    Return True when the line appears to introduce a new soil/rock
    description layer (e.g. "SAND with silt, medium dense").

    Rules
    -----
    - Must contain at least one recognisable soil keyword.
    - Must have ≥ 3 words total and ≥ 2 meaningful words (length ≥ 3).
    - Continuation fragments ("with …", "and …") are excluded.
    - Lines containing "SPT" are not description updates.
    - Noise lines are excluded.
    """
    lower = (line or "").strip().lower()
    lead_clean = re.sub(r"^[^a-z]+", "", lower)
    if not lower:
        return False
    if "spt" in lower:
        return False
    if lead_clean.startswith(("with ", "and ", "or ")):
        return False
    if is_noise_line(line):
        return False
    if re.search(r"\b(?:description|end of borehole|sample|ground\s*level)\b", lower):
        return False

    words = re.findall(r"[A-Za-z]+", line)
    if len(words) < 3:
        return False
    meaningful_words = [w for w in words if len(w) >= 3]
    if len(meaningful_words) < 2:
        return False

    return bool(
        re.search(
            r"\b(?:sand|silt|clay|gravel|rock|asphalt|limestone|shale|gypsum"
            r"|lateritic|stone|fill|topsoil)\b",
            lower,
        )
    )


# ---------------------------------------------------------------------------
# Soil-type classifier
# ---------------------------------------------------------------------------


def has_strong_inline_soil_signal(text: str) -> bool:
    """True when *text* contains an unambiguous primary soil component."""
    snippet = text or ""
    lower = snippet.lower()
    for tok in re.findall(r"\b[A-Z]{3,}\b", snippet):
        if tok in PRIMARY_SOIL_TOKEN_MAP:
            return True
    if re.search(r"\b(?:silty|clayey|gravelly|sandy)\s+(sand|silt|clay|gravel|gravels)\b", lower):
        return True
    if re.search(
        r"\b(?:sand|silt|clay|gravel|gravels|asphalt|limestone|shale|gypsum"
        r"|lateritic|claystone|sandstone)\b\s+with\b",
        lower,
    ):
        return True
    return False


def pick_soil_type(text: str):
    """
    Infer the dominant geotechnical soil/rock classification from a
    description snippet.

    Scoring strategy (highest weight wins):
    6 – Uppercase token present in OCR text (strongest OCR signal)
    5 – Modifier→head pattern ("silty sand" → SAND, "sandy silt" → SILT)
    4 – Token appears left of "with" (the head component)
    3 – Longest matching term in SOIL_TERMS dictionary
    1 – Fallback: any token in word order

    Returns a string from PRIMARY_SOIL_TOKEN_MAP values, or None.
    """
    original = text or ""
    stripped = original.strip()
    lower = stripped.lower()
    lead_clean = re.sub(r"^[^a-z]+", "", lower)

    if not lower:
        return None
    if lead_clean.startswith(("with ", "and ", "or ")):
        return None

    words = re.findall(r"[A-Za-z]+", stripped)
    if "continued" in lower and len(words) <= 4:
        return None

    scores: dict = {}

    def add_score(token, weight):
        if token in PRIMARY_SOIL_TOKEN_MAP:
            scores[token] = scores.get(token, 0) + weight

    # 1) Uppercase heads
    for tok in re.findall(r"\b[A-Z]{3,}\b", original):
        add_score(tok, 6)

    # 2) Head appears before "with"
    left_part = re.split(r"\bwith\b", re.sub(r"^[^A-Za-z]+", "", stripped), maxsplit=1, flags=re.I)[0]
    for tok in re.findall(r"[A-Za-z]+", left_part):
        add_score(tok.upper(), 4)

    # 3) Modifier→head phrases
    for m in re.finditer(r"\b(?:silty|clayey|gravelly|sandy)\s+(sand|silt|clay|gravel|gravels)\b", lower):
        add_score(m.group(1).upper(), 5)

    # 4) Dictionary terms
    matches = [term for term in SOIL_TERMS if term in lower]
    if matches:
        best = sorted(matches, key=len, reverse=True)[0]
        for tok in re.findall(r"[A-Za-z]+", best):
            add_score(tok.upper(), 3)

    # 5) Fallback token scan
    for tok in words:
        add_score(tok.upper(), 1)

    if not scores:
        return None

    winning_token, winning_score = max(scores.items(), key=lambda x: x[1])

    # Prevent SPT context without a clear soil name from forcing a wrong class.
    if "spt" in lower and winning_score <= 2:
        return None

    return PRIMARY_SOIL_TOKEN_MAP[winning_token]


# ---------------------------------------------------------------------------
# Metric parsers
# ---------------------------------------------------------------------------


def pick_metric(text: str, pattern):
    """Extract an integer percentage/index (0–100) using *pattern*."""
    m = pattern.search(text)
    if not m:
        return None
    value = int(m.group(1))
    return value if 0 <= value <= 100 else None


def pick_water_depth(text: str):
    m = WATER_PATTERN.search(text)
    if not m:
        return None
    value = float(m.group(1))
    return value if 0 <= value <= 200 else None


def pick_nvalue(snippet: str, depth: float, line_nvalue=None):
    """
    Attempt to extract the SPT N-value from *snippet* around *depth*.

    Priority:
    1. Refusal keyword → "R"
    2. Explicitly passed *line_nvalue*
    3. Pattern "N = 15"
    4. Pattern "15 SPT"
    5. First integer in snippet that is not (approximately) the depth
    """
    if "spt" in (snippet or "").lower() and re.search(r"\b(?:R|REFUSAL)\b", snippet or "", re.I):
        return "R"
    if line_nvalue is not None and 0 <= line_nvalue <= 100:
        return line_nvalue
    m = NVALUE_PATTERN.search(snippet)
    if m:
        value = int(m.group(1))
        if 0 <= value <= 100:
            return value
    m = re.search(r"\b(\d{1,3})\s*SPT\b", snippet, re.I)
    if m:
        value = int(m.group(1))
        if 0 <= value <= 100:
            return value
    for token in re.findall(r"\b\d{1,3}\b", snippet):
        value = int(token)
        if value > 100:
            continue
        if abs(value - depth) < 0.6:
            continue
        return value
    return None


# ---------------------------------------------------------------------------
# Token / cell parsers
# ---------------------------------------------------------------------------


def clean_ocr_table_token(raw) -> str:
    return re.sub(r"[^A-Za-z0-9/.-]", "", str(raw or "")).upper()


def parse_depth_from_token(raw):
    token = str(raw or "").replace(" ", "")
    m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*m\.?$", token, re.I)
    if not m:
        return None
    depth = float(m.group(1))
    return round(depth, 2) if 0 < depth <= 100 else None


def extract_spt_depths_from_text(text: str) -> list:
    """Return an ordered, deduplicated list of SPT depths found in *text*."""
    if not text:
        return []
    depths = []
    for m in re.finditer(r"\bSPT\s*at\s*(\d{1,3}(?:\.\d+)?)\s*m\b", text, re.I):
        depth = round(float(m.group(1)), 2)
        if 0 < depth <= 100:
            depths.append(depth)
    seen: set = set()
    return [d for d in depths if not (d in seen or seen.add(d))]  # type: ignore[func-returns-value]


def parse_n_cell_token(raw):
    """Parse an SPT N-value cell token; returns int, "R", or None."""
    cleaned = clean_ocr_table_token(raw)
    if not cleaned:
        return None
    if cleaned in REFUSAL_TOKENS:
        return "R"
    if re.fullmatch(r"\d{1,3}", cleaned):
        value = int(cleaned)
        if 0 <= value <= 100:
            return value
    return None


def parse_blow_cell_token(raw):
    """
    Parse a blow-count cell token of the form "N" or "N/M".

    Returns a dict with keys:
      raw        – normalised text representation
      value      – integer blow count (capped at 50) or None
      is_refusal – True when the first increment is ≥ 50

    Returns None when the token cannot be parsed.
    """
    cleaned = clean_ocr_table_token(raw)
    if not cleaned:
        return None
    if cleaned in REFUSAL_TOKENS:
        return {"raw": cleaned, "value": None, "is_refusal": True}
    m = re.fullmatch(r"(\d{1,3})(?:/(\d{0,3}))?", cleaned)
    if not m:
        return None
    first = int(m.group(1))
    suffix = m.group(2)
    has_slash = "/" in cleaned
    raw_text = str(first)
    if has_slash:
        raw_text = f"{first}/" if suffix in {None, ""} else f"{first}/{suffix}"
    return {"raw": raw_text, "value": min(first, 50), "is_refusal": first >= 50}


def merge_slash_split_tokens(tokens: list) -> list:
    """
    Merge OCR-split "N/M" blow-count tokens where the slash and the second
    number appear as separate spatial tokens due to OCR segmentation.
    """
    if not tokens:
        return tokens
    merged = []
    i = 0
    while i < len(tokens):
        current = tokens[i]
        if (
            current["raw"].endswith("/")
            and i + 1 < len(tokens)
            and re.fullmatch(r"\d{1,3}", tokens[i + 1]["raw"])
            and abs(tokens[i + 1]["x"] - current["x"]) <= 0.03
            and 0.0 <= (tokens[i + 1]["y"] - current["y"]) <= 0.045
        ):
            nxt = tokens[i + 1]
            first = current["raw"].split("/")[0]
            combined_raw = f"{first}/{nxt['raw']}"
            merged.append(
                {
                    "raw": combined_raw,
                    "value": min(int(first), 50),
                    "is_refusal": int(first) >= 50,
                    "x": current["x"],
                    "y": current["y"],
                }
            )
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def compute_n_from_blow_values(values: list, mode: str):
    """
    Compute the SPT N-value from blow-count increments.

    mode = "7.5cm" – 6 increments of 7.5 cm (N = sum of last 4)
    mode = "15cm"  – 3 increments of 15 cm  (N = sum of last 2)
    """
    if not values:
        return None
    if mode == "7.5cm":
        if len(values) < 6:
            return None
        n_val = sum(values[-4:])
    else:
        if len(values) < 3:
            return None
        n_val = values[-2] + values[-1]
    return "R" if n_val >= 50 else int(n_val)


def empty_blow_payload() -> dict:
    return {
        "Blow_Count_Raw": "",
        "Blow_Count_1": np.nan,
        "Blow_Count_2": np.nan,
        "Blow_Count_3": np.nan,
        "Blow_Count_4": np.nan,
        "Blow_Count_5": np.nan,
        "Blow_Count_6": np.nan,
    }


def build_blow_payload(blow_tokens: list) -> dict:
    payload = empty_blow_payload()
    if not blow_tokens:
        return payload
    payload["Blow_Count_Raw"] = ", ".join(tok["raw"] for tok in blow_tokens)
    for idx, tok in enumerate(blow_tokens[:6], start=1):
        payload[f"Blow_Count_{idx}"] = tok["raw"]
    return payload


# ---------------------------------------------------------------------------
# SPT table extraction from pixmap
# ---------------------------------------------------------------------------


def extract_spt_n_overrides_from_page(page, spt_depths_hint=None) -> dict:
    """
    Perform a dedicated high-resolution OCR pass over the SPT results table
    region of *page* to extract accurate N-values and blow counts.

    Returns a dict mapping depth (float, rounded to 2 dp) → payload dict
    with keys: N_value, Blow_Count_Raw, Blow_Count_1 … Blow_Count_6.

    This supplements the text-based extraction with spatial awareness of
    the tabular layout, aligning blow-count columns to SPT row anchors.
    """
    img = _render_to_bgr(page, dpi=400)
    thresh = _binarize(img)
    h, w = thresh.shape

    def ocr_data(psm):
        data = pytesseract.image_to_data(
            thresh,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DATAFRAME,
        )
        empty = pd.DataFrame(columns=["text", "left", "top", "conf", "x", "y"])
        if data is None or data.empty:
            return empty
        data = data.dropna(subset=["text"]).copy()
        data["text"] = data["text"].astype(str).str.strip()
        data = data[data["text"] != ""]
        if data.empty:
            return empty
        data["x"] = data["left"] / float(w)
        data["y"] = data["top"] / float(h)
        return data

    d11 = ocr_data(11)
    if d11.empty:
        return {}

    raw_text = " ".join(d11["text"].astype(str).tolist()).lower()
    penetration_mode = "7.5cm" if re.search(r"7\.?5\s*cm", raw_text) else "15cm"

    spt_rows = d11[
        (d11["text"].astype(str).str.upper() == "SPT")
        & (d11["x"] >= 0.73)
        & (d11["x"] <= 0.79)
        & (d11["y"] >= 0.22)
        & (d11["y"] <= 0.96)
    ].copy()

    if spt_rows.empty:
        return {}

    row_ys = sorted(round(float(y), 4) for y in spt_rows["y"].tolist())

    if spt_depths_hint:
        depths = [round(float(d), 2) for d in spt_depths_hint if 0 < float(d) <= 100]
        depth_points = [(depths[i], row_ys[i]) for i in range(min(len(depths), len(row_ys)))]
    else:
        depth_tokens = d11[
            (d11["x"] >= 0.79) & (d11["x"] <= 0.90) & (d11["y"] >= 0.22) & (d11["y"] <= 0.96)
        ]
        found_depths = sorted(
            (
                (parse_depth_from_token(row["text"]), float(row["y"]))
                for _, row in depth_tokens.iterrows()
                if parse_depth_from_token(row["text"]) is not None
            ),
            key=lambda x: x[1],
        )
        depth_points = found_depths[: len(row_ys)]

    if not depth_points:
        return {}

    value_tokens = d11[
        (d11["x"] >= 0.54) & (d11["x"] <= 0.69) & (d11["y"] >= 0.22) & (d11["y"] <= 0.96)
    ].copy()

    overrides: dict = {}
    prev_n = None

    for idx, (depth, y_row) in enumerate(depth_points):
        y_next = depth_points[idx + 1][1] if idx + 1 < len(depth_points) else min(y_row + 0.10, 0.985)

        n_band = value_tokens[
            (value_tokens["y"] >= y_row - 0.008)
            & (value_tokens["y"] <= y_row + 0.018)
            & (value_tokens["x"] >= 0.54)
            & (value_tokens["x"] <= 0.62)
        ].sort_values(["x", "y"])

        n_candidate = None
        for _, nr in n_band.iterrows():
            parsed = parse_n_cell_token(nr["text"])
            if parsed is not None:
                n_candidate = parsed
                break

        blow_band = value_tokens[
            (value_tokens["y"] >= y_row - 0.006)
            & (value_tokens["y"] < y_next - 0.004)
            & (value_tokens["x"] >= 0.60)
            & (value_tokens["x"] <= 0.69)
        ].sort_values(["y", "x"])

        parsed_blows = []
        for _, br in blow_band.iterrows():
            parsed = parse_blow_cell_token(br["text"])
            if parsed is None:
                continue
            parsed_blows.append(
                {
                    "raw": parsed["raw"],
                    "value": parsed["value"],
                    "is_refusal": parsed["is_refusal"],
                    "x": float(br["x"]),
                    "y": float(br["y"]),
                }
            )

        parsed_blows = merge_slash_split_tokens(parsed_blows)
        blow_values = [bt["value"] for bt in parsed_blows if bt["value"] is not None]
        has_refusal_blow = any(bt["is_refusal"] for bt in parsed_blows)
        calc_n = compute_n_from_blow_values(blow_values, penetration_mode)

        selected_n = None
        if n_candidate == "R" or calc_n == "R" or has_refusal_blow:
            selected_n = "R"
        elif calc_n is not None:
            selected_n = calc_n
        elif isinstance(n_candidate, int):
            selected_n = n_candidate

        if selected_n is None and prev_n == "R":
            selected_n = "R"

        if selected_n is None:
            continue

        payload = build_blow_payload(parsed_blows)
        payload["N_value"] = selected_n
        overrides[round(depth, 2)] = payload
        prev_n = selected_n

    return overrides


# ---------------------------------------------------------------------------
# SPT row extraction from OCR text
# ---------------------------------------------------------------------------


def extract_spt_rows(text: str, meta: dict, page_no: int, source_file: str, spt_n_overrides=None) -> list:
    """
    Parse all SPT test rows from the OCR text of one page.

    *meta* carries state across lines on the same page:
      bh_id, easting, northing, elevation, active_soil_type

    *spt_n_overrides* is a depth-keyed dict from extract_spt_n_overrides_from_page()
    that provides more accurate N-values and blow counts when available.

    Each returned row is a dict whose keys match OUTPUT_COLUMNS.
    """
    if spt_n_overrides is None:
        spt_n_overrides = {}

    rows = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    active_soil_type = meta.get("active_soil_type")

    for idx, line in enumerate(lines):
        # Track the active soil layer as we scan down the page.
        if is_description_update_line(line):
            line_soil = pick_soil_type(line)
            if line_soil:
                active_soil_type = line_soil

        if "spt" not in line.lower():
            continue
        if is_noise_line(line):
            continue

        from_idx = max(0, idx - 1)
        to_idx = min(len(lines), idx + 3)
        context = clean_note(" ".join(lines[from_idx:to_idx]))

        context_soil = pick_soil_type(context)
        soil_for_row = active_soil_type
        if context_soil:
            if soil_for_row is None or (context_soil != soil_for_row and has_strong_inline_soil_signal(context)):
                soil_for_row = context_soil
                active_soil_type = context_soil

        # Parse depth and N-value pairs from the line.
        pair_matches = list(SPT_PAIR_PATTERN.finditer(line))
        if pair_matches:
            for pm in pair_matches:
                n_raw = int(pm.group(1))
                depth = round(float(pm.group(2)), 2)
                n_value = n_raw if 0 <= n_raw <= 100 else None

                # Override with pixmap-based extraction if available.
                override = spt_n_overrides.get(depth, {})
                n_value = override.get("N_value", n_value)
                blow_payload = {k: override.get(k, v) for k, v in empty_blow_payload().items()}

                rqd = pick_metric(context, RQD_PATTERN)
                tcr = pick_metric(context, TCR_PATTERN)
                scr = pick_metric(context, SCR_PATTERN)
                water = pick_water_depth(context)

                row = {
                    "Borehole_ID": meta.get("bh_id"),
                    "Easting": meta.get("easting"),
                    "Northing": meta.get("northing"),
                    "Elevation_m": meta.get("elevation"),
                    "Depth_from": max(0.0, round(depth - 1.5, 2)),
                    "Depth_to": depth,
                    "Soil_Type": soil_for_row,
                    "N_value": n_value,
                    "RQD": rqd,
                    "TCR": tcr,
                    "SCR": scr,
                    "Water_Depth": water,
                    "Notes": clean_note(context),
                    "Page_No": page_no,
                    "Source_File": source_file,
                    **blow_payload,
                }
                rows.append(row)
        else:
            # No explicit "N SPT at Xm" pattern – try to find depth + N-value
            # from the surrounding context.
            depth_match = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*m\b", line)
            if not depth_match:
                continue
            depth = round(float(depth_match.group(1)), 2)
            if not (0 < depth <= 100):
                continue

            line_nvalue = None
            m = re.search(r"\bN\s*[:=]\s*(\d{1,3})\b", line, re.I)
            if m:
                v = int(m.group(1))
                line_nvalue = v if 0 <= v <= 100 else None

            n_value = pick_nvalue(context, depth, line_nvalue)

            override = spt_n_overrides.get(depth, {})
            n_value = override.get("N_value", n_value)
            blow_payload = {k: override.get(k, v) for k, v in empty_blow_payload().items()}

            rqd = pick_metric(context, RQD_PATTERN)
            tcr = pick_metric(context, TCR_PATTERN)
            scr = pick_metric(context, SCR_PATTERN)
            water = pick_water_depth(context)

            row = {
                "Borehole_ID": meta.get("bh_id"),
                "Easting": meta.get("easting"),
                "Northing": meta.get("northing"),
                "Elevation_m": meta.get("elevation"),
                "Depth_from": max(0.0, round(depth - 1.5, 2)),
                "Depth_to": depth,
                "Soil_Type": soil_for_row,
                "N_value": n_value,
                "RQD": rqd,
                "TCR": tcr,
                "SCR": scr,
                "Water_Depth": water,
                "Notes": clean_note(context),
                "Page_No": page_no,
                "Source_File": source_file,
                **blow_payload,
            }
            rows.append(row)

    meta["active_soil_type"] = active_soil_type
    return rows


# ---------------------------------------------------------------------------
# Page-level processor
# ---------------------------------------------------------------------------


def process_page(page, page_no: int, source_file: str, carry_meta: dict) -> list:
    """
    Process one PDF page and return a list of extracted data rows.

    *carry_meta* is a mutable dict that persists across pages within the
    same PDF (borehole ID, coordinates, elevation, active soil layer).
    """
    # --- OCR passes ---
    full_text = ocr_page(page)
    header_text = ocr_page_header(page)
    coord_text = ocr_page_coord_region(page)

    # Combine sources; header/coord regions are higher-fidelity for metadata.
    combined_header = header_text + "\n" + coord_text + "\n" + top_lines(full_text)

    bh_id, easting, northing, elev = extract_header(combined_header)

    # Apply manual overrides when OCR is unreliable.
    if bh_id and bh_id in MANUAL_BH_METADATA:
        override = MANUAL_BH_METADATA[bh_id]
        easting = override.get("Easting", easting)
        northing = override.get("Northing", northing)
        elev = override.get("Elevation_m", elev)

    # Carry forward metadata from previous pages of the same borehole.
    if bh_id:
        carry_meta["bh_id"] = bh_id
    if easting is not None:
        carry_meta["easting"] = easting
    if northing is not None:
        carry_meta["northing"] = northing
    if elev is not None:
        carry_meta["elevation"] = elev

    # --- SPT N-value pixmap pass ---
    spt_depths_hint = extract_spt_depths_from_text(full_text)
    spt_n_overrides = extract_spt_n_overrides_from_page(page, spt_depths_hint or None)

    # --- Row extraction ---
    rows = extract_spt_rows(
        full_text,
        meta=carry_meta,
        page_no=page_no,
        source_file=source_file,
        spt_n_overrides=spt_n_overrides,
    )
    return rows


# ---------------------------------------------------------------------------
# PDF-level processor
# ---------------------------------------------------------------------------


def process_pdf(pdf_path: str) -> list:
    """
    Process all pages in a single PDF and return aggregated data rows.
    """
    source_file = os.path.basename(pdf_path)
    log.info("Processing: %s", source_file)
    all_rows = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        log.error("Cannot open %s: %s", pdf_path, exc)
        return []

    carry_meta: dict = {}

    for page_no, page in enumerate(doc, start=1):
        try:
            rows = process_page(page, page_no, source_file, carry_meta)
            all_rows.extend(rows)
            log.debug("  Page %d: %d rows", page_no, len(rows))
        except Exception as exc:
            log.warning("  Page %d error (%s): %s", page_no, source_file, exc)

    doc.close()
    log.info("  → %d rows extracted from %s", len(all_rows), source_file)
    return all_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline():
    """
    Entry-point: process every PDF in INPUT_DIR and write results to
    OUTPUT_DIR as both an Excel workbook and a CSV file.
    """
    pdf_files = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        log.warning("No PDF files found in '%s'. Place borehole report PDFs there and re-run.", INPUT_DIR)
        return

    all_rows = []
    for pdf_path in pdf_files:
        all_rows.extend(process_pdf(pdf_path))

    if not all_rows:
        log.warning("No data rows extracted from any PDF.")
        return

    df = pd.DataFrame(all_rows)

    # Ensure all expected columns are present (fill missing ones with NaN).
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[OUTPUT_COLUMNS]

    excel_path = os.path.join(OUTPUT_DIR, "borehole_data.xlsx")
    csv_path = os.path.join(OUTPUT_DIR, "borehole_data.csv")

    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)

    log.info("Saved %d rows → %s", len(df), excel_path)
    log.info("Saved %d rows → %s", len(df), csv_path)
    return df


if __name__ == "__main__":
    run_pipeline()
