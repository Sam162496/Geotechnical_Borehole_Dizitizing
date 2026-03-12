#!/usr/bin/env python3
"""
Geotechnical Borehole Log Digitizer
====================================
Processes PDF borehole logs and extracts SPT (Standard Penetration Test) data
into structured CSV / Excel / JSON output.

Usage
-----
    python process_boreholes.py

Place borehole-log PDFs inside the ``input_reports/`` folder.
Results are written to ``output/``.
"""

import os
import re

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_reports")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Manual overrides for borehole metadata (BH_ID -> {Easting, Northing, Elevation_m}).
# Populate this dict to hard-code coordinates when OCR cannot resolve them.
# Example:
#   MANUAL_BH_METADATA = {
#       "BH01": {"Easting": 123456.0, "Northing": 234567.0, "Elevation_m": 45.0},
#   }
MANUAL_BH_METADATA: dict = {}

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

# Number of surrounding lines used when searching for contextual metrics.
CONTEXT_WINDOW = 8

# Small epsilon for floating-point depth comparisons (avoids equality failures
# caused by binary floating-point representation of 0.5 m steps).
DEPTH_EPSILON = 1e-9

# Tolerance added to the upper bound of np.arange() when building 0.5 m bins so
# that the final bin is always included despite floating-point rounding.
BIN_TOLERANCE = 0.001

# ============================================================
# Regex patterns
# ============================================================

# Matches N-value / depth pairs of the form "N=24/3.0m" or "24@3.0" etc.
# Group 1 = N-value, Group 2 = depth.
PAIR_PATTERN = re.compile(
    r"""
    (?<!\d)
    (\d{1,3})                          # N-value  (group 1)
    \s*[/@]\s*
    (\d{1,2}(?:\.\d{1,2})?)            # depth in metres (group 2)
    \s*m?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches lines where N-value appears first, followed by depth.
# Typical format: "24 3.0", "N=24 at 3.0m", "SPT 24 @ 3.0m"
# Group 1 = N-value, Group 2 = depth.
DEPTH_SPT_PATTERN = re.compile(
    r"""
    (?:(?:N|SPT\s*N?)\s*[=:]?\s*)?    # optional prefix
    (\d{1,3})                          # N-value  (group 1)
    \s+
    (?:at\s+|@\s*|depth\s*[=:]?\s*)?
    (\d{1,2}(?:\.\d{1,2})?)            # depth     (group 2)
    \s*m?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches a standalone depth value such as "3.0m" or "@ 3.0".
# Group 1 = depth.
SPT_DEPTH_PATTERN = re.compile(
    r"""
    (?:@\s*|depth\s*[=:]?\s*)?
    (\d{1,2}(?:\.\d{1,2})?)            # depth (group 1)
    \s*m\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Rock-quality metrics.
RQD_PATTERN = re.compile(r"RQD\s*[=:]\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
TCR_PATTERN = re.compile(r"TCR\s*[=:]\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
SCR_PATTERN = re.compile(r"SCR\s*[=:]\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)

# Ground-water / water-table depth.
_WATER_DEPTH_PATTERN = re.compile(
    r"(?:water|gwl|wl|water\s+table|ground\s+water)"
    r"\s*(?:at|depth|level)?\s*[=:@]?\s*"
    r"(\d+(?:\.\d+)?)\s*m?",
    re.IGNORECASE,
)

# Borehole ID.
_BH_PATTERN = re.compile(
    r"(?:BH|BHN|BOREHOLE|B\.H\.|BORING)[.\s\-_]*(\w+(?:[.\s\-]\w+)?)",
    re.IGNORECASE,
)

# Coordinate / elevation labels.
_EASTING_PATTERN = re.compile(
    r"(?:E(?:asting)?|X)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_NORTHING_PATTERN = re.compile(
    r"(?:N(?:orthing)?|Y)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_ELEVATION_PATTERN = re.compile(
    r"(?:RL|Elevation|Elev(?:ation)?|R\.L\.|Level)\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

# Soil-type keywords used to identify material-description lines.
_SOIL_KEYWORDS = re.compile(
    r"\b(FILL|SAND|CLAY|SILT|GRAVEL|PEAT|LOAM|ROCK|SHALE|LIMESTONE|MUDSTONE|"
    r"SANDSTONE|GRANITE|BASALT|CHALK|MARL|TUFF|ALLUVIUM|COLLUVIUM|LATERITE|"
    r"TOPSOIL|COBBLE|BOULDER|DECOMPOSED)\b",
    re.IGNORECASE,
)

# Blow-count patterns (3-reading and 6-reading forms).
_BLOW_EXTENDED_PATTERN = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)"
)
_BLOW_RAW_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)")

# N-value / refusal label used in SPT tables.
_N_REFUSAL_PATTERN = re.compile(
    r"(?:N\s*=\s*|SPT\s*N\s*=?\s*|N\s*value\s*[=:]\s*)(\d+|R(?:efusal)?)",
    re.IGNORECASE,
)

# ============================================================
# Utility / OCR helpers
# ============================================================


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip non-ASCII noise."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def top_lines(text: str, max_lines: int = 18) -> str:
    """Return at most *max_lines* from the beginning of *text*."""
    return "\n".join(text.splitlines()[:max_lines])


def ocr_page(page) -> str:
    """Return full normalized text from a PyMuPDF page object."""
    try:
        return normalize_text(page.get_text("text"))
    except Exception:
        return ""


def ocr_page_header(page) -> str:
    """Return text from the top 20 % of the page (header band)."""
    try:
        rect = page.rect
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.20)
        return normalize_text(page.get_text("text", clip=clip))
    except Exception:
        return ""


def ocr_page_coord_region(page) -> str:
    """Return text from the upper-right quadrant of the page (coordinates)."""
    try:
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + rect.width * 0.5,
            rect.y0,
            rect.x1,
            rect.y0 + rect.height * 0.20,
        )
        return normalize_text(page.get_text("text", clip=clip))
    except Exception:
        return ""


# ============================================================
# Header / metadata extraction
# ============================================================


def extract_header(text: str):
    """
    Parse *text* for borehole ID, easting, northing, and elevation.

    Returns (bh_id, easting, northing, elevation) where any undetected
    value is *None*.
    """
    bh = easting = northing = elevation = None

    m = _BH_PATTERN.search(text)
    if m:
        bh = m.group(1).strip().upper()

    m = _EASTING_PATTERN.search(text)
    if m:
        try:
            easting = float(m.group(1))
        except ValueError:
            pass

    m = _NORTHING_PATTERN.search(text)
    if m:
        try:
            northing = float(m.group(1))
        except ValueError:
            pass

    m = _ELEVATION_PATTERN.search(text)
    if m:
        try:
            elevation = float(m.group(1))
        except ValueError:
            pass

    return bh, easting, northing, elevation


# ============================================================
# Metric / annotation helpers
# ============================================================


def pick_metric(context, pattern) -> float:
    """Search *context* lines for *pattern*; return the first float match or NaN."""
    for line in context:
        m = pattern.search(line)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return np.nan


def pick_water_depth(context) -> float:
    """Return water-table depth found in *context* lines, or NaN."""
    return pick_metric(context, _WATER_DEPTH_PATTERN)


def clean_note(line: str) -> str:
    """Return the line stripped of leading/trailing whitespace."""
    return line.strip()


def normalize_coordinate_value(value_str: str, direction: str) -> float:
    """
    Correct common OCR leading-digit drift in coordinate strings.

    *direction* is ``'E'`` (Easting) or ``'N'`` (Northing).
    """
    try:
        v = float(re.sub(r"[^\d.]", "", value_str))
    except (ValueError, TypeError):
        return np.nan

    # Scale up 3- or 4-digit values that have lost a leading digit.
    if 100.0 <= v <= 999.0:
        v *= 1000.0
    elif 1000.0 <= v <= 9999.0:
        v *= 100.0

    return v


def empty_blow_payload() -> dict:
    """Return a dict representing an absent blow-count record."""
    return {
        "Blow_Count_Raw": "",
        "Blow_Count_1": np.nan,
        "Blow_Count_2": np.nan,
        "Blow_Count_3": np.nan,
        "Blow_Count_4": np.nan,
        "Blow_Count_5": np.nan,
        "Blow_Count_6": np.nan,
    }


# ============================================================
# SPT override / depth extraction
# ============================================================


def extract_spt_depths_from_text(text: str) -> list:
    """Return a sorted list of unique SPT depths (0 < d <= 100) found in *text*."""
    depths = set()
    for m in SPT_DEPTH_PATTERN.finditer(text):
        try:
            d = float(m.group(1))
            if 0 < d <= 100:
                depths.add(round(d, 2))
        except ValueError:
            pass
    return sorted(depths)


def extract_spt_n_overrides_from_page(page, spt_depths_hint=None) -> dict:
    """
    Scan a PDF page for explicit SPT N-value / blow-count data.

    Returns ``{depth_key: payload_dict}`` where *depth_key* is a float
    rounded to 2 d.p. and *payload_dict* contains N_value and
    Blow_Count_* keys.
    """
    overrides = {}
    try:
        text = normalize_text(page.get_text("text"))
    except Exception:
        return overrides

    for line in text.splitlines():
        depth = None

        dm = SPT_DEPTH_PATTERN.search(line)
        if dm:
            try:
                depth = float(dm.group(1))
            except ValueError:
                pass

        if depth is None and spt_depths_hint:
            for hint_d in spt_depths_hint:
                if str(hint_d) in line or f"{hint_d:.1f}" in line:
                    depth = hint_d
                    break

        if depth is None or not (0 < depth <= 100):
            continue

        depth_key = round(depth, 2)
        payload: dict = {}

        nm = _N_REFUSAL_PATTERN.search(line)
        if nm:
            raw_n = nm.group(1).strip().upper()
            if raw_n.startswith("R"):
                payload["N_value"] = "R"
            else:
                try:
                    payload["N_value"] = int(raw_n)
                except ValueError:
                    pass

        bm6 = _BLOW_EXTENDED_PATTERN.search(line)
        if bm6:
            payload["Blow_Count_Raw"] = "/".join(bm6.group(i) for i in range(1, 7))
            for j, col in enumerate(
                [
                    "Blow_Count_1",
                    "Blow_Count_2",
                    "Blow_Count_3",
                    "Blow_Count_4",
                    "Blow_Count_5",
                    "Blow_Count_6",
                ],
                start=1,
            ):
                try:
                    payload[col] = int(bm6.group(j))
                except ValueError:
                    payload[col] = np.nan
        else:
            bm3 = _BLOW_RAW_PATTERN.search(line)
            if bm3:
                payload["Blow_Count_Raw"] = "/".join(bm3.group(i) for i in range(1, 4))
                for j, col in enumerate(
                    ["Blow_Count_1", "Blow_Count_2", "Blow_Count_3"],
                    start=1,
                ):
                    try:
                        payload[col] = int(bm3.group(j))
                    except ValueError:
                        payload[col] = np.nan

                if "N_value" not in payload:
                    try:
                        payload["N_value"] = int(bm3.group(2)) + int(bm3.group(3))
                    except ValueError:
                        pass

        if payload:
            overrides[depth_key] = payload

    return overrides


def pick_nvalue(context, depth, line_nvalue=None):
    """
    Determine the best SPT N-value for a row.

    Prefers *line_nvalue* when valid, then searches *context* lines for
    an explicit N-value label.  Returns an int or ``np.nan``.
    """
    if line_nvalue is not None and 0 <= line_nvalue <= 100:
        return line_nvalue

    n_pattern = re.compile(
        r"(?:N\s*[=:]?\s*|SPT\s*N?\s*[=:]?\s*)(\d{1,3})",
        re.IGNORECASE,
    )
    for line in context:
        m = n_pattern.search(line)
        if m:
            try:
                n = int(m.group(1))
                if 0 <= n <= 100:
                    return n
            except ValueError:
                pass

    return np.nan


# ============================================================
# SPT row extraction
# ============================================================


def extract_spt_rows(text, meta, page_no, source_file, spt_n_overrides=None):
    """
    Parse *text* (a single normalised PDF page) and return a list of
    SPT row dicts ready to be appended to a DataFrame.
    """
    rows = []
    lines = text.splitlines()
    active_soil_type = meta.get("active_soil_type")

    for i, line in enumerate(lines):
        # Build a contextual window of surrounding lines for metric search.
        ctx_start = max(0, i - CONTEXT_WINDOW)
        ctx_end = min(len(lines), i + CONTEXT_WINDOW + 1)
        context = lines[ctx_start:ctx_end]

        # Update the running soil/material type when we encounter a
        # description line that contains a recognisable soil keyword.
        soil_m = _SOIL_KEYWORDS.search(line)
        if soil_m and len(line.strip()) > 3:
            active_soil_type = line.strip()

        soil_for_row = active_soil_type

        # ── Try multi-pair pattern first ────────────────────────────────
        # Some logs encode several N/depth values on a single line.
        pair_matches = list(PAIR_PATTERN.finditer(line))

        if pair_matches:
            for pm in pair_matches:
                n_value = int(pm.group(1))
                depth = float(pm.group(2))
                if not (0 < depth <= 100 and 0 <= n_value <= 100):
                    continue

                row = {
                    "Borehole_ID": meta["bh"],
                    "Easting": meta["easting"],
                    "Northing": meta["northing"],
                    "Elevation_m": meta["elevation"],
                    "Depth_from": max(depth - 0.5, 0.0),
                    "Depth_to": depth,
                    "Soil_Type": soil_for_row,
                    "N_value": n_value,
                    "Blow_Count_Raw": "",
                    "Blow_Count_1": np.nan,
                    "Blow_Count_2": np.nan,
                    "Blow_Count_3": np.nan,
                    "Blow_Count_4": np.nan,
                    "Blow_Count_5": np.nan,
                    "Blow_Count_6": np.nan,
                    "RQD": pick_metric(context, RQD_PATTERN),
                    "TCR": pick_metric(context, TCR_PATTERN),
                    "SCR": pick_metric(context, SCR_PATTERN),
                    "Water_Depth": pick_water_depth(context),
                    "Notes": clean_note(line),
                    "Page_No": page_no,
                    "Source_File": source_file,
                }
                if spt_n_overrides:
                    depth_key = round(depth, 2)
                    override_payload = spt_n_overrides.get(depth_key)
                    if override_payload is not None:
                        row["N_value"] = override_payload.get("N_value", row["N_value"])
                        for key in [
                            "Blow_Count_Raw",
                            "Blow_Count_1",
                            "Blow_Count_2",
                            "Blow_Count_3",
                            "Blow_Count_4",
                            "Blow_Count_5",
                            "Blow_Count_6",
                        ]:
                            row[key] = override_payload.get(key, row.get(key))
                rows.append(row)
            continue

        m = DEPTH_SPT_PATTERN.search(line)
        line_nvalue = int(m.group(1)) if m else None
        depth = float(m.group(2)) if m else None

        if depth is None:
            m = SPT_DEPTH_PATTERN.search(line)
            if m:
                depth = float(m.group(1))

        if depth is None or not (0 < depth <= 100):
            continue

        n_value = pick_nvalue(context, depth, line_nvalue=line_nvalue)

        row = {
            "Borehole_ID": meta["bh"],
            "Easting": meta["easting"],
            "Northing": meta["northing"],
            "Elevation_m": meta["elevation"],
            "Depth_from": max(depth - 0.5, 0.0),
            "Depth_to": depth,
            "Soil_Type": soil_for_row,
            "N_value": n_value,
            "Blow_Count_Raw": "",
            "Blow_Count_1": np.nan,
            "Blow_Count_2": np.nan,
            "Blow_Count_3": np.nan,
            "Blow_Count_4": np.nan,
            "Blow_Count_5": np.nan,
            "Blow_Count_6": np.nan,
            "RQD": pick_metric(context, RQD_PATTERN),
            "TCR": pick_metric(context, TCR_PATTERN),
            "SCR": pick_metric(context, SCR_PATTERN),
            "Water_Depth": pick_water_depth(context),
            "Notes": clean_note(line),
            "Page_No": page_no,
            "Source_File": source_file,
        }

        if spt_n_overrides:
            depth_key = round(depth, 2)
            override_payload = spt_n_overrides.get(depth_key)
            if override_payload is not None:
                row["N_value"] = override_payload.get("N_value", row["N_value"])
                for key in [
                    "Blow_Count_Raw",
                    "Blow_Count_1",
                    "Blow_Count_2",
                    "Blow_Count_3",
                    "Blow_Count_4",
                    "Blow_Count_5",
                    "Blow_Count_6",
                ]:
                    row[key] = override_payload.get(key, row.get(key))

        rows.append(row)

    # Persist active description material across pages for the same borehole.
    meta["active_soil_type"] = active_soil_type

    return rows


def round_half_meter(value):
    return round(float(value) * 2.0) / 2.0


def regularize_depth_intervals(frame):
    if frame.empty:
        return frame

    interval_rows = []
    group_cols = ["Source_File", "Borehole_ID"]

    for (source_file, bh_id), grp in frame.groupby(group_cols, dropna=False):
        g = grp.copy()
        g["Depth_to"] = pd.to_numeric(g["Depth_to"], errors="coerce")
        g = g.dropna(subset=["Depth_to"])
        if g.empty:
            continue

        g["Depth_to"] = g["Depth_to"].apply(round_half_meter)
        g = g.sort_values(["Depth_to", "Page_No"], na_position="last")

        max_depth = g["Depth_to"].max()
        if pd.isna(max_depth) or max_depth <= 0:
            continue

        # Build 0.5 m bins as requested: 0-0.5, 0.5-1.0, ...
        bins = np.arange(0.5, max_depth + BIN_TOLERANCE, 0.5)

        easting = g["Easting"].dropna().iloc[0] if g["Easting"].notna().any() else np.nan
        northing = g["Northing"].dropna().iloc[0] if g["Northing"].notna().any() else np.nan
        elevation = g["Elevation_m"].dropna().iloc[0] if g["Elevation_m"].notna().any() else np.nan

        per_depth = {}
        for _, row in g.iterrows():
            d = row["Depth_to"]
            if d not in per_depth:
                per_depth[d] = row

        # SPT-N changes at sampled depth and applies downward until next change.
        n_events = []
        for _, row in g.iterrows():
            n_val = row.get("N_value")
            if pd.isna(n_val):
                continue
            n_events.append((row["Depth_to"], n_val))
        n_events.sort(key=lambda x: x[0])

        blow_events = []
        for _, row in g.iterrows():
            depth_to = row["Depth_to"]
            has_blow = False
            for bcol in [
                "Blow_Count_Raw",
                "Blow_Count_1",
                "Blow_Count_2",
                "Blow_Count_3",
                "Blow_Count_4",
                "Blow_Count_5",
                "Blow_Count_6",
            ]:
                val = row.get(bcol)
                if isinstance(val, str) and val.strip():
                    has_blow = True
                elif pd.notna(val):
                    has_blow = True
            if has_blow:
                blow_events.append(
                    (
                        depth_to,
                        {
                            "Blow_Count_Raw": row.get("Blow_Count_Raw", ""),
                            "Blow_Count_1": row.get("Blow_Count_1", np.nan),
                            "Blow_Count_2": row.get("Blow_Count_2", np.nan),
                            "Blow_Count_3": row.get("Blow_Count_3", np.nan),
                            "Blow_Count_4": row.get("Blow_Count_4", np.nan),
                            "Blow_Count_5": row.get("Blow_Count_5", np.nan),
                            "Blow_Count_6": row.get("Blow_Count_6", np.nan),
                        },
                    )
                )
        blow_events.sort(key=lambda x: x[0])

        known_soils = []
        for _, row in g.iterrows():
            soil = row.get("Soil_Type")
            if isinstance(soil, str) and soil.strip():
                known_soils.append((row["Depth_to"], soil.strip()))
        known_soils.sort(key=lambda x: x[0])

        for d_to in bins:
            d_to = round_half_meter(d_to)
            d_from = round_half_meter(max(d_to - 0.5, 0.0))
            exact = per_depth.get(d_to)

            n_value = np.nan
            for ev_depth, ev_n in n_events:
                if ev_depth <= d_from + DEPTH_EPSILON:
                    n_value = ev_n
                else:
                    break

            blow_payload = empty_blow_payload()
            for ev_depth, ev_payload in blow_events:
                if ev_depth <= d_from + DEPTH_EPSILON:
                    blow_payload = ev_payload
                else:
                    break

            # Carry soil forward by depth; if no previous layer exists, use first known layer.
            soil = None
            for k_depth, k_soil in known_soils:
                if k_depth <= d_to:
                    soil = k_soil
                else:
                    break
            if soil is None and known_soils:
                soil = known_soils[0][1]

            interval_rows.append(
                {
                    "Borehole_ID": bh_id,
                    "Easting": easting,
                    "Northing": northing,
                    "Elevation_m": elevation,
                    "Depth_from": d_from,
                    "Depth_to": d_to,
                    "Soil_Type": soil,
                    "N_value": n_value,
                    "Blow_Count_Raw": blow_payload.get("Blow_Count_Raw", ""),
                    "Blow_Count_1": blow_payload.get("Blow_Count_1", np.nan),
                    "Blow_Count_2": blow_payload.get("Blow_Count_2", np.nan),
                    "Blow_Count_3": blow_payload.get("Blow_Count_3", np.nan),
                    "Blow_Count_4": blow_payload.get("Blow_Count_4", np.nan),
                    "Blow_Count_5": blow_payload.get("Blow_Count_5", np.nan),
                    "Blow_Count_6": blow_payload.get("Blow_Count_6", np.nan),
                    "RQD": exact["RQD"] if exact is not None else np.nan,
                    "TCR": exact["TCR"] if exact is not None else np.nan,
                    "SCR": exact["SCR"] if exact is not None else np.nan,
                    "Water_Depth": exact["Water_Depth"] if exact is not None else np.nan,
                    "Notes": exact["Notes"] if exact is not None else "",
                    "Page_No": exact["Page_No"] if exact is not None else np.nan,
                    "Source_File": source_file,
                }
            )

    if not interval_rows:
        return frame

    return pd.DataFrame(interval_rows)


def apply_spt_table_events_to_frame(frame, pdf_path):
    if frame.empty or "Page_No" not in frame.columns:
        return frame

    updated = frame.copy()
    if "N_value" in updated.columns:
        updated["N_value"] = updated["N_value"].astype(object)
    for col in [
        "Blow_Count_Raw",
        "Blow_Count_1",
        "Blow_Count_2",
        "Blow_Count_3",
        "Blow_Count_4",
        "Blow_Count_5",
        "Blow_Count_6",
    ]:
        if col not in updated.columns:
            updated[col] = "" if col == "Blow_Count_Raw" else np.nan
        updated[col] = updated[col].astype(object)

    page_mask = updated["Page_No"].notna() & updated["Notes"].fillna("").astype(str).str.contains(r"\bSPT\b", case=False, regex=True)
    page_rows = updated.loc[page_mask].copy()
    if page_rows.empty:
        return updated

    page_rows["Page_No"] = pd.to_numeric(page_rows["Page_No"], errors="coerce")
    page_rows = page_rows.dropna(subset=["Page_No"])
    if page_rows.empty:
        return updated

    page_rows["Page_No"] = page_rows["Page_No"].astype(int)
    events_by_group = {}

    with fitz.open(pdf_path) as doc:
        for page_no, grp in page_rows.groupby("Page_No"):
            if page_no < 1 or page_no > doc.page_count:
                continue

            bh_values = [str(v).strip() for v in grp["Borehole_ID"].dropna().unique() if str(v).strip()]
            if not bh_values:
                continue
            bh_id = bh_values[0]

            source_values = [str(v).strip() for v in grp["Source_File"].dropna().unique() if str(v).strip()]
            source_file = source_values[0] if source_values else os.path.basename(pdf_path)

            page = doc[page_no - 1]
            hint_depths = {
                round(float(v), 2)
                for v in grp["Depth_to"].tolist()
                if pd.notna(v)
            }
            hint_depths.update(extract_spt_depths_from_text(ocr_page(page)))
            hint_depths = sorted(hint_depths)
            overrides = extract_spt_n_overrides_from_page(page, spt_depths_hint=hint_depths)
            if not overrides:
                continue

            key = (source_file, bh_id)
            events_by_group.setdefault(key, {})
            for depth, payload in overrides.items():
                events_by_group[key][round(float(depth), 2)] = payload

    if not events_by_group:
        return updated

    updated["Depth_from"] = pd.to_numeric(updated["Depth_from"], errors="coerce")
    updated["Depth_to"] = pd.to_numeric(updated["Depth_to"], errors="coerce")

    for (source_file, bh_id), depth_map in events_by_group.items():
        ordered_events = sorted(depth_map.items(), key=lambda x: x[0])
        mask = (updated["Source_File"] == source_file) & (updated["Borehole_ID"] == bh_id)
        if not mask.any():
            continue

        subset = updated.loc[mask].sort_values(["Depth_from", "Depth_to"], na_position="last")
        current_payload = None

        for row_idx, row in subset.iterrows():
            depth_from = row.get("Depth_from")
            if pd.isna(depth_from):
                continue

            for event_depth, payload in ordered_events:
                if event_depth <= depth_from + DEPTH_EPSILON:
                    current_payload = payload
                else:
                    break

            if current_payload is None:
                updated.at[row_idx, "N_value"] = np.nan
                updated.at[row_idx, "Blow_Count_Raw"] = ""
                for col in [
                    "Blow_Count_1",
                    "Blow_Count_2",
                    "Blow_Count_3",
                    "Blow_Count_4",
                    "Blow_Count_5",
                    "Blow_Count_6",
                ]:
                    updated.at[row_idx, col] = np.nan
                continue

            updated.at[row_idx, "N_value"] = current_payload.get("N_value", np.nan)
            updated.at[row_idx, "Blow_Count_Raw"] = current_payload.get("Blow_Count_Raw", "")
            for col in [
                "Blow_Count_1",
                "Blow_Count_2",
                "Blow_Count_3",
                "Blow_Count_4",
                "Blow_Count_5",
                "Blow_Count_6",
            ]:
                updated.at[row_idx, col] = current_payload.get(col, np.nan)

    return updated


############################################
# PDF processing
############################################


def process_pdf(pdf_path):
    rows = []
    source_file = os.path.basename(pdf_path)
    bh_meta_map = {}

    meta = {
        "bh": None,
        "easting": None,
        "northing": None,
        "elevation": None,
        "active_soil_type": None,
    }

    with fitz.open(pdf_path) as doc:
        for page_no in range(1, doc.page_count + 1):
            page = doc[page_no - 1]
            text = normalize_text(page.get_text("text"))
            header_text = ocr_page_header(page)
            coord_text = ocr_page_coord_region(page)

            if len(text) < 80:
                text = ocr_page(page)

            spt_depths_hint = extract_spt_depths_from_text(text)
            spt_n_overrides = extract_spt_n_overrides_from_page(page, spt_depths_hint=spt_depths_hint)

            text_top = top_lines(text, max_lines=18)

            _, e_c, n_c, el_c = extract_header(coord_text)
            bh_h, e_h, n_h, el_h = extract_header(header_text)
            bh_t, e_t, n_t, el_t = extract_header(text_top)

            bh = bh_h or bh_t
            easting = e_c if e_c is not None else (e_h if e_h is not None else e_t)
            northing = n_c if n_c is not None else (n_h if n_h is not None else n_t)
            elev = el_h if el_h is not None else (el_c if el_c is not None else el_t)

            if bh:
                meta["bh"] = bh
            current_bh = meta["bh"]
            if current_bh:
                if current_bh not in bh_meta_map:
                    bh_meta_map[current_bh] = {
                        "easting": None,
                        "northing": None,
                        "elevation": None,
                        "active_soil_type": None,
                    }

                # Update metadata only when BH is explicitly detected on this page.
                # This avoids carrying coordinates from the next borehole's page.
                if bh is not None:
                    if easting is not None and bh_meta_map[current_bh]["easting"] is None:
                        bh_meta_map[current_bh]["easting"] = easting
                    if northing is not None and bh_meta_map[current_bh]["northing"] is None:
                        bh_meta_map[current_bh]["northing"] = northing
                    if elev is not None and bh_meta_map[current_bh]["elevation"] is None:
                        bh_meta_map[current_bh]["elevation"] = elev

                meta["easting"] = bh_meta_map[current_bh]["easting"]
                meta["northing"] = bh_meta_map[current_bh]["northing"]
                meta["elevation"] = bh_meta_map[current_bh]["elevation"]
                meta["active_soil_type"] = bh_meta_map[current_bh]["active_soil_type"]

            rows.extend(extract_spt_rows(text, meta, page_no, source_file, spt_n_overrides=spt_n_overrides))

            if current_bh:
                bh_meta_map[current_bh]["active_soil_type"] = meta.get("active_soil_type")

    # Apply manual BH metadata overrides if provided.
    if MANUAL_BH_METADATA:
        for row in rows:
            bh = row.get("Borehole_ID")
            if not bh or bh not in MANUAL_BH_METADATA:
                continue
            info = MANUAL_BH_METADATA[bh]
            if "Easting" in info:
                row["Easting"] = info["Easting"]
            if "Northing" in info:
                row["Northing"] = info["Northing"]
            if "Elevation_m" in info:
                row["Elevation_m"] = info["Elevation_m"]

    return pd.DataFrame(rows)


############################################
# Batch runner
############################################


def run():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")])

    if not files:
        print("No PDF files found in input_reports")
        return

    all_frames = []
    for file_name in files:
        path = os.path.join(INPUT_DIR, file_name)
        print("Processing:", file_name)
        frame = process_pdf(path)
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        print("No structured SPT rows extracted from the available PDFs")
        return

    final = pd.concat(all_frames, ignore_index=True)

    for col in OUTPUT_COLUMNS:
        if col not in final.columns:
            final[col] = np.nan

    final = final[OUTPUT_COLUMNS]

    # Standardize depth bands at 0.5 m increments for each borehole.
    final = regularize_depth_intervals(final)

    # Re-apply SPT table events (N/refusal + blow counts) on intervalized rows.
    # This keeps the final depth bins aligned with remarks-driven sampling depths.
    for file_name in files:
        pdf_path = os.path.join(INPUT_DIR, file_name)
        final = apply_spt_table_events_to_frame(final, pdf_path)

    # Global coordinate sanitation to fix OCR leading-digit drift.
    final["Easting"] = final["Easting"].apply(
        lambda v: normalize_coordinate_value(str(v), "E") if pd.notna(v) else np.nan
    )
    final["Northing"] = final["Northing"].apply(
        lambda v: normalize_coordinate_value(str(v), "N") if pd.notna(v) else np.nan
    )

    if MANUAL_BH_METADATA:
        normalized_bh = final["Borehole_ID"].fillna("").astype(str).str.strip().str.upper()
        for bh, info in MANUAL_BH_METADATA.items():
            mask = normalized_bh == bh.strip().upper()
            if "Easting" in info:
                final.loc[mask, "Easting"] = info["Easting"]
            if "Northing" in info:
                final.loc[mask, "Northing"] = info["Northing"]
            if "Elevation_m" in info:
                final.loc[mask, "Elevation_m"] = info["Elevation_m"]

    # Forward-fill coordinates and elevation within each borehole group.
    for col in ["Easting", "Northing", "Elevation_m"]:
        final[col] = final.groupby(["Source_File", "Borehole_ID"])[col].ffill().bfill()

    final = final.drop_duplicates(subset=["Source_File", "Borehole_ID", "Depth_to"])
    final = final.sort_values(["Source_File", "Borehole_ID", "Depth_to", "Page_No"], na_position="last")

    csv_path = os.path.join(OUTPUT_DIR, "boreholes.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, "boreholes.xlsx")
    json_path = os.path.join(OUTPUT_DIR, "boreholes.json")

    final.to_csv(csv_path, index=False)
    final.to_excel(xlsx_path, index=False)
    final.to_json(json_path, orient="records", indent=2)

    print(f"Saved {len(final)} rows")
    print("Saved:", csv_path)
    print("Saved:", xlsx_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    run()
