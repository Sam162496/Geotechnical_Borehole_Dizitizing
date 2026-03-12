"""
data_extractor.py
------------------
Converts raw column-oriented table dicts (produced by table_detector.py) into
structured pandas DataFrames with standardised column names and cleaned values.

Key responsibilities:
  1. Apply column_mapper to rename headers to standard field names.
  2. Clean and type-cast each standard field.
  3. Post-process special fields:
       – Parse SPT N-values from strings like "50/7" (refusal), "R", ">50".
       – Split combined blow-count strings (e.g., "5/4/7") into three fields.
       – Convert depth columns to float.
       – Normalise soil description text.
  4. Add a 'layer_thickness_m' column where depth_from_m and depth_to_m exist.
  5. Merge multiple table fragments from the same page into one DataFrame.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from borehole_parser.column_mapper import rename_dataframe_columns


# ---------------------------------------------------------------------------
# SPT-specific parsing helpers
# ---------------------------------------------------------------------------

_REFUSAL_PATTERN = re.compile(r"^(?:R|REF|REFUSAL)[./]?\s*(\d+)?$", re.I)
_FRACTION_PATTERN = re.compile(r"^(\d+)/(\d+)$")          # e.g. "50/7"
_BLOWS_3_PATTERN  = re.compile(r"^(\d+)[/,](\d+)[/,](\d+)$")  # e.g. "5/4/7"
_GT_PATTERN       = re.compile(r"^>(\d+)$")                # e.g. ">10"


def parse_spt_string(raw: str) -> Optional[int]:
    """
    Convert a raw SPT string to an integer N-value.

    Rules (per ASTM D1586 / BS EN ISO 22476-3):
      "22"         → 22
      "50/7"       → 50   (refusal – use 50 as conventional cap)
      "R" / "REF"  → 50   (refusal without blow count)
      "5/4/7"      → 11   (last two 6-inch intervals only; first interval
                            is discarded as the seating drive per standard
                            geotechnical practice)
      ">10"        → 10   (lower bound)
      ""  / None   → None
    """
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s in ("-", "NP", "NV", "N.P.", "N.V."):
        return None

    # Pure integer
    if s.isdigit():
        return int(s)

    # Refusal with depth: "50/7" → 50
    m = _FRACTION_PATTERN.match(s)
    if m:
        return int(m.group(1))

    # Three-interval blow counts: "5/4/7" → 4+7=11
    # Per ASTM D1586, the N-value is the sum of the LAST TWO 6-inch drive
    # intervals; the first (seating) interval is not counted.
    m = _BLOWS_3_PATTERN.match(s)
    if m:
        return int(m.group(2)) + int(m.group(3))

    # Pure refusal: "R", "REF"
    m = _REFUSAL_PATTERN.match(s)
    if m:
        return 50   # conventional geotechnical practice

    # Greater-than notation: ">10"
    m = _GT_PATTERN.match(s)
    if m:
        return int(m.group(1))

    return None


def split_blow_counts(raw: str) -> Dict[str, Optional[int]]:
    """
    Split a "5/4/7" blow-count string into three interval fields.

    Returns
    -------
    dict with keys: blows_1st_6in, blows_2nd_6in, blows_3rd_6in
    """
    empty = {"blows_1st_6in": None, "blows_2nd_6in": None, "blows_3rd_6in": None}
    if not raw:
        return empty
    m = _BLOWS_3_PATTERN.match(str(raw).strip())
    if m:
        return {
            "blows_1st_6in": int(m.group(1)),
            "blows_2nd_6in": int(m.group(2)),
            "blows_3rd_6in": int(m.group(3)),
        }
    return empty


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ABBREV_MAP = {
    r"\bSP\b": "POORLY GRADED SAND",
    r"\bSW\b": "WELL GRADED SAND",
    r"\bGP\b": "POORLY GRADED GRAVEL",
    r"\bGW\b": "WELL GRADED GRAVEL",
    r"\bML\b": "SILT OF LOW PLASTICITY",
    r"\bMH\b": "SILT OF HIGH PLASTICITY",
    r"\bCL\b": "CLAY OF LOW PLASTICITY",
    r"\bCH\b": "CLAY OF HIGH PLASTICITY",
    r"\bSM\b": "SILTY SAND",
    r"\bSC\b": "CLAYEY SAND",
    r"\bGM\b": "SILTY GRAVEL",
    r"\bGC\b": "CLAYEY GRAVEL",
    r"\bPT\b": "PEAT",
    r"\bOL\b": "ORGANIC SILT/CLAY LOW PLASTICITY",
    r"\bOH\b": "ORGANIC CLAY HIGH PLASTICITY",
}


def normalise_description(text: str) -> str:
    """Clean and expand abbreviations in a soil description string."""
    if not isinstance(text, str):
        return ""
    t = text.upper().strip()
    for pattern, expansion in _ABBREV_MAP.items():
        t = re.sub(pattern, expansion, t)
    t = re.sub(r"\s+", " ", t)
    return t


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def table_to_dataframe(
    col_dict: Dict[str, List],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert a column-oriented dict to a standardised DataFrame.

    Parameters
    ----------
    col_dict : dict {header: [values]}
    verbose : bool

    Returns
    -------
    pd.DataFrame with standardised column names and cleaned values.
    """
    if not col_dict:
        return pd.DataFrame()

    df = pd.DataFrame(col_dict)
    df, mappings = rename_dataframe_columns(df, verbose=verbose)

    # ── Numeric coercions ─────────────────────────────────────────────────
    for col in ("depth_from_m", "depth_to_m", "depth_m",
                "rqd_pct", "tcr_pct", "scr_pct",
                "water_content_pct", "liquid_limit_pct",
                "plastic_limit_pct", "plasticity_index",
                "elevation_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")

    # ── SPT N-value ────────────────────────────────────────────────────────
    if "spt_n_value" in df.columns:
        df["spt_n_value"] = df["spt_n_value"].apply(
            lambda v: parse_spt_string(str(v)) if pd.notna(v) and str(v).strip() else None
        )

    # ── Blow-count split ───────────────────────────────────────────────────
    if "blow_counts" in df.columns:
        bc_expanded = df["blow_counts"].apply(
            lambda v: split_blow_counts(str(v)) if pd.notna(v) else split_blow_counts("")
        )
        bc_df = pd.DataFrame(bc_expanded.tolist())
        df = pd.concat([df, bc_df], axis=1)

    # ── Soil description normalisation ─────────────────────────────────────
    if "soil_description" in df.columns:
        df["soil_description_clean"] = df["soil_description"].apply(normalise_description)

    # ── Layer thickness ────────────────────────────────────────────────────
    if "depth_from_m" in df.columns and "depth_to_m" in df.columns:
        df["layer_thickness_m"] = (
            pd.to_numeric(df["depth_to_m"], errors="coerce")
            - pd.to_numeric(df["depth_from_m"], errors="coerce")
        ).round(3)

    # ── Drop rows that are entirely empty ──────────────────────────────────
    df = df.dropna(how="all").reset_index(drop=True)

    return df


def merge_tables(
    dfs: List[pd.DataFrame],
    prefer_longer: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple DataFrames extracted from the same document.

    If all DataFrames share the same columns they are concatenated.
    Otherwise the 'largest' (most columns) is used as the primary table
    and others are dropped (with a warning).

    Parameters
    ----------
    dfs : list of pd.DataFrame
    prefer_longer : bool
        When True prefer the table with most columns if they differ.

    Returns
    -------
    pd.DataFrame
    """
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]

    # Try concat if columns are compatible
    col_sets = [set(d.columns) for d in dfs]
    if len(set(frozenset(cs) for cs in col_sets)) == 1:
        return pd.concat(dfs, ignore_index=True)

    # Different columns: return the one with the most standard columns
    def _std_count(d: pd.DataFrame) -> int:
        from borehole_parser.field_aliases import STANDARD_FIELDS
        return sum(1 for c in d.columns if c in STANDARD_FIELDS)

    dfs_sorted = sorted(dfs, key=_std_count, reverse=True)
    return dfs_sorted[0]
