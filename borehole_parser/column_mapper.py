"""
column_mapper.py
----------------
Implements the "human-like thinking" layer for column-header recognition.

Algorithm (cascading, highest confidence first):
  1. Exact match  – normalised header equals a known alias exactly.
  2. Contains match – a known alias is a substring of the normalised header.
  3. Fuzzy match   – rapidfuzz partial-ratio score ≥ FUZZY_THRESHOLD.
  4. Value-pattern match – if no header match, inspect the column's data
     values and compare against known regex/range validators.
  5. Unresolved    – field is kept with the original header name and marked
     with confidence=0.

The mapper returns a ColumnMapping dataclass per column that records:
  - standard_field  : mapped standard field name (or original if unresolved)
  - confidence      : float 0–1
  - match_strategy  : how the match was found
  - original_header : the raw header text as seen in the document
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rapidfuzz import fuzz

from borehole_parser.field_aliases import (
    FIELD_ALIASES,
    VALUE_PATTERNS,
    VALUE_RANGES,
    normalise,
)

# Minimum fuzzy score (0-100) to accept a match.
# 87 was chosen to avoid false positives (e.g. "COL1" → "colour" scores ~86).
# This value can be lowered if your reports contain very non-standard header
# names that need looser matching.
FUZZY_THRESHOLD = 87

# Minimum fraction of column values that must pass VALUE_PATTERNS to accept
# a value-pattern match
VALUE_MATCH_MIN_FRACTION = 0.6


@dataclass
class ColumnMapping:
    original_header: str
    standard_field: str
    confidence: float          # 0.0 – 1.0
    match_strategy: str        # "exact" | "contains" | "fuzzy" | "value" | "unresolved"
    fuzzy_score: Optional[float] = None

    def is_resolved(self) -> bool:
        return self.standard_field != self.original_header

    def __repr__(self) -> str:
        return (
            f"ColumnMapping({self.original_header!r} → {self.standard_field!r} "
            f"[{self.match_strategy}, conf={self.confidence:.2f}])"
        )


# Pre-build a flat lookup: normalised_alias → standard_field
_ALIAS_LOOKUP: Dict[str, str] = {}
for _std_field, _aliases in FIELD_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_LOOKUP[normalise(_alias)] = _std_field


def map_column(
    header: str,
    sample_values: Optional[Sequence] = None,
) -> ColumnMapping:
    """
    Map a single column header to a standard field name.

    Parameters
    ----------
    header : str
        The raw column header as extracted from the document.
    sample_values : sequence, optional
        A sample of non-null values from the column.  Used for value-based
        fallback matching when header matching is inconclusive.

    Returns
    -------
    ColumnMapping
    """
    norm = normalise(header)

    # ── 1. Exact match ──────────────────────────────────────────────────────
    if norm in _ALIAS_LOOKUP:
        return ColumnMapping(
            original_header=header,
            standard_field=_ALIAS_LOOKUP[norm],
            confidence=1.0,
            match_strategy="exact",
        )

    # ── 2. Contains match ───────────────────────────────────────────────────
    # Check if the normalised header *contains* a known alias (e.g. the
    # header "PENET BU/15CM" contains "penet bu/15cm").
    # Require alias to be at least 3 characters to avoid single-letter
    # false positives (e.g. "n" matching "unknown").
    best_contains: Optional[tuple] = None  # (alias_len, std_field, alias)
    for alias_norm, std_field in _ALIAS_LOOKUP.items():
        if alias_norm and len(alias_norm) >= 3 and alias_norm in norm:
            # Prefer longer (more specific) alias matches
            if best_contains is None or len(alias_norm) > best_contains[0]:
                best_contains = (len(alias_norm), std_field, alias_norm)
    if best_contains is not None:
        alias_len, std_field, _ = best_contains
        conf = min(0.95, 0.70 + alias_len / (len(norm) + 1) * 0.25)
        return ColumnMapping(
            original_header=header,
            standard_field=std_field,
            confidence=conf,
            match_strategy="contains",
        )

    # ── 3. Fuzzy match ──────────────────────────────────────────────────────
    # Skip very short aliases (< 3 chars) to prevent false positives where
    # single-char aliases like "n" or "z" match any string that contains them.
    best_score = 0.0
    best_field_fuzzy: Optional[str] = None
    for alias_norm, std_field in _ALIAS_LOOKUP.items():
        if len(alias_norm) < 3:
            continue
        score = fuzz.partial_ratio(norm, alias_norm)
        if score > best_score:
            best_score = score
            best_field_fuzzy = std_field

    if best_score >= FUZZY_THRESHOLD and best_field_fuzzy:
        return ColumnMapping(
            original_header=header,
            standard_field=best_field_fuzzy,
            confidence=best_score / 100.0 * 0.85,
            match_strategy="fuzzy",
            fuzzy_score=best_score,
        )

    # ── 4. Value-pattern match ──────────────────────────────────────────────
    if sample_values:
        sv = [str(v).strip().upper() for v in sample_values if v is not None and str(v).strip()]
        if sv:
            for std_field, pattern in VALUE_PATTERNS.items():
                matched = sum(1 for v in sv if re.match(pattern, v, re.I))
                fraction = matched / len(sv)
                if fraction >= VALUE_MATCH_MIN_FRACTION:
                    # Additional range check for numeric fields
                    lo, hi = VALUE_RANGES.get(std_field, (None, None))
                    if lo is not None:
                        nums = []
                        for v in sv:
                            try:
                                nums.append(float(v))
                            except ValueError:
                                pass
                        # For spt_n_value, non-numeric (R, 50/7) values are valid
                        if std_field == "spt_n_value":
                            if nums and all(lo <= n <= hi for n in nums):
                                return ColumnMapping(
                                    original_header=header,
                                    standard_field=std_field,
                                    confidence=fraction * 0.70,
                                    match_strategy="value",
                                )
                        elif nums and all(lo <= n <= hi for n in nums):
                            return ColumnMapping(
                                original_header=header,
                                standard_field=std_field,
                                confidence=fraction * 0.70,
                                match_strategy="value",
                            )
                    else:
                        return ColumnMapping(
                            original_header=header,
                            standard_field=std_field,
                            confidence=fraction * 0.60,
                            match_strategy="value",
                        )

    # ── 5. Unresolved ───────────────────────────────────────────────────────
    return ColumnMapping(
        original_header=header,
        standard_field=header,   # keep original
        confidence=0.0,
        match_strategy="unresolved",
    )


def map_columns(
    headers: List[str],
    column_samples: Optional[Dict[str, List]] = None,
) -> Dict[str, ColumnMapping]:
    """
    Map a list of column headers, resolving duplicate assignments by keeping
    the highest-confidence match for each standard field.

    Parameters
    ----------
    headers : list of str
    column_samples : dict {header: [sample_values]}, optional

    Returns
    -------
    dict {original_header: ColumnMapping}
    """
    if column_samples is None:
        column_samples = {}

    mappings: Dict[str, ColumnMapping] = {}
    for h in headers:
        m = map_column(h, column_samples.get(h))
        mappings[h] = m

    # De-duplicate: if two headers map to the same standard field keep the
    # higher-confidence one; demote the lower to "unresolved".
    assigned: Dict[str, str] = {}   # standard_field → winning original_header
    for orig, cm in mappings.items():
        sf = cm.standard_field
        if sf in assigned:
            prev_orig = assigned[sf]
            if cm.confidence > mappings[prev_orig].confidence:
                # Demote the previous winner
                mappings[prev_orig] = ColumnMapping(
                    original_header=prev_orig,
                    standard_field=prev_orig,
                    confidence=0.0,
                    match_strategy="unresolved",
                )
                assigned[sf] = orig
            else:
                # Demote the current mapping
                mappings[orig] = ColumnMapping(
                    original_header=orig,
                    standard_field=orig,
                    confidence=0.0,
                    match_strategy="unresolved",
                )
        else:
            assigned[sf] = orig

    return mappings


def rename_dataframe_columns(df, verbose: bool = False):
    """
    Apply column mapping to a pandas DataFrame, renaming columns in-place.

    Parameters
    ----------
    df : pandas.DataFrame
    verbose : bool
        If True, print the mapping summary.

    Returns
    -------
    (renamed_df, mappings_dict)
    """
    import pandas as pd  # local import to avoid hard dep at module load time

    samples: Dict[str, List] = {}
    for col in df.columns:
        vals = df[col].dropna().head(20).tolist()
        samples[str(col)] = [str(v) for v in vals]

    mappings = map_columns([str(c) for c in df.columns], samples)

    rename_map = {}
    for orig, cm in mappings.items():
        if cm.is_resolved():
            rename_map[orig] = cm.standard_field
            if verbose:
                print(
                    f"  {orig!r:35s} → {cm.standard_field!r:25s} "
                    f"({cm.match_strategy}, conf={cm.confidence:.2f})"
                )
        else:
            if verbose:
                print(f"  {orig!r:35s} → [unresolved]")

    df_renamed = df.rename(columns=rename_map)
    return df_renamed, mappings
