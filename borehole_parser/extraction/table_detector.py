"""
table_detector.py
------------------
Detects and parses table structures from:
  1. pdfplumber native tables (already structured as list[list[str]])
  2. Raw page text (heuristic line-by-line parsing)

The module uses two complementary strategies:

Strategy A – Structured tables (from pdfplumber)
  Tables already come as nested lists.  We simply clean them, detect which
  row is the header row, and return a dict-of-lists representation.

Strategy B – Text-based parsing (fallback / augment)
  We scan page text line-by-line looking for:
    • A "header line" that contains known field aliases.
    • Subsequent data lines whose cell count matches the header.
  This strategy handles cases where pdfplumber cannot detect table borders
  (e.g. tab-separated columns, space-aligned columns).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from borehole_parser.column_mapper import map_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_row(line: str, sep: Optional[str] = None) -> List[str]:
    """
    Split a table row into cells.

    Tries tab-split first, then multi-space-split, then falls back to
    single-character delimiters ( | ).
    """
    line = line.rstrip()
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    if " | " in line or "|" in line:
        parts = [c.strip() for c in line.split("|")]
        return [p for p in parts if p]
    # Collapse multiple spaces → treat ≥2 consecutive spaces as delimiter
    parts = re.split(r"  +", line)
    return [p.strip() for p in parts if p.strip()]


def _looks_like_header(cells: List[str], min_geo_fields: int = 2) -> bool:
    """
    Heuristic: does this row look like a column-header row?

    A header row tends to contain words rather than pure numbers and should
    match at least *min_geo_fields* known geotechnical aliases.
    """
    from borehole_parser.column_mapper import map_columns as _mc
    if len(cells) < 2:
        return False
    # Must not be entirely numeric
    numeric_count = sum(1 for c in cells if re.match(r"^[\d.\-+/]+$", c))
    if numeric_count == len(cells):
        return False
    mappings = _mc(cells)
    resolved = sum(1 for m in mappings.values() if m.is_resolved())
    return resolved >= min_geo_fields


def _is_data_row(cells: List[str]) -> bool:
    """
    A data row contains at least one numeric cell.
    """
    return any(re.search(r"\d", c) for c in cells)


# ---------------------------------------------------------------------------
# Strategy A – structured tables from pdfplumber
# ---------------------------------------------------------------------------

def parse_structured_table(
    table: List[List[str]],
) -> Optional[Dict[str, List]]:
    """
    Convert a pdfplumber table (list[list[str]]) into a column-oriented dict.

    Detects which row is the header (tries row 0, then row 1 if 0 is all
    empty, then falls back to positional headers like col_0, col_1 …).

    Returns
    -------
    dict {original_header: [cell, …]}  or  None if table is unusable.
    """
    if not table or len(table) < 2:
        return None

    # Find header row
    header_row_idx = 0
    for idx, row in enumerate(table[:3]):
        non_empty = [c for c in row if c and c.strip()]
        if len(non_empty) >= 2:
            header_row_idx = idx
            break

    headers = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[header_row_idx])]

    # Remove duplicate or empty headers
    seen: Dict[str, int] = {}
    clean_headers = []
    for h in headers:
        if not h or h in seen:
            count = seen.get(h, 0) + 1
            seen[h] = count
            h = f"{h}_{count}" if h else f"col_{len(clean_headers)}"
        seen[h] = seen.get(h, 0) + 1
        clean_headers.append(h)

    # Build column dict
    result: Dict[str, List] = {h: [] for h in clean_headers}
    for row in table[header_row_idx + 1:]:
        if not any(c and c.strip() for c in row):
            continue  # skip blank rows
        for i, h in enumerate(clean_headers):
            val = row[i].strip() if i < len(row) and row[i] else ""
            result[h].append(val)

    # Discard if all columns are empty
    if all(not v for v in result.values()):
        return None

    return result


# ---------------------------------------------------------------------------
# Strategy B – text-based table parsing
# ---------------------------------------------------------------------------

def parse_text_table(
    page_text: str,
    min_columns: int = 3,
) -> List[Dict[str, List]]:
    """
    Scan page text for table-like structures and extract them.

    Parameters
    ----------
    page_text : str
    min_columns : int
        Minimum number of columns a detected table must have.

    Returns
    -------
    list of column-oriented dicts (one per detected table).
    """
    lines = page_text.splitlines()
    tables_found: List[Dict[str, List]] = []

    i = 0
    while i < len(lines):
        cells = _split_row(lines[i])
        if len(cells) >= min_columns and _looks_like_header(cells):
            headers = cells
            data_rows: List[List[str]] = []
            j = i + 1
            # Allow one or two sub-header rows before data starts
            sub_header_skipped = 0
            while j < len(lines) and sub_header_skipped < 2:
                sub_cells = _split_row(lines[j])
                if (
                    len(sub_cells) >= min_columns
                    and _looks_like_header(sub_cells, min_geo_fields=1)
                    and not _is_data_row(sub_cells)
                ):
                    j += 1
                    sub_header_skipped += 1
                else:
                    break

            # Collect data rows
            while j < len(lines):
                row_cells = _split_row(lines[j])
                if not row_cells or (len(row_cells) == 1 and not row_cells[0]):
                    j += 1
                    continue
                # Stop if we hit another header-like row (new table section)
                if _looks_like_header(row_cells) and not _is_data_row(row_cells):
                    break
                if len(row_cells) >= min_columns:
                    data_rows.append(row_cells)
                j += 1

            if data_rows:
                col_dict: Dict[str, List] = {h: [] for h in headers}
                for row in data_rows:
                    for k, h in enumerate(headers):
                        col_dict[h].append(row[k] if k < len(row) else "")
                tables_found.append(col_dict)
                i = j
                continue
        i += 1

    return tables_found


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_tables(
    page_text: str,
    structured_tables: Optional[List[List[List[str]]]] = None,
) -> List[Dict[str, List]]:
    """
    Extract tables from a page using both structured and text-based strategies.

    Parameters
    ----------
    page_text : str
    structured_tables : list of raw pdfplumber tables, optional

    Returns
    -------
    list of column-oriented dicts
    """
    result: List[Dict[str, List]] = []

    # Strategy A: structured
    if structured_tables:
        for tbl in structured_tables:
            parsed = parse_structured_table(tbl)
            if parsed:
                result.append(parsed)

    # Strategy B: text (only if strategy A found nothing useful)
    if not result and page_text:
        result.extend(parse_text_table(page_text))

    return result
