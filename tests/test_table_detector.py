"""
tests/test_table_detector.py
-----------------------------
Unit tests for table detection and parsing from structured and text sources.
"""

import pytest
from borehole_parser.extraction.table_detector import (
    parse_structured_table,
    parse_text_table,
    extract_tables,
)


class TestParseStructuredTable:
    def _make_table(self):
        return [
            ["Depth (m)", "Description", "SPT-N", "RQD %"],
            ["1.0", "Brown sandy CLAY", "22", ""],
            ["2.0", "Brown sandy CLAY", "21", ""],
            ["3.0", "Dense SAND", "35", ""],
        ]

    def test_returns_dict(self):
        result = parse_structured_table(self._make_table())
        assert isinstance(result, dict)

    def test_correct_headers(self):
        result = parse_structured_table(self._make_table())
        assert "Depth (m)" in result
        assert "Description" in result
        assert "SPT-N" in result

    def test_correct_row_count(self):
        result = parse_structured_table(self._make_table())
        assert len(result["Depth (m)"]) == 3

    def test_values_correct(self):
        result = parse_structured_table(self._make_table())
        assert result["SPT-N"][0] == "22"
        assert result["Depth (m)"][2] == "3.0"

    def test_empty_table(self):
        assert parse_structured_table([]) is None

    def test_single_row(self):
        assert parse_structured_table([["DEPTH (M)", "DESCRIPTION"]]) is None


class TestParseTextTable:
    SAMPLE_TEXT = (
        "DEPTH (M)  DESCRIPTION                           SPT-N\n"
        "1.0        Light brown medium dense sandy CLAY    22\n"
        "2.0        Light brown medium dense sandy CLAY    21\n"
        "3.0        Dense poorly graded SAND               35\n"
    )

    def test_finds_table(self):
        tables = parse_text_table(self.SAMPLE_TEXT)
        assert len(tables) >= 1

    def test_table_has_correct_columns(self):
        tables = parse_text_table(self.SAMPLE_TEXT)
        assert tables, "No tables detected"
        headers = list(tables[0].keys())
        assert any("DEPTH" in h.upper() for h in headers)

    def test_no_table_in_empty_text(self):
        tables = parse_text_table("")
        assert tables == []


class TestExtractTables:
    def test_prefers_structured_over_text(self):
        structured = [
            [
                ["DEPTH", "DESCRIPTION", "N-VALUE"],
                ["1.0", "Clay", "10"],
                ["2.0", "Sand", "25"],
            ]
        ]
        text = "some unrelated text\nno real table here\n"
        result = extract_tables(text, structured)
        assert result
        assert any("DEPTH" in r or "depth" in str(list(r.keys())).lower() for r in result)

    def test_falls_back_to_text_when_no_structured(self):
        text = (
            "DEPTH (M)  DESCRIPTION          SPT-N\n"
            "1.0        Brown sandy Clay     14\n"
            "2.0        Dense gravel Sand    28\n"
        )
        result = extract_tables(text, structured_tables=None)
        assert result
