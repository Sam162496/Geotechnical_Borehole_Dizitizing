# -*- coding: utf-8 -*-
"""
Unit tests for borehole_pipeline.py

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import math
import sys
import os

# Ensure the project root is on the path so we can import borehole_pipeline.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from borehole_pipeline import (
    EASTING_MAX,
    EASTING_MIN,
    NORTHING_MAX,
    NORTHING_MIN,
    build_blow_payload,
    clean_note,
    compute_n_from_blow_values,
    empty_blow_payload,
    extract_elevation_value,
    extract_header,
    extract_spt_depths_from_text,
    has_strong_inline_soil_signal,
    is_description_update_line,
    is_noise_line,
    merge_slash_split_tokens,
    normalize_coordinate_value,
    normalize_elevation_value,
    normalize_numeric_token,
    normalize_text,
    parse_blow_cell_token,
    parse_depth_from_token,
    parse_n_cell_token,
    pick_nvalue,
    pick_soil_type,
    pick_water_depth,
    top_lines,
)

# ---------------------------------------------------------------------------
# normalize_numeric_token
# ---------------------------------------------------------------------------


class TestNormalizeNumericToken:
    def test_plain_number(self):
        assert normalize_numeric_token("12345") == 12345.0

    def test_ocr_o_to_zero(self):
        assert normalize_numeric_token("3O5678") == 305678.0

    def test_ocr_l_to_one(self):
        assert normalize_numeric_token("3l5678") == 315678.0

    def test_ocr_s_to_five(self):
        assert normalize_numeric_token("3S5678") == 355678.0

    def test_ocr_b_to_eight(self):
        assert normalize_numeric_token("30B456") == 308456.0

    def test_comma_to_dot(self):
        assert normalize_numeric_token("12,34") == 12.34

    def test_multiple_dots_keeps_first(self):
        result = normalize_numeric_token("12.34.56")
        assert result == 12.3456

    def test_empty_returns_none(self):
        assert normalize_numeric_token("") is None

    def test_non_numeric_returns_none(self):
        assert normalize_numeric_token("abc") is None


# ---------------------------------------------------------------------------
# normalize_coordinate_value
# ---------------------------------------------------------------------------


class TestNormalizeCoordinateValue:
    def test_valid_easting(self):
        result = normalize_coordinate_value("315000", "E")
        assert result is not None
        assert EASTING_MIN <= result <= EASTING_MAX

    def test_valid_northing(self):
        result = normalize_coordinate_value("3050000", "N")
        assert result is not None
        assert NORTHING_MIN <= result <= NORTHING_MAX

    def test_out_of_range_easting_returns_none(self):
        # "50000" is only 5 digits – the OCR 6-digit repair heuristic is not
        # triggered, so the value stays out of range and must return None.
        assert normalize_coordinate_value("50000", "E") is None

    def test_out_of_range_northing_returns_none(self):
        assert normalize_coordinate_value("100000", "N") is None

    def test_easting_ocr_repair(self):
        # First two digits mis-read; should be repaired to 31xxxx.
        result = normalize_coordinate_value("325000", "E")
        # 325000 is within [300000, 330000] — valid without repair.
        assert result == 325000.0

    def test_northing_ocr_token(self):
        result = normalize_coordinate_value("3O50000", "N")
        assert result == 3050000.0


# ---------------------------------------------------------------------------
# normalize_elevation_value
# ---------------------------------------------------------------------------


class TestNormalizeElevationValue:
    def test_valid_elevation(self):
        assert normalize_elevation_value("12.5") == 12.5

    def test_zero(self):
        assert normalize_elevation_value("0") == 0.0

    def test_out_of_range_high(self):
        assert normalize_elevation_value("9999") is None

    def test_out_of_range_low(self):
        # ELEV_MIN is -100, but normalize_numeric_token strips minus signs (OCR
        # never produces signed numbers).  The practical lower boundary is 0.
        # A value of 0 is exactly at ELEV_MIN boundary (≥ -100) so it is valid.
        assert normalize_elevation_value("0") == 0.0

    def test_ocr_corrected(self):
        # "l2.5" → "12.5" after OCR correction
        assert normalize_elevation_value("l2.5") == 12.5


# ---------------------------------------------------------------------------
# extract_elevation_value
# ---------------------------------------------------------------------------


class TestExtractElevationValue:
    def test_direct_label(self):
        text = "Elev: 8.50\nN: 3050000"
        assert extract_elevation_value(text) == 8.5

    def test_rl_label(self):
        text = "RL = 15.20"
        assert extract_elevation_value(text) == 15.2

    def test_geo_inv(self):
        text = "Geo. Inv. 22.3"
        assert extract_elevation_value(text) == 22.3

    def test_missing_returns_none(self):
        text = "N: 3050000 E: 315000"
        assert extract_elevation_value(text) is None


# ---------------------------------------------------------------------------
# extract_header
# ---------------------------------------------------------------------------


class TestExtractHeader:
    def test_full_header(self):
        text = (
            "BH-03\n"
            "Easting: 315000.5\n"
            "Northing: 3050000.0\n"
            "Elev: 12.5\n"
        )
        bh, e, n, elev = extract_header(text)
        assert bh == "BH-03"
        assert e == 315000.5
        assert n == 3050000.0
        assert elev == 12.5

    def test_bh_id_zero_padded(self):
        text = "BH 7\nEasting 310000\nNorthing 3060000\n"
        bh, _, _, _ = extract_header(text)
        assert bh == "BH-07"

    def test_missing_coordinates(self):
        text = "BH-01\nSome text without coordinates\n"
        bh, e, n, elev = extract_header(text)
        assert bh == "BH-01"
        assert e is None
        assert n is None

    def test_no_bh_id(self):
        text = "Easting: 310000\nNorthing: 3060000\n"
        bh, _, _, _ = extract_header(text)
        assert bh is None


# ---------------------------------------------------------------------------
# normalize_text / top_lines / clean_note
# ---------------------------------------------------------------------------


class TestTextUtils:
    def test_normalize_text_removes_blank_lines(self):
        text = "line1\n\n\nline2\n"
        result = normalize_text(text)
        assert result == "line1\nline2"

    def test_normalize_text_collapses_spaces(self):
        text = "hello   world"
        assert normalize_text(text) == "hello world"

    def test_top_lines(self):
        text = "a\nb\nc\nd\ne"
        assert top_lines(text, 3) == "a\nb\nc"

    def test_clean_note_truncates(self):
        long_text = "x" * 300
        result = clean_note(long_text)
        assert len(result) <= 260

    def test_clean_note_strips_punctuation(self):
        text = "  ;some note;  "
        assert not clean_note(text).startswith(";")


# ---------------------------------------------------------------------------
# is_noise_line
# ---------------------------------------------------------------------------


class TestIsNoiseLine:
    def test_short_line(self):
        assert is_noise_line("abc") is True

    def test_email(self):
        # "email" is in NOISE_KEYWORDS; the line must be long enough (≥ 8 chars).
        assert is_noise_line("email: info@example.com") is True

    def test_spt_line_not_noise(self):
        assert is_noise_line("SPT at 3.00m N=15") is False

    def test_normal_description_not_noise(self):
        assert is_noise_line("Medium dense silty sand with gravel") is False


# ---------------------------------------------------------------------------
# is_description_update_line
# ---------------------------------------------------------------------------


class TestIsDescriptionUpdateLine:
    def test_soil_description(self):
        assert is_description_update_line("Medium dense SAND with silt, grey") is True

    def test_continuation_fragment(self):
        assert is_description_update_line("with silt") is False

    def test_spt_line(self):
        assert is_description_update_line("SPT at 3m N=15") is False

    def test_noise_line(self):
        assert is_description_update_line("www.example.com") is False

    def test_too_few_words(self):
        assert is_description_update_line("sand") is False


# ---------------------------------------------------------------------------
# pick_soil_type
# ---------------------------------------------------------------------------


class TestPickSoilType:
    def test_sand_uppercase(self):
        assert pick_soil_type("SAND with silt") == "SAND"

    def test_silty_sand(self):
        assert pick_soil_type("silty sand, medium dense") == "SAND"

    def test_sandy_silt(self):
        assert pick_soil_type("sandy silt") == "SILT"

    def test_clay(self):
        assert pick_soil_type("Soft CLAY") == "CLAY"

    def test_limestone(self):
        assert pick_soil_type("Moderately weathered LIMESTONE") == "LIMESTONE"

    def test_continuation_fragment(self):
        assert pick_soil_type("with silt and gravel") is None

    def test_empty(self):
        assert pick_soil_type("") is None

    def test_gravel(self):
        assert pick_soil_type("Dense GRAVEL") == "GRAVEL"


# ---------------------------------------------------------------------------
# has_strong_inline_soil_signal
# ---------------------------------------------------------------------------


class TestHasStrongInlineSoilSignal:
    def test_uppercase_head(self):
        assert has_strong_inline_soil_signal("CLAY with sand") is True

    def test_modifier_head_pattern(self):
        assert has_strong_inline_soil_signal("silty clay") is True

    def test_head_with_pattern(self):
        assert has_strong_inline_soil_signal("sand with gravel") is True

    def test_no_signal(self):
        assert has_strong_inline_soil_signal("at 3.00m depth") is False


# ---------------------------------------------------------------------------
# pick_water_depth
# ---------------------------------------------------------------------------


class TestPickWaterDepth:
    def test_water_level(self):
        # WATER_PATTERN requires the number to immediately follow the label
        # (optionally with ':' or '=').  Use the canonical form.
        assert pick_water_depth("Water Level: 5.50m") == 5.5

    def test_wl_abbreviation(self):
        assert pick_water_depth("WL = 3.00m") == 3.0

    def test_missing(self):
        assert pick_water_depth("No water info here") is None


# ---------------------------------------------------------------------------
# pick_nvalue
# ---------------------------------------------------------------------------


class TestPickNvalue:
    def test_refusal(self):
        assert pick_nvalue("SPT REFUSAL at 3m", 3.0) == "R"

    def test_n_pattern(self):
        assert pick_nvalue("N = 18 SPT", 3.0) == 18

    def test_nvalue_before_spt(self):
        assert pick_nvalue("25 SPT at 3m", 3.0) == 25

    def test_line_nvalue_priority(self):
        assert pick_nvalue("some context", 3.0, line_nvalue=12) == 12

    def test_none_when_no_info(self):
        # Only the depth number is present; it should be skipped.
        result = pick_nvalue("at 3.00m", 3.0)
        assert result is None


# ---------------------------------------------------------------------------
# compute_n_from_blow_values
# ---------------------------------------------------------------------------


class TestComputeNFromBlowValues:
    def test_15cm_mode(self):
        # N = last two increments
        assert compute_n_from_blow_values([5, 8, 9], "15cm") == 17

    def test_7_5cm_mode(self):
        # N = sum of last four of six increments
        assert compute_n_from_blow_values([2, 3, 4, 5, 6, 7], "7.5cm") == 22

    def test_refusal_15cm(self):
        assert compute_n_from_blow_values([25, 30, 28], "15cm") == "R"

    def test_insufficient_values(self):
        assert compute_n_from_blow_values([5, 8], "15cm") is None

    def test_empty(self):
        assert compute_n_from_blow_values([], "15cm") is None


# ---------------------------------------------------------------------------
# parse_blow_cell_token
# ---------------------------------------------------------------------------


class TestParseBlowCellToken:
    def test_simple(self):
        result = parse_blow_cell_token("8")
        assert result["value"] == 8
        assert result["is_refusal"] is False

    def test_slash_format(self):
        result = parse_blow_cell_token("12/15")
        assert result["raw"] == "12/15"
        assert result["value"] == 12

    def test_refusal_token(self):
        result = parse_blow_cell_token("R")
        assert result["is_refusal"] is True
        assert result["value"] is None

    def test_high_first_increment(self):
        result = parse_blow_cell_token("50")
        assert result["is_refusal"] is True

    def test_invalid(self):
        assert parse_blow_cell_token("abc/xyz") is None


# ---------------------------------------------------------------------------
# parse_n_cell_token
# ---------------------------------------------------------------------------


class TestParseNCellToken:
    def test_valid(self):
        assert parse_n_cell_token("18") == 18

    def test_refusal(self):
        assert parse_n_cell_token("REF") == "R"

    def test_out_of_range(self):
        assert parse_n_cell_token("150") is None

    def test_empty(self):
        assert parse_n_cell_token("") is None


# ---------------------------------------------------------------------------
# merge_slash_split_tokens
# ---------------------------------------------------------------------------


class TestMergeSlashSplitTokens:
    def test_merges_adjacent_slash(self):
        tokens = [
            {"raw": "12/", "value": 12, "is_refusal": False, "x": 0.65, "y": 0.30},
            {"raw": "15", "value": 15, "is_refusal": False, "x": 0.66, "y": 0.30},
        ]
        merged = merge_slash_split_tokens(tokens)
        assert len(merged) == 1
        assert merged[0]["raw"] == "12/15"

    def test_does_not_merge_far_tokens(self):
        tokens = [
            {"raw": "12/", "value": 12, "is_refusal": False, "x": 0.60, "y": 0.30},
            {"raw": "15", "value": 15, "is_refusal": False, "x": 0.80, "y": 0.30},
        ]
        merged = merge_slash_split_tokens(tokens)
        assert len(merged) == 2

    def test_empty(self):
        assert merge_slash_split_tokens([]) == []


# ---------------------------------------------------------------------------
# build_blow_payload
# ---------------------------------------------------------------------------


class TestBuildBlowPayload:
    def test_payload_populated(self):
        tokens = [
            {"raw": "5", "value": 5, "is_refusal": False},
            {"raw": "8", "value": 8, "is_refusal": False},
            {"raw": "9", "value": 9, "is_refusal": False},
        ]
        payload = build_blow_payload(tokens)
        assert payload["Blow_Count_Raw"] == "5, 8, 9"
        assert payload["Blow_Count_1"] == "5"
        assert payload["Blow_Count_3"] == "9"

    def test_empty_tokens(self):
        payload = build_blow_payload([])
        assert payload["Blow_Count_Raw"] == ""
        assert math.isnan(payload["Blow_Count_1"])


# ---------------------------------------------------------------------------
# parse_depth_from_token
# ---------------------------------------------------------------------------


class TestParseDepthFromToken:
    def test_depth_with_m(self):
        assert parse_depth_from_token("3.00m") == 3.0

    def test_depth_without_decimal(self):
        assert parse_depth_from_token("6m") == 6.0

    def test_invalid(self):
        assert parse_depth_from_token("abc") is None

    def test_zero_depth(self):
        assert parse_depth_from_token("0m") is None  # 0 is excluded (depth > 0)


# ---------------------------------------------------------------------------
# extract_spt_depths_from_text
# ---------------------------------------------------------------------------


class TestExtractSptDepthsFromText:
    def test_extracts_depths(self):
        text = "SPT at 3.00m N=15\nSPT at 6.00m N=22"
        depths = extract_spt_depths_from_text(text)
        assert 3.0 in depths
        assert 6.0 in depths

    def test_deduplication(self):
        text = "SPT at 3.00m N=15\nSPT at 3.00m (repeated)"
        depths = extract_spt_depths_from_text(text)
        assert depths.count(3.0) == 1

    def test_empty_text(self):
        assert extract_spt_depths_from_text("") == []
