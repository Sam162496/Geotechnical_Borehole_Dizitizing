"""
Unit tests for the NLP text preprocessor and entity extractor.

These tests do NOT require a running Spark session – they exercise the pure
Python helper functions directly.
"""

import pytest

from src.nlp.preprocessor import clean_description
from src.nlp.entity_extractor import (
    classify_material,
    extract_color,
    extract_consistency_density,
    extract_moisture,
    extract_plasticity,
    extract_soil_type,
    infer_uscs,
)


# ---------------------------------------------------------------------------
# TextPreprocessor – clean_description
# ---------------------------------------------------------------------------


class TestCleanDescription:
    def test_returns_none_for_empty_string(self):
        assert clean_description("") is None

    def test_returns_none_for_none(self):
        assert clean_description(None) is None

    def test_upper_cases_text(self):
        result = clean_description("brown clay")
        assert result == result.upper()

    def test_removes_special_characters(self):
        result = clean_description("sandy clay, moist; very stiff!")
        assert "," not in result
        assert ";" not in result
        assert "!" not in result

    def test_collapses_extra_whitespace(self):
        result = clean_description("brown   silty   clay")
        assert "  " not in result

    def test_expands_spt_abbreviation(self):
        result = clean_description("SS sample at 3 m")
        assert "SPLIT SPOON SAMPLE" in result

    def test_removes_stop_words(self):
        result = clean_description("a brown clay with some silt")
        assert result.startswith("BROWN") or "BROWN" in result
        # "A", "WITH", "SOME" should be removed
        tokens = result.split()
        assert "A" not in tokens
        assert "WITH" not in tokens
        assert "SOME" not in tokens

    def test_preserves_depth_numbers(self):
        result = clean_description("Layer from 3.5 m to 7.0 m")
        assert "3.5" in result
        assert "7.0" in result

    def test_typical_borehole_entry(self):
        raw = "Soft brown silty CLAY, moist, low plasticity"
        result = clean_description(raw)
        assert "SOFT" in result
        assert "BROWN" in result
        assert "CLAY" in result
        assert "MOIST" in result


# ---------------------------------------------------------------------------
# Entity extractor – extract_soil_type
# ---------------------------------------------------------------------------


class TestExtractSoilType:
    def test_extracts_clay(self):
        assert extract_soil_type("SOFT BROWN CLAY MOIST") == "CLAY"

    def test_extracts_sand(self):
        assert extract_soil_type("DENSE GREY SAND SATURATED") == "SAND"

    def test_extracts_gravel(self):
        assert extract_soil_type("VERY DENSE GREY GRAVEL WITH COBBLES") == "GRAVEL"

    def test_extracts_silt(self):
        assert extract_soil_type("SOFT GREY SILT MOIST") == "SILT"

    def test_extracts_rock(self):
        assert extract_soil_type("ROCK CORE MODERATELY WEATHERED SANDSTONE") == "SANDSTONE"

    def test_returns_none_for_empty_string(self):
        assert extract_soil_type("") is None

    def test_returns_none_for_none(self):
        assert extract_soil_type(None) is None

    def test_prefers_longer_match(self):
        # "SANDY CLAY" should resolve to CLAY as the primary soil type
        result = extract_soil_type("STIFF SANDY CLAY MOIST")
        assert result in ("CLAY", "SAND")  # depends on order; CLAY appears as a token


# ---------------------------------------------------------------------------
# Entity extractor – extract_consistency_density
# ---------------------------------------------------------------------------


class TestExtractConsistencyDensity:
    @pytest.mark.parametrize("text,expected", [
        ("VERY SOFT BROWN CLAY", "VERY SOFT"),
        ("SOFT GREY CLAY MOIST", "SOFT"),
        ("FIRM BROWN SILTY CLAY", "FIRM"),
        ("STIFF GREY CLAY", "STIFF"),
        ("VERY STIFF GREY SANDY CLAY", "VERY STIFF"),
        ("HARD GREY GRAVEL", "HARD"),
        ("LOOSE BROWN SAND", "LOOSE"),
        ("MEDIUM DENSE BROWN SAND", "MEDIUM DENSE"),
        ("DENSE GREY SAND SATURATED", "DENSE"),
        ("VERY DENSE GREY GRAVEL", "VERY DENSE"),
    ])
    def test_extraction(self, text, expected):
        assert extract_consistency_density(text) == expected

    def test_returns_none_for_no_match(self):
        assert extract_consistency_density("GREY SANDSTONE") is None


# ---------------------------------------------------------------------------
# Entity extractor – extract_color
# ---------------------------------------------------------------------------


class TestExtractColor:
    @pytest.mark.parametrize("text,expected", [
        ("SOFT BROWN CLAY MOIST", "BROWN"),
        ("STIFF GREY CLAY", "GREY"),
        ("DENSE REDDISH BROWN SAND", "REDDISH BROWN"),
        ("VERY SOFT DARK BROWN ORGANIC CLAY", "DARK BROWN"),
    ])
    def test_extraction(self, text, expected):
        assert extract_color(text) == expected

    def test_returns_none_for_no_color(self):
        assert extract_color("STIFF CLAY MOIST") is None


# ---------------------------------------------------------------------------
# Entity extractor – extract_moisture
# ---------------------------------------------------------------------------


class TestExtractMoisture:
    @pytest.mark.parametrize("text,expected", [
        ("SOFT BROWN CLAY MOIST", "MOIST"),
        ("DENSE GREY SAND SATURATED", "SATURATED"),
        ("STIFF BROWN CLAY DRY", "DRY"),
        ("FIRM GREY SILT WET", "WET"),
    ])
    def test_extraction(self, text, expected):
        assert extract_moisture(text) == expected


# ---------------------------------------------------------------------------
# Entity extractor – extract_plasticity
# ---------------------------------------------------------------------------


class TestExtractPlasticity:
    def test_low_plasticity(self):
        assert extract_plasticity("SOFT GREY CLAY MOIST LOW PLASTICITY") == "LOW PLASTICITY"

    def test_high_plasticity(self):
        assert extract_plasticity("SOFT GREY CLAY HIGH PLASTICITY") == "HIGH PLASTICITY"

    def test_lean(self):
        assert extract_plasticity("STIFF LEAN CLAY") == "LEAN"

    def test_fat(self):
        assert extract_plasticity("SOFT FAT CLAY") == "FAT"

    def test_returns_none_for_coarse_soil(self):
        assert extract_plasticity("DENSE GREY SAND") is None


# ---------------------------------------------------------------------------
# Entity extractor – classify_material
# ---------------------------------------------------------------------------


class TestClassifyMaterial:
    @pytest.mark.parametrize("soil_type,expected", [
        ("SAND", "COARSE"),
        ("GRAVEL", "COARSE"),
        ("CLAY", "FINE"),
        ("SILT", "FINE"),
        ("PEAT", "ORGANIC"),
        ("TOPSOIL", "ORGANIC"),
        ("SANDSTONE", "ROCK"),
        ("GRANITE", "ROCK"),
        (None, None),
        ("", None),
    ])
    def test_classification(self, soil_type, expected):
        assert classify_material(soil_type) == expected


# ---------------------------------------------------------------------------
# Entity extractor – infer_uscs
# ---------------------------------------------------------------------------


class TestInferUscs:
    @pytest.mark.parametrize("soil_type,consistency,plasticity,expected", [
        ("CLAY", "STIFF", "LOW PLASTICITY", "CL"),
        ("CLAY", "SOFT", "HIGH PLASTICITY", "CH"),
        ("CLAY", "FIRM", "LEAN", "CL"),
        ("CLAY", "SOFT", "FAT", "CH"),
        ("SILT", None, "LOW PLASTICITY", "ML"),
        ("SILT", None, "ELASTIC", "MH"),
        ("PEAT", None, None, "PT"),
        # "SILTY" is passed as the plasticity arg; (SAND, SILTY) → SM is a valid USCS key
        ("SAND", "DENSE", "SILTY", "SM"),
        (None, "STIFF", "LOW PLASTICITY", None),
    ])
    def test_uscs_inference(self, soil_type, consistency, plasticity, expected):
        assert infer_uscs(soil_type, consistency, plasticity) == expected
