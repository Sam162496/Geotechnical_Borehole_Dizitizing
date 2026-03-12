"""
tests/test_column_mapper.py
----------------------------
Unit tests for the "human-like" column mapping intelligence layer.
These tests verify that arbitrary column headers – as they appear in real
geotechnical borehole log reports – are correctly mapped to standard field
names.
"""

import pytest
from borehole_parser.column_mapper import map_column, map_columns, ColumnMapping
from borehole_parser.field_aliases import normalise


class TestNormalise:
    def test_lowercase(self):
        assert normalise("DEPTH (M)") == "depth (m)"

    def test_strip_whitespace(self):
        assert normalise("  N-Value  ") == "n value"

    def test_hyphen_to_space(self):
        assert normalise("SPT-N") == "spt n"

    def test_collapse_spaces(self):
        assert normalise("SPT  N   VALUE") == "spt n value"


class TestExactMatch:
    """Column headers that exactly match a known alias."""

    def test_depth_m(self):
        m = map_column("DEPTH (M)")
        assert m.standard_field == "depth_m"
        assert m.match_strategy == "exact"
        assert m.confidence == 1.0

    def test_spt_n_value_exact(self):
        m = map_column("N-Value")
        assert m.standard_field == "spt_n_value"
        assert m.match_strategy == "exact"

    def test_rqd(self):
        m = map_column("RQD (%)")
        assert m.standard_field == "rqd_pct"
        assert m.match_strategy == "exact"

    def test_uscs(self):
        m = map_column("USCS")
        assert m.standard_field == "uscs_symbol"

    def test_description(self):
        m = map_column("Description")
        assert m.standard_field == "soil_description"

    def test_liquid_limit(self):
        m = map_column("LL")
        assert m.standard_field == "liquid_limit_pct"

    def test_plastic_limit(self):
        m = map_column("PL")
        assert m.standard_field == "plastic_limit_pct"

    def test_plasticity_index(self):
        m = map_column("PI")
        assert m.standard_field == "plasticity_index"

    def test_water_content(self):
        m = map_column("W(%)")
        assert m.standard_field == "water_content_pct"

    def test_elevation(self):
        m = map_column("Reduced Level")
        assert m.standard_field == "elevation_m"

    def test_tcr(self):
        m = map_column("Rec %")
        assert m.standard_field == "tcr_pct"


class TestContainsMatch:
    """Headers that contain a known alias as a substring (real-world variants)."""

    def test_penet_bu_15cm(self):
        """Saudi/Middle-East SPT column header 'PENET BU/15CM'."""
        m = map_column("PENET BU/15CM")
        assert m.standard_field == "spt_n_value"
        assert m.match_strategy in ("exact", "contains", "fuzzy")

    def test_depth_from_m_verbose(self):
        m = map_column("DEPTH FROM (M)")
        assert m.standard_field == "depth_from_m"

    def test_depth_to_m_verbose(self):
        m = map_column("DEPTH TO (M)")
        assert m.standard_field == "depth_to_m"

    def test_reduced_level_abbrev(self):
        m = map_column("R.L.")
        assert m.standard_field == "elevation_m"


class TestFuzzyMatch:
    """Headers with typos or unexpected word order – fuzzy matching."""

    def test_descriptoin_typo(self):
        m = map_column("DESCRIPTOIN")
        assert m.standard_field == "soil_description"

    def test_spt_variation(self):
        m = map_column("SPT N-VALUE (BLOWS)")
        assert m.standard_field == "spt_n_value"

    def test_fracture_index_variant(self):
        m = map_column("Fracture_Index")
        assert m.standard_field == "fracture_index"

    def test_rqd_variant(self):
        m = map_column("R.Q.D. (%)")
        assert m.standard_field == "rqd_pct"


class TestValueBasedMatch:
    """If header is unrecognised, inspect column values to infer field."""

    def test_depth_from_values(self):
        m = map_column("Z", sample_values=["0.5", "1.0", "1.5", "2.0", "2.5"])
        # Should map to depth_m (or depth_from_m) based on numeric pattern
        assert m.standard_field in ("depth_m", "depth_from_m", "depth_to_m", "elevation_m")

    def test_spt_refusal_values(self):
        """Column with values like '22', 'R', '50/7' → spt_n_value."""
        m = map_column("COL1", sample_values=["22", "11", "R", "50/7", "31"])
        assert m.standard_field == "spt_n_value"


class TestMapColumns:
    """Test bulk column mapping."""

    def test_format_1_headers(self):
        """Riyadh Geotechnique & Foundations report headers."""
        headers = ["DEPTH (M)", "DESCRIPTION", "SPT-N", "PENET BU/15CM", "REC. %", "RQD %"]
        mappings = map_columns(headers)
        assert mappings["DEPTH (M)"].standard_field == "depth_m"
        assert mappings["DESCRIPTION"].standard_field == "soil_description"
        assert mappings["SPT-N"].standard_field == "spt_n_value"
        assert mappings["RQD %"].standard_field == "rqd_pct"

    def test_format_2_headers(self):
        """Applus Arabia borehole log headers."""
        headers = ["LL", "PL", "PI", "W(C)%", "RQD (%)", "Fracture_Index",
                   "N-Value", "USCS", "Description", "Depth (m)", "Reduced Level"]
        mappings = map_columns(headers)
        assert mappings["LL"].standard_field == "liquid_limit_pct"
        assert mappings["PL"].standard_field == "plastic_limit_pct"
        assert mappings["PI"].standard_field == "plasticity_index"
        assert mappings["N-Value"].standard_field == "spt_n_value"
        assert mappings["USCS"].standard_field == "uscs_symbol"
        assert mappings["Description"].standard_field == "soil_description"
        assert mappings["Depth (m)"].standard_field == "depth_m"
        assert mappings["Reduced Level"].standard_field == "elevation_m"

    def test_no_duplicate_standard_fields(self):
        """Two headers mapping to the same field – only one should win."""
        headers = ["DEPTH (M)", "Depth", "SPT-N", "N Value"]
        mappings = map_columns(headers)
        # depth_m should be assigned to exactly one header
        depth_winners = [h for h, m in mappings.items() if m.standard_field == "depth_m"]
        assert len(depth_winners) == 1
        # spt_n_value should be assigned to exactly one header
        spt_winners = [h for h, m in mappings.items() if m.standard_field == "spt_n_value"]
        assert len(spt_winners) == 1

    def test_unresolved_header(self):
        """Completely unrecognisable header stays as is."""
        m = map_column("XYZZY_UNKNOWN_COL_999")
        assert m.match_strategy == "unresolved"
        assert m.confidence == 0.0
