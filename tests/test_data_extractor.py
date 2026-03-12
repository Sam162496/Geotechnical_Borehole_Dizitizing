"""
tests/test_data_extractor.py
-----------------------------
Unit tests for SPT parsing, blow-count splitting, and DataFrame creation.
"""

import pytest
import pandas as pd
from borehole_parser.extraction.data_extractor import (
    parse_spt_string,
    split_blow_counts,
    normalise_description,
    table_to_dataframe,
)


class TestParseSptString:
    def test_integer(self):
        assert parse_spt_string("22") == 22

    def test_refusal_fraction(self):
        assert parse_spt_string("50/7") == 50

    def test_refusal_letter_r(self):
        assert parse_spt_string("R") == 50

    def test_refusal_ref(self):
        assert parse_spt_string("REF") == 50

    def test_three_interval_blows(self):
        # Per ASTM D1586: N = sum of last two 6-inch intervals (seating drive discarded)
        assert parse_spt_string("5/4/7") == 11

    def test_greater_than(self):
        assert parse_spt_string(">10") == 10

    def test_none(self):
        assert parse_spt_string(None) is None

    def test_empty(self):
        assert parse_spt_string("") is None

    def test_np(self):
        assert parse_spt_string("NP") is None

    def test_nv(self):
        assert parse_spt_string("NV") is None

    def test_11(self):
        assert parse_spt_string("11") == 11

    def test_large_value(self):
        assert parse_spt_string("100") == 100


class TestSplitBlowCounts:
    def test_three_intervals(self):
        result = split_blow_counts("5/4/7")
        assert result["blows_1st_6in"] == 5
        assert result["blows_2nd_6in"] == 4
        assert result["blows_3rd_6in"] == 7

    def test_empty(self):
        result = split_blow_counts("")
        assert result["blows_1st_6in"] is None

    def test_not_matching(self):
        result = split_blow_counts("22")
        assert result["blows_1st_6in"] is None


class TestNormaliseDescription:
    def test_expands_sp(self):
        result = normalise_description("SP")
        assert "POORLY GRADED SAND" in result

    def test_expands_cl(self):
        result = normalise_description("CL")
        assert "CLAY OF LOW PLASTICITY" in result

    def test_empty(self):
        assert normalise_description("") == ""

    def test_none(self):
        assert normalise_description(None) == ""


class TestTableToDataFrame:
    def _format_1_table(self):
        """Simulate Riyadh Geotechnique & Foundations table."""
        return {
            "DEPTH (M)": ["1.0", "2.0", "3.0", "4.0"],
            "DESCRIPTION": [
                "Light brown, medium dense, poorly graded SAND with silt.",
                "Light brown, medium dense, poorly graded SAND with silt.",
                "Light brown, medium dense, poorly graded SAND with silt.",
                "Light brown, very dense, poorly graded SAND with silt.",
            ],
            "SPT-N": ["22", "21", "22", "11"],
            "PENET BU/15CM": ["8\n10\n12", "7\n10\n11", "8\n10\n12", "5\n4\n7"],
            "REC. %": ["", "", "", ""],
            "RQD %": ["", "", "", ""],
        }

    def _format_2_table(self):
        """Simulate Applus Arabia table."""
        return {
            "Depth (m)": ["0.00", "1.00", "2.00", "3.00"],
            "Reduced Level": ["+637.29", "+636.29", "", "+634.29"],
            "N-Value": ["48", "100", "100", ""],
            "USCS": ["SM", "GM", "GM", ""],
            "Description": [
                "FILL MATERIAL (Mixture of Silty sand with gravel)",
                "FILL MATERIAL (Silty gravel with sand)",
                "FILL MATERIAL (Silty gravel with sand) 2.00 - 3.00 Ditto",
                "LIMESTONE.",
            ],
            "LL": ["", "", "", ""],
            "PL": ["", "", "", ""],
            "PI": ["", "", "", ""],
        }

    def test_format_1_columns_mapped(self):
        df = table_to_dataframe(self._format_1_table())
        assert "depth_m" in df.columns
        assert "soil_description" in df.columns
        assert "spt_n_value" in df.columns
        assert "rqd_pct" in df.columns

    def test_format_1_spt_numeric(self):
        df = table_to_dataframe(self._format_1_table())
        assert pd.api.types.is_integer_dtype(df["spt_n_value"].dropna()) or \
               df["spt_n_value"].dropna().apply(lambda x: isinstance(x, (int, float))).all()

    def test_format_2_columns_mapped(self):
        df = table_to_dataframe(self._format_2_table())
        assert "depth_m" in df.columns
        assert "elevation_m" in df.columns
        assert "spt_n_value" in df.columns
        assert "uscs_symbol" in df.columns
        assert "soil_description" in df.columns

    def test_format_2_n_values(self):
        df = table_to_dataframe(self._format_2_table())
        n_vals = df["spt_n_value"].dropna().tolist()
        assert 48 in n_vals
        assert 100 in n_vals

    def test_description_clean_column_added(self):
        df = table_to_dataframe(self._format_1_table())
        assert "soil_description_clean" in df.columns

    def test_empty_input(self):
        df = table_to_dataframe({})
        assert df.empty
