"""
tests/test_pipeline_csv.py
---------------------------
Integration test: run the full BoreholePipeline on synthetic CSV inputs
that mimic the two contrasting report formats from the problem statement.
"""

import csv
import os
import tempfile
import pytest

import pandas as pd

from borehole_parser import BoreholePipeline


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Format 1 – Riyadh Geotechnique & Foundations style
# ---------------------------------------------------------------------------
FORMAT_1_ROWS = [
    {"DEPTH (M)": "1.0", "DESCRIPTION": "Light brown medium dense poorly graded SAND", "SPT-N": "22", "PENET BU/15CM": "8/10/12", "REC. %": "", "RQD %": ""},
    {"DEPTH (M)": "2.0", "DESCRIPTION": "Light brown medium dense poorly graded SAND", "SPT-N": "21", "PENET BU/15CM": "7/10/11", "REC. %": "", "RQD %": ""},
    {"DEPTH (M)": "3.0", "DESCRIPTION": "Light brown medium dense poorly graded SAND", "SPT-N": "22", "PENET BU/15CM": "8/10/12", "REC. %": "", "RQD %": ""},
    {"DEPTH (M)": "4.0", "DESCRIPTION": "Light brown very dense poorly graded SAND with silt", "SPT-N": "11", "PENET BU/15CM": "5/4/7",  "REC. %": "", "RQD %": ""},
    {"DEPTH (M)": "7.5", "DESCRIPTION": "Light brown very dense poorly graded SAND with silt", "SPT-N": "R",  "PENET BU/15CM": "50/14",   "REC. %": "", "RQD %": ""},
]

# Format 2 – Applus Arabia style
FORMAT_2_ROWS = [
    {"Depth (m)": "0.00", "Reduced Level": "+637.29", "N-Value": "48",  "USCS": "SM", "Description": "FILL MATERIAL Mixture of Silty sand with gravel", "LL": "", "PL": "", "PI": "", "W(C)%": "", "RQD (%)": "", "Fracture_Index": ""},
    {"Depth (m)": "1.00", "Reduced Level": "+636.29", "N-Value": "100", "USCS": "GM", "Description": "FILL MATERIAL Silty gravel with sand",               "LL": "", "PL": "", "PI": "", "W(C)%": "", "RQD (%)": "", "Fracture_Index": ""},
    {"Depth (m)": "2.00", "Reduced Level": "",         "N-Value": "100", "USCS": "GM", "Description": "FILL MATERIAL 2.00 - 3.00 Ditto",                    "LL": "", "PL": "", "PI": "", "W(C)%": "", "RQD (%)": "", "Fracture_Index": ""},
    {"Depth (m)": "3.00", "Reduced Level": "+634.29", "N-Value": "",    "USCS": "",   "Description": "LIMESTONE yellowish white fine to medium texture",     "LL": "", "PL": "", "PI": "", "W(C)%": "94", "RQD (%)": "13", "Fracture_Index": ">10"},
]


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestFormat1Pipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.csv_path = os.path.join(tmp_dir, "format1.csv")
        _write_csv(FORMAT_1_ROWS, self.csv_path)
        pipeline = BoreholePipeline(verbose=False)
        self.result = pipeline.run(self.csv_path, output_dir=tmp_dir, fmt="csv")

    def test_success(self):
        assert self.result.success

    def test_has_depth_column(self):
        assert "depth_m" in self.result.df.columns

    def test_has_spt_column(self):
        assert "spt_n_value" in self.result.df.columns

    def test_has_description_column(self):
        assert "soil_description" in self.result.df.columns

    def test_spt_values_numeric(self):
        n_vals = self.result.df["spt_n_value"].dropna()
        assert len(n_vals) > 0
        for v in n_vals:
            assert isinstance(v, (int, float))

    def test_refusal_mapped_to_50(self):
        n_vals = self.result.df["spt_n_value"].dropna().tolist()
        assert 50 in n_vals   # "R" and "50/14" both → 50

    def test_output_file_created(self):
        assert self.result.output_path is not None
        assert os.path.isfile(self.result.output_path)

    def test_output_csv_readable(self):
        df = pd.read_csv(self.result.output_path)
        assert not df.empty


class TestFormat2Pipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.csv_path = os.path.join(tmp_dir, "format2.csv")
        _write_csv(FORMAT_2_ROWS, self.csv_path)
        pipeline = BoreholePipeline(verbose=False)
        self.result = pipeline.run(self.csv_path, output_dir=tmp_dir, fmt="csv")

    def test_success(self):
        assert self.result.success

    def test_has_elevation_column(self):
        assert "elevation_m" in self.result.df.columns

    def test_has_uscs_column(self):
        assert "uscs_symbol" in self.result.df.columns

    def test_has_rqd_column(self):
        assert "rqd_pct" in self.result.df.columns

    def test_has_n_value_column(self):
        assert "spt_n_value" in self.result.df.columns

    def test_n_value_48(self):
        n_vals = self.result.df["spt_n_value"].dropna().tolist()
        assert 48 in n_vals


class TestExcelOutput:
    def test_excel_output(self, tmp_dir):
        csv_path = os.path.join(tmp_dir, "excel_test.csv")
        _write_csv(FORMAT_1_ROWS, csv_path)
        pipeline = BoreholePipeline(verbose=False)
        result = pipeline.run(csv_path, output_dir=tmp_dir, fmt="excel")
        assert result.success
        assert result.output_path.endswith(".xlsx")
        assert os.path.isfile(result.output_path)


class TestJsonOutput:
    def test_json_output(self, tmp_dir):
        import json
        csv_path = os.path.join(tmp_dir, "json_test.csv")
        _write_csv(FORMAT_2_ROWS, csv_path)
        pipeline = BoreholePipeline(verbose=False)
        result = pipeline.run(csv_path, output_dir=tmp_dir, fmt="json")
        assert result.success
        with open(result.output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "data" in data
        assert len(data["data"]) > 0
