"""
Integration tests for the BoreholePipeline.

These tests spin up a local PySpark session and run the full pipeline against
the sample CSV shipped in data/sample/.
"""

import os

import pytest

pytest.importorskip("pyspark", reason="PySpark not installed – skipping pipeline tests")

from pyspark.sql import SparkSession

from src.pipeline.borehole_pipeline import BoreholePipeline, _parse_spt


# ---------------------------------------------------------------------------
# Shared Spark session (module-level fixture for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("test_pipeline")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def sample_csv():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "sample", "sample_borehole_log.csv")
    assert os.path.exists(path), f"Sample CSV not found: {path}"
    return path


# ---------------------------------------------------------------------------
# _parse_spt helper
# ---------------------------------------------------------------------------

class TestParseSpt:
    def test_numeric_string(self):
        assert _parse_spt("28") == 28

    def test_refusal_notation(self):
        assert _parse_spt("50+") == 50

    def test_empty_string(self):
        assert _parse_spt("") is None

    def test_none(self):
        assert _parse_spt(None) is None

    def test_whitespace_only(self):
        assert _parse_spt("  ") is None


# ---------------------------------------------------------------------------
# BoreholePipeline integration tests
# ---------------------------------------------------------------------------

class TestBoreholePipeline:
    def test_ingest_loads_rows(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        assert df.count() > 0

    def test_ingest_has_expected_columns(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        expected = {"borehole_id", "depth_from_m", "depth_to_m", "description"}
        assert expected.issubset(set(df.columns))

    def test_validate_drops_null_borehole_id(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        count_before = df.count()
        df_valid = pipeline.validate(df)
        # All sample rows should have borehole_id so count is unchanged
        assert df_valid.count() == count_before

    def test_validate_raises_on_missing_column(self, spark):
        pipeline = BoreholePipeline(spark=spark)
        df = spark.createDataFrame([(1,)], ["wrong_col"])
        with pytest.raises(ValueError, match="missing required columns"):
            pipeline.validate(df)

    def test_preprocess_adds_cleaned_description(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        df = pipeline.validate(df)
        df = pipeline.preprocess(df)
        assert "cleaned_description" in df.columns

    def test_extract_entities_adds_soil_type(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        df = pipeline.validate(df)
        df = pipeline.preprocess(df)
        df = pipeline.extract_entities(df)
        assert "soil_type" in df.columns
        assert "consistency_density" in df.columns
        assert "color" in df.columns
        assert "material_class" in df.columns
        assert "uscs_symbol" in df.columns

    def test_enrich_adds_layer_thickness(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.ingest(sample_csv)
        df = pipeline.validate(df)
        df = pipeline.preprocess(df)
        df = pipeline.extract_entities(df)
        df = pipeline.enrich(df)
        assert "layer_thickness_m" in df.columns
        # Verify a known row: BH-01 first row is 0.0 – 0.5 → 0.5 m
        row = df.filter(
            (df["borehole_id"] == "BH-01") & (df["depth_from_m"] == 0.0)
        ).first()
        assert row is not None
        assert abs(row["layer_thickness_m"] - 0.5) < 0.001

    def test_run_returns_dataframe(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        assert df is not None
        assert df.count() > 0

    def test_run_soil_type_extraction(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        soil_types = {row["soil_type"] for row in df.select("soil_type").collect() if row["soil_type"]}
        # The sample contains at least CLAY, SAND, and GRAVEL
        assert "CLAY" in soil_types
        assert "SAND" in soil_types
        assert "GRAVEL" in soil_types

    def test_run_spt_parsed_as_integer(self, spark, sample_csv):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        spt_col = df.select("spt_n_value").filter(df["spt_n_value"].isNotNull())
        # All non-null SPT values should be integers (no "50+" strings remaining)
        for row in spt_col.collect():
            assert isinstance(row["spt_n_value"], int)

    def test_export_csv(self, spark, sample_csv, tmp_path):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        out = str(tmp_path / "output")
        pipeline.export(df, out, output_format="csv")
        files = list((tmp_path / "output").iterdir())
        assert any(f.suffix == ".csv" for f in files)

    def test_export_json(self, spark, sample_csv, tmp_path):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        out = str(tmp_path / "output_json")
        pipeline.export(df, out, output_format="json")
        files = list((tmp_path / "output_json").iterdir())
        assert any(f.suffix == ".json" for f in files)

    def test_export_invalid_format_raises(self, spark, sample_csv, tmp_path):
        pipeline = BoreholePipeline(spark=spark)
        df = pipeline.run(input_path=sample_csv)
        with pytest.raises(ValueError, match="Unsupported output format"):
            pipeline.export(df, str(tmp_path / "out"), output_format="xlsx")
