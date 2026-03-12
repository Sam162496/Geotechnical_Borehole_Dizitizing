"""
Main PySpark pipeline for geotechnical borehole log digitizing.

Pipeline stages
---------------
1. **Ingest** – Load raw CSV borehole log data into a Spark DataFrame.
2. **Validate** – Check required columns and drop rows with critical nulls.
3. **Preprocess** – Clean and normalise description text (TextPreprocessor).
4. **Extract** – Run NLP entity extraction (GeotechnicalEntityExtractor).
5. **Enrich** – Compute derived numeric columns (layer thickness, numeric SPT).
6. **Export** – Write the enriched DataFrame to the configured output path.

Usage (from command line)
-------------------------
.. code-block:: bash

    python -m src.pipeline.borehole_pipeline \\
        --input  data/sample/sample_borehole_log.csv \\
        --output output/processed_borehole_log.parquet \\
        --format parquet

Usage (as a library)
--------------------
.. code-block:: python

    from src.pipeline.borehole_pipeline import BoreholePipeline

    pipeline = BoreholePipeline(app_name="MyBoreholeRun")
    pipeline.run(input_path="data/sample/sample_borehole_log.csv",
                 output_path="output/result.parquet",
                 output_format="parquet")
"""

import argparse
import logging
import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from src.nlp.entity_extractor import GeotechnicalEntityExtractor
from src.nlp.preprocessor import TextPreprocessor
from src.utils.schema import BoreholeSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# SPT normalisation helper
# ---------------------------------------------------------------------------

def _parse_spt(value: Optional[str]) -> Optional[int]:
    """
    Convert a raw SPT string to an integer.

    * ``"50+"`` → ``50``
    * ``"28"``  → ``28``
    * ``""`` / ``None`` → ``None``
    """
    if value is None:
        return None
    cleaned = str(value).strip().replace("+", "")
    if cleaned == "":
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


_parse_spt_udf = F.udf(_parse_spt, IntegerType())


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class BoreholePipeline:
    """
    End-to-end PySpark + NLP pipeline for borehole log data extraction.

    Parameters
    ----------
    app_name:
        Spark application name displayed in the Spark UI.
    spark:
        Optional pre-existing :class:`~pyspark.sql.SparkSession`.  When
        omitted a local session is created automatically.
    """

    #: Columns that *must* be present in the raw input
    REQUIRED_COLUMNS = {"borehole_id", "depth_from_m", "depth_to_m"}

    def __init__(
        self,
        app_name: str = "BoreholePipeline",
        spark: Optional[SparkSession] = None,
    ) -> None:
        self.app_name = app_name
        self._spark = spark

    # ------------------------------------------------------------------
    # Spark session management
    # ------------------------------------------------------------------

    @property
    def spark(self) -> SparkSession:
        """Return (or lazily create) the Spark session."""
        if self._spark is None:
            self._spark = (
                SparkSession.builder.appName(self.app_name)
                .master("local[*]")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "4")
                .getOrCreate()
            )
            self._spark.sparkContext.setLogLevel("WARN")
        return self._spark

    def stop(self) -> None:
        """Stop the underlying Spark session."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def ingest(self, input_path: str) -> DataFrame:
        """
        Stage 1 – Load CSV borehole log into a Spark DataFrame.

        Parameters
        ----------
        input_path:
            Path to the input CSV file.

        Returns
        -------
        pyspark.sql.DataFrame
        """
        logger.info("Ingesting data from: %s", input_path)
        df = (
            self.spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "false")  # use explicit schema
            .option("nullValue", "")
            .schema(BoreholeSchema.RAW)
            .load(input_path)
        )
        logger.info("Loaded %d rows.", df.count())
        return df

    def validate(self, df: DataFrame) -> DataFrame:
        """
        Stage 2 – Validate schema and drop rows with missing critical fields.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            Validated DataFrame with invalid rows removed.
        """
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")

        before = df.count()
        df = df.dropna(subset=list(self.REQUIRED_COLUMNS))
        after = df.count()
        if before != after:
            logger.warning("Dropped %d rows with null values in required columns.", before - after)
        return df

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Stage 3 – Text normalisation via :class:`~src.nlp.preprocessor.TextPreprocessor`.
        """
        return TextPreprocessor().transform(df)

    def extract_entities(self, df: DataFrame) -> DataFrame:
        """
        Stage 4 – NLP entity extraction via
        :class:`~src.nlp.entity_extractor.GeotechnicalEntityExtractor`.
        """
        return GeotechnicalEntityExtractor().transform(df)

    def enrich(self, df: DataFrame) -> DataFrame:
        """
        Stage 5 – Compute derived numeric / categorical columns.

        Adds:
        * ``layer_thickness_m`` = ``depth_to_m`` − ``depth_from_m``
        * ``spt_n_value``       = parsed integer from the raw string column
        """
        df = df.withColumn(
            "layer_thickness_m",
            F.round(F.col("depth_to_m") - F.col("depth_from_m"), 2),
        )
        df = df.withColumn("spt_n_value", _parse_spt_udf(F.col("spt_n_value")))
        return df

    def export(self, df: DataFrame, output_path: str, output_format: str = "parquet") -> None:
        """
        Stage 6 – Persist the enriched DataFrame.

        Parameters
        ----------
        df:
            Enriched DataFrame to write.
        output_path:
            Destination path (directory for Parquet/CSV, file for JSON-lines).
        output_format:
            ``"parquet"`` (default), ``"csv"``, or ``"json"``.
        """
        logger.info("Writing output to: %s (format=%s)", output_path, output_format)
        writer = df.coalesce(1).write.mode("overwrite")

        if output_format == "parquet":
            writer.parquet(output_path)
        elif output_format == "csv":
            writer.option("header", "true").csv(output_path)
        elif output_format == "json":
            writer.json(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format!r}")

        logger.info("Export complete.")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "parquet",
    ) -> DataFrame:
        """
        Execute all pipeline stages in order.

        Parameters
        ----------
        input_path:
            Path to the raw CSV borehole log.
        output_path:
            Destination path for the processed output.  When ``None`` the
            result is returned but not persisted.
        output_format:
            Output file format (``"parquet"``, ``"csv"``, or ``"json"``).

        Returns
        -------
        pyspark.sql.DataFrame
            Fully processed and enriched DataFrame.
        """
        df = self.ingest(input_path)
        df = self.validate(df)
        df = self.preprocess(df)
        df = self.extract_entities(df)
        df = self.enrich(df)

        # Reorder columns to match the processed schema
        output_cols = [f.name for f in BoreholeSchema.PROCESSED.fields if f.name in df.columns]
        df = df.select(output_cols)

        if output_path:
            self.export(df, output_path, output_format)

        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Geotechnical Borehole Log NLP Extraction Pipeline",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to the raw borehole log CSV file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Destination path for the processed output (optional).",
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "csv", "json"],
        help="Output file format (default: parquet).",
    )
    parser.add_argument(
        "--app-name",
        default="BoreholePipeline",
        help="Spark application name.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    pipeline = BoreholePipeline(app_name=args.app_name)
    try:
        result = pipeline.run(
            input_path=args.input,
            output_path=args.output,
            output_format=args.format,
        )
        result.show(truncate=False)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
