"""
PySpark schema definitions for geotechnical borehole log data.

The raw schema represents data as it arrives from CSV / scanned-log ingestion.
The processed schema represents the fully enriched output after the NLP pipeline.
"""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


class BoreholeSchema:
    """Central repository of Spark schemas used throughout the pipeline."""

    # ------------------------------------------------------------------
    # Raw input schema  (CSV columns)
    # ------------------------------------------------------------------
    RAW = StructType(
        [
            StructField("borehole_id", StringType(), nullable=False),
            StructField("depth_from_m", DoubleType(), nullable=False),
            StructField("depth_to_m", DoubleType(), nullable=False),
            StructField("sample_type", StringType(), nullable=True),
            StructField("spt_n_value", StringType(), nullable=True),
            StructField("description", StringType(), nullable=True),
            StructField("water_table_m", DoubleType(), nullable=True),
            StructField("remarks", StringType(), nullable=True),
        ]
    )

    # ------------------------------------------------------------------
    # Processed / enriched schema (output of NLP pipeline)
    # ------------------------------------------------------------------
    PROCESSED = StructType(
        [
            StructField("borehole_id", StringType(), nullable=False),
            StructField("depth_from_m", DoubleType(), nullable=False),
            StructField("depth_to_m", DoubleType(), nullable=False),
            StructField("layer_thickness_m", DoubleType(), nullable=True),
            StructField("sample_type", StringType(), nullable=True),
            # SPT stored as integer; 50+ is mapped to 50
            StructField("spt_n_value", IntegerType(), nullable=True),
            StructField("description", StringType(), nullable=True),
            StructField("cleaned_description", StringType(), nullable=True),
            # NLP-extracted entities
            StructField("soil_type", StringType(), nullable=True),
            StructField("consistency_density", StringType(), nullable=True),
            StructField("color", StringType(), nullable=True),
            StructField("moisture", StringType(), nullable=True),
            StructField("plasticity", StringType(), nullable=True),
            # Derived classification
            StructField("material_class", StringType(), nullable=True),
            StructField("uscs_symbol", StringType(), nullable=True),
            StructField("water_table_m", DoubleType(), nullable=True),
            StructField("remarks", StringType(), nullable=True),
        ]
    )

    # ------------------------------------------------------------------
    # Soil-type groupings used for USCS inference
    # ------------------------------------------------------------------
    COARSE_SOILS = {"GRAVEL", "SAND", "COBBLES", "BOULDERS"}
    FINE_SOILS = {"CLAY", "SILT"}
    ORGANIC_SOILS = {"PEAT", "ORGANIC", "TOPSOIL"}
    ROCK_TYPES = {"SANDSTONE", "GRANITE", "LIMESTONE", "MUDSTONE", "SHALE", "BASALT"}

    # ------------------------------------------------------------------
    # Consistency descriptors mapped to approximate cu ranges (kPa)
    # ------------------------------------------------------------------
    CONSISTENCY_MAP = {
        "VERY SOFT": (0, 20),
        "SOFT": (20, 40),
        "FIRM": (40, 75),
        "STIFF": (75, 150),
        "VERY STIFF": (150, 300),
        "HARD": (300, None),
    }

    # ------------------------------------------------------------------
    # Density descriptors mapped to approximate SPT-N ranges
    # ------------------------------------------------------------------
    DENSITY_MAP = {
        "VERY LOOSE": (0, 4),
        "LOOSE": (4, 10),
        "MEDIUM DENSE": (10, 30),
        "DENSE": (30, 50),
        "VERY DENSE": (50, None),
    }
