"""
Geotechnical Named-Entity Extractor.

Uses rule-based NLP (regular expressions + vocabulary look-ups) to extract
the following entities from cleaned borehole-log description text:

  * ``soil_type``           – primary material (SAND, CLAY, SILT, GRAVEL, …)
  * ``consistency_density`` – strength / density descriptor
  * ``color``               – Munsell-style colour term
  * ``moisture``            – moisture state
  * ``plasticity``          – plasticity descriptor (fine-grained soils)
  * ``material_class``      – broad class: COARSE, FINE, ORGANIC, or ROCK
  * ``uscs_symbol``         – USCS classification symbol inferred from entities

All extraction functions operate on plain Python strings so they can be used
both as PySpark UDFs and in unit tests without a running Spark session.
"""

import re
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

_SOIL_TYPES = [
    "GRAVEL", "SAND", "SILT", "CLAY", "COBBLES", "BOULDERS", "PEAT",
    "SANDSTONE", "GRANITE", "LIMESTONE", "MUDSTONE", "SHALE", "BASALT",
    "TOPSOIL", "ORGANIC",
]

_CONSISTENCY_FINE = [
    "VERY SOFT", "SOFT", "FIRM", "VERY STIFF", "STIFF", "HARD",
]

_DENSITY_COARSE = [
    "VERY DENSE", "VERY LOOSE", "MEDIUM DENSE", "DENSE", "LOOSE",
]

_ALL_STRENGTH_DENSITY = _CONSISTENCY_FINE + _DENSITY_COARSE

_COLORS = [
    "BLACK", "DARK BROWN", "BROWN", "LIGHT BROWN", "TAN", "REDDISH BROWN",
    "RED", "ORANGE", "YELLOW", "OLIVE", "GREEN", "DARK GREY", "GREY",
    "LIGHT GREY", "WHITE",
]

_MOISTURE = [
    "DRY", "SLIGHTLY MOIST", "MOIST", "WET", "SATURATED", "VERY MOIST",
]

_PLASTICITY = [
    "LOW PLASTICITY", "MEDIUM PLASTICITY", "HIGH PLASTICITY",
    "NON-PLASTIC", "LEAN", "FAT", "ELASTIC",
]

# USCS symbol lookup: (soil_type, descriptor) → symbol
_USCS_LOOKUP: dict[tuple[str, str], str] = {
    ("GRAVEL", "WELL GRADED"): "GW",
    ("GRAVEL", "POORLY GRADED"): "GP",
    ("GRAVEL", "SILTY"): "GM",
    ("GRAVEL", "CLAYEY"): "GC",
    ("SAND", "WELL GRADED"): "SW",
    ("SAND", "POORLY GRADED"): "SP",
    ("SAND", "SILTY"): "SM",
    ("SAND", "CLAYEY"): "SC",
    ("CLAY", "LOW PLASTICITY"): "CL",
    ("CLAY", "HIGH PLASTICITY"): "CH",
    ("CLAY", "LEAN"): "CL",
    ("CLAY", "FAT"): "CH",
    ("SILT", "LOW PLASTICITY"): "ML",
    ("SILT", "ELASTIC"): "MH",
    ("PEAT", ""): "PT",
    ("ORGANIC", "LOW PLASTICITY"): "OL",
    ("ORGANIC", "HIGH PLASTICITY"): "OH",
}


# ---------------------------------------------------------------------------
# Helper: longest-match extraction from a sorted vocabulary list
# ---------------------------------------------------------------------------

def _extract_first(text: str, vocabulary: list[str]) -> Optional[str]:
    """
    Return the vocabulary term whose first occurrence appears earliest in *text*.

    Multi-word terms (e.g. "VERY STIFF") are checked before single-word terms
    of the same length to ensure the most specific match is preferred.  When
    two terms start at the same position the longer term wins.
    """
    if not text:
        return None
    best_term: Optional[str] = None
    best_start: int = len(text) + 1
    best_len: int = 0
    for term in vocabulary:
        m = re.search(r"\b" + re.escape(term) + r"\b", text)
        if m:
            start = m.start()
            # Prefer earlier position; break ties by longer match
            if start < best_start or (start == best_start and len(term) > best_len):
                best_term = term
                best_start = start
                best_len = len(term)
    return best_term


# ---------------------------------------------------------------------------
# Individual entity extraction functions
# ---------------------------------------------------------------------------

def extract_soil_type(text: Optional[str]) -> Optional[str]:
    """Extract primary soil / rock type from *text*."""
    return _extract_first(text or "", _SOIL_TYPES)


def extract_consistency_density(text: Optional[str]) -> Optional[str]:
    """Extract consistency (fine soils) or density (coarse soils) descriptor."""
    return _extract_first(text or "", _ALL_STRENGTH_DENSITY)


def extract_color(text: Optional[str]) -> Optional[str]:
    """Extract colour descriptor from *text*."""
    return _extract_first(text or "", _COLORS)


def extract_moisture(text: Optional[str]) -> Optional[str]:
    """Extract moisture state from *text*."""
    return _extract_first(text or "", _MOISTURE)


def extract_plasticity(text: Optional[str]) -> Optional[str]:
    """Extract plasticity descriptor from *text* (relevant for fine soils)."""
    return _extract_first(text or "", _PLASTICITY)


def classify_material(soil_type: Optional[str]) -> Optional[str]:
    """
    Assign a broad material class based on the extracted soil type.

    Returns one of: ``COARSE``, ``FINE``, ``ORGANIC``, ``ROCK``, or ``None``.
    """
    if not soil_type:
        return None
    st = soil_type.upper()
    if st in {"GRAVEL", "SAND", "COBBLES", "BOULDERS"}:
        return "COARSE"
    if st in {"CLAY", "SILT"}:
        return "FINE"
    if st in {"PEAT", "ORGANIC", "TOPSOIL"}:
        return "ORGANIC"
    if st in {"SANDSTONE", "GRANITE", "LIMESTONE", "MUDSTONE", "SHALE", "BASALT"}:
        return "ROCK"
    return None


def infer_uscs(
    soil_type: Optional[str],
    consistency_density: Optional[str],
    plasticity: Optional[str],
) -> Optional[str]:
    """
    Infer a USCS (Unified Soil Classification System) symbol.

    Looks up *(soil_type, modifier)* in a pre-built table, trying both the
    consistency/density and plasticity modifiers.
    """
    if not soil_type:
        return None
    st = (soil_type or "").upper()
    cd = (consistency_density or "").upper()
    pl = (plasticity or "").upper()

    # Special cases
    if st == "PEAT":
        return "PT"

    # Try plasticity first (more specific for fine soils)
    for modifier in [pl, cd]:
        key = (st, modifier)
        if key in _USCS_LOOKUP:
            return _USCS_LOOKUP[key]

    return None


# ---------------------------------------------------------------------------
# PySpark UDFs
# ---------------------------------------------------------------------------

_soil_type_udf = F.udf(extract_soil_type, StringType())
_consistency_udf = F.udf(extract_consistency_density, StringType())
_color_udf = F.udf(extract_color, StringType())
_moisture_udf = F.udf(extract_moisture, StringType())
_plasticity_udf = F.udf(extract_plasticity, StringType())
_material_class_udf = F.udf(classify_material, StringType())


@F.udf(StringType())
def _uscs_udf(soil_type: Optional[str], consistency_density: Optional[str], plasticity: Optional[str]) -> Optional[str]:
    return infer_uscs(soil_type, consistency_density, plasticity)


# ---------------------------------------------------------------------------
# Main transformer class
# ---------------------------------------------------------------------------

class GeotechnicalEntityExtractor:
    """
    PySpark stage that extracts NLP entities from ``cleaned_description``.

    Adds the following columns to the input DataFrame:

    * ``soil_type``
    * ``consistency_density``
    * ``color``
    * ``moisture``
    * ``plasticity``
    * ``material_class``
    * ``uscs_symbol``

    Usage
    -----
    >>> extractor = GeotechnicalEntityExtractor()
    >>> df_enriched = extractor.transform(df_preprocessed)
    """

    INPUT_COL = "cleaned_description"

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply all entity-extraction UDFs and return the enriched DataFrame.

        Parameters
        ----------
        df:
            Spark DataFrame with a ``cleaned_description`` column (produced by
            :class:`~src.nlp.preprocessor.TextPreprocessor`).

        Returns
        -------
        pyspark.sql.DataFrame
            Input DataFrame extended with entity columns.
        """
        col = F.col(self.INPUT_COL)

        df = df.withColumn("soil_type", _soil_type_udf(col))
        df = df.withColumn("consistency_density", _consistency_udf(col))
        df = df.withColumn("color", _color_udf(col))
        df = df.withColumn("moisture", _moisture_udf(col))
        df = df.withColumn("plasticity", _plasticity_udf(col))
        df = df.withColumn("material_class", _material_class_udf(F.col("soil_type")))
        df = df.withColumn(
            "uscs_symbol",
            _uscs_udf(
                F.col("soil_type"),
                F.col("consistency_density"),
                F.col("plasticity"),
            ),
        )
        return df
