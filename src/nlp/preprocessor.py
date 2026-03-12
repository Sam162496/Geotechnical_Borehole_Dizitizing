"""
NLP Text Preprocessor for geotechnical borehole log descriptions.

Implements a PySpark-compatible preprocessing stage that:
  1. Normalises whitespace and punctuation
  2. Upper-cases text for consistent token matching
  3. Expands common geotechnical abbreviations
  4. Removes non-informative stop words while preserving domain terms
  5. Produces a ``cleaned_description`` column ready for entity extraction
"""

import re
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


# ---------------------------------------------------------------------------
# Geotechnical abbreviation expansions
# ---------------------------------------------------------------------------
_ABBREVIATIONS: dict[str, str] = {
    r"\bSP\b": "POORLY GRADED SAND",
    r"\bSW\b": "WELL GRADED SAND",
    r"\bSM\b": "SILTY SAND",
    r"\bSC\b": "CLAYEY SAND",
    r"\bGP\b": "POORLY GRADED GRAVEL",
    r"\bGW\b": "WELL GRADED GRAVEL",
    r"\bGM\b": "SILTY GRAVEL",
    r"\bGC\b": "CLAYEY GRAVEL",
    r"\bCL\b": "LEAN CLAY",
    r"\bCH\b": "FAT CLAY",
    r"\bML\b": "SILT LOW PLASTICITY",
    r"\bMH\b": "ELASTIC SILT",
    r"\bOL\b": "ORGANIC CLAY LOW PLASTICITY",
    r"\bOH\b": "ORGANIC CLAY HIGH PLASTICITY",
    r"\bPT\b": "PEAT",
    r"\bSS\b": "SPLIT SPOON SAMPLE",
    r"\bCS\b": "CHUNK SAMPLE",
    r"\bRC\b": "ROCK CORE",
    r"\bRQD\b": "ROCK QUALITY DESIGNATION",
    r"\bW/T\b": "WATER TABLE",
    r"\bGWT\b": "GROUNDWATER TABLE",
    r"\bSPT\b": "STANDARD PENETRATION TEST",
    r"\bN\b(?=\s*=)": "SPT N VALUE",
    r"\bw/c\b": "MOISTURE CONTENT",
    r"\bLL\b": "LIQUID LIMIT",
    r"\bPL\b": "PLASTIC LIMIT",
    r"\bPI\b": "PLASTICITY INDEX",
}

# ---------------------------------------------------------------------------
# Domain stop words – common English words with no geotechnical meaning
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "A", "AN", "THE", "AND", "OR", "OF", "WITH", "AT", "BY", "FROM",
    "TO", "IN", "ON", "IS", "ARE", "WAS", "WERE", "BE", "BEEN",
    "OCCASIONAL", "SOME", "MINOR", "TRACES",
}


def _expand_abbreviations(text: str) -> str:
    """Expand domain-specific abbreviations in *text* (already upper-cased)."""
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _normalise(text: str) -> str:
    """Clean and normalise a raw description string."""
    if not text:
        return ""
    # Upper-case for uniform matching
    text = text.upper()
    # Remove special characters except hyphens and forward slashes (used in depth ranges)
    text = re.sub(r"[^\w\s\-/%.+]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _remove_stop_words(text: str) -> str:
    """Remove stop words from *text* while preserving word order."""
    tokens = text.split()
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    return " ".join(filtered)


def clean_description(raw: Optional[str]) -> Optional[str]:
    """
    Full preprocessing pipeline applied to a single description string.

    Parameters
    ----------
    raw:
        Raw description text as read from the borehole log.

    Returns
    -------
    str or None
        Cleaned and normalised description, or ``None`` if input is empty.
    """
    if raw is None or str(raw).strip() == "":
        return None
    text = _normalise(str(raw))
    text = _expand_abbreviations(text)
    text = _remove_stop_words(text)
    return text.strip()


# PySpark UDF wrapper so the function can be used inside Spark transformations
_clean_udf = F.udf(clean_description, StringType())


class TextPreprocessor:
    """
    PySpark stage that adds a ``cleaned_description`` column to a DataFrame.

    Usage
    -----
    >>> preprocessor = TextPreprocessor()
    >>> df_clean = preprocessor.transform(df_raw)
    """

    #: Column containing the raw free-text description
    INPUT_COL = "description"
    #: Column that will hold the cleaned / normalised text
    OUTPUT_COL = "cleaned_description"

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply text normalisation to every row and return the enriched DataFrame.

        Parameters
        ----------
        df:
            Input Spark DataFrame that **must** contain a ``description`` column.

        Returns
        -------
        pyspark.sql.DataFrame
            Same DataFrame with an additional ``cleaned_description`` column.
        """
        return df.withColumn(self.OUTPUT_COL, _clean_udf(F.col(self.INPUT_COL)))
