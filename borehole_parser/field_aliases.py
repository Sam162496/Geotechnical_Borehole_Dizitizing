"""
field_aliases.py
----------------
Comprehensive geo-vocabulary knowledge base.

Every entry maps one *standard field name* (used in our output schema) to a
list of human-readable aliases that appear in real borehole log reports.
Aliases include abbreviations, variations in spacing/punctuation, multiple
languages and naming conventions observed in Saudi Arabian and international
geotechnical practice.

"Human-like thinking" principle:
  A geotechnical engineer reading a column labelled "PENET BU/15CM" instantly
  recognises it as the SPT penetration resistance (N-value).  We replicate
  that recognition by maintaining an exhaustive alias registry and using
  fuzzy matching as a fallback when no exact alias hit is found.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import re

# ---------------------------------------------------------------------------
# Standard field names (output schema keys)
# ---------------------------------------------------------------------------
STANDARD_FIELDS = [
    "depth_from_m",
    "depth_to_m",
    "depth_m",
    "spt_n_value",
    "blow_counts",
    "soil_description",
    "soil_type",
    "uscs_symbol",
    "sample_id",
    "sample_type",
    "color",
    "moisture",
    "consistency_density",
    "rqd_pct",
    "tcr_pct",
    "scr_pct",
    "fracture_index",
    "water_content_pct",
    "liquid_limit_pct",
    "plastic_limit_pct",
    "plasticity_index",
    "elevation_m",
    "remarks",
    "profile",
    "legend",
    "n_value_raw",          # raw / unreduced blow-count strings like "50/14"
]

# ---------------------------------------------------------------------------
# Alias registry
# ---------------------------------------------------------------------------
# Each value is a list of lowercase normalised strings.
# Normalisation used: strip, lower, collapse whitespace, strip punctuation.
FIELD_ALIASES: Dict[str, List[str]] = {

    # ── Depth from (top of layer) ──────────────────────────────────────────
    "depth_from_m": [
        "depth from", "depth from m", "depth from (m)", "from (m)", "from",
        "from m", "depth top", "top depth", "top of layer",
        "start depth", "depth start", "depth begin",
        "depth (m) from", "depthfrom", "d from",
        "level from", "elevation from",
    ],

    # ── Depth to (base of layer) ───────────────────────────────────────────
    "depth_to_m": [
        "depth to", "depth to m", "depth to (m)", "to (m)", "to",
        "to m", "depth bottom", "bottom depth", "base depth",
        "end depth", "depth end",
        "depth (m) to", "depthto", "d to",
        "level to", "elevation to",
    ],

    # ── Single-column depth ────────────────────────────────────────────────
    "depth_m": [
        "depth", "depth (m)", "depth(m)", "depth m", "d (m)", "d(m)",
        "depth meter", "depth meters", "depth metre", "depth metres",
        "hole depth", "borehole depth", "boring depth",
        "z", "z (m)", "z(m)",
        "sample depth", "test depth",
        "depth (ft)", "depth ft",   # sometimes imperial – pipeline converts
        "depth/m", "depth / m",
    ],

    # ── SPT N-value ────────────────────────────────────────────────────────
    "spt_n_value": [
        # Standard names
        "n value", "n-value", "n_value", "nvalue",
        "spt n", "spt-n", "spt n value", "spt-n value",
        "spt n-value", "spt blow count",
        "standard penetration", "standard penetration test",
        "standard penetration resistance",
        # Abbreviated forms
        "spt", "n", "n (blows)", "n (blows/ft)", "blows/ft", "blows/30cm",
        "blows per foot", "blows per 30 cm",
        # International / regional
        "penetration resistance", "penet resistance",
        "penet bu/15cm",    # appears in Saudi/Middle East reports
        "penet 15 cm", "penet bu 15cm", "penet",
        "blow count", "blow counts total",
        "n30", "n-30", "n30cm",
        "spt value", "spt-value",
        "spt count", "spt-count",
        "resistance", "pen resistance",
        "n (spt)", "(n) spt",
        # Refusal notations
        "50/n",             # e.g. 50/7 means 50 blows in 7 cm
        "r (refusal)",
        # Column header variants
        "spt-n (blows)", "n blows/30cm",
        "n blows", "blows",
    ],

    # ── Raw blow-count string (e.g. "5/4/7", "50/14") ─────────────────────
    "blow_counts": [
        "blow counts", "blows", "blow count detail",
        "intervals", "interval blows", "blow count intervals",
        "spt blows", "blows detail", "blows per interval",
        "individual blows", "partial blows",
    ],

    # ── Soil / rock description ────────────────────────────────────────────
    "soil_description": [
        "description", "soil description", "rock description",
        "material description", "strata description",
        "lithological description", "lithology",
        "geological description", "geology",
        "soil and rock description",
        "stratigraphy", "layer description",
        "desc", "descriptoin",      # typo variant
        "log description", "boring log description",
        "field description",
        "visual description", "visual classification",
        "soil profile description",
        "rock / soil description",
        "remarks description",
        "ground description",
        "material",
    ],

    # ── Classified soil type ───────────────────────────────────────────────
    "soil_type": [
        "soil type", "soil classification", "material type",
        "soil class", "material class",
        "stratum type", "layer type",
        "rock type",
        "fill type", "formation type",
    ],

    # ── USCS symbol ────────────────────────────────────────────────────────
    "uscs_symbol": [
        "uscs", "uscs symbol", "uscs class", "uscs classification",
        "uscs code",
        "unified soil", "unified soil classification",
        "unified", "us classification",
        "aashto", "astm classification",
        "soil symbol", "classification symbol",
        "group symbol",
    ],

    # ── Sample identifier ──────────────────────────────────────────────────
    "sample_id": [
        "sample id", "sample no", "sample number", "sample #",
        "sample", "samples", "sample label",
        "specimen id", "specimen no",
        "b-", "b ",    # prefix patterns handled separately
        "core id", "core number",
    ],

    # ── Sample type ────────────────────────────────────────────────────────
    "sample_type": [
        "sample type", "type of sample", "sample method",
        "sampling method", "sampler",
        "sampler type", "sampling type",
        "ss",   # split spoon
        "ut",   # undisturbed tube
        "rb",   # rock block
        "cs",   # core sample
    ],

    # ── Soil colour ────────────────────────────────────────────────────────
    "color": [
        "color", "colour", "soil colour", "soil color",
        "hue", "munsell color",
    ],

    # ── Moisture state ─────────────────────────────────────────────────────
    "moisture": [
        "moisture", "moisture condition", "moisture state",
        "moisture content condition",
        "water condition", "wet condition",
    ],

    # ── Consistency / density ──────────────────────────────────────────────
    "consistency_density": [
        "consistency", "density", "relative density",
        "consistency density", "consistency/density",
        "firmness", "compactness",
        "relative compactness",
    ],

    # ── RQD ───────────────────────────────────────────────────────────────
    "rqd_pct": [
        "rqd", "rqd %", "rqd(%)", "rqd (%)","r.q.d", "r.q.d.", "r.q.d (%)",
        "rock quality designation", "rock quality",
        "core quality", "rqd percent",
    ],

    # ── TCR (Total Core Recovery) ──────────────────────────────────────────
    "tcr_pct": [
        "tcr", "tcr %", "tcr(%)", "tcr (%)", "t.c.r",
        "total core recovery", "total core rec",
        "core recovery", "total recovery",
        "rec %", "rec.", "recovery %", "recovery(%)",
        "rec (%)", "rec. (%)",
    ],

    # ── SCR (Solid Core Recovery) ──────────────────────────────────────────
    "scr_pct": [
        "scr", "scr %", "scr(%)", "scr (%)", "s.c.r",
        "solid core recovery", "solid core rec",
        "solid recovery",
    ],

    # ── Fracture index ────────────────────────────────────────────────────
    "fracture_index": [
        "fracture index", "fracture_index", "fi", "f.i.",
        "fractures per metre", "fractures/m", "fractures per meter",
        "joint count", "joint frequency",
        "discontinuity frequency",
    ],

    # ── Water content ─────────────────────────────────────────────────────
    "water_content_pct": [
        "water content", "w%", "w (%)", "w(%)", "wc", "wc%",
        "moisture content", "moisture content %",
        "natural water content",
        "w.c.%", "w.c (%)", "water content (%)",
        "w(c)%", "w(c) %", "w (c)%",   # Applus Arabia style
    ],

    # ── Liquid limit ──────────────────────────────────────────────────────
    "liquid_limit_pct": [
        "ll", "ll%", "ll (%)", "ll(%)",
        "liquid limit", "liquid limit %", "liquid limit (%)",
        "atterberg ll", "wl",
    ],

    # ── Plastic limit ─────────────────────────────────────────────────────
    "plastic_limit_pct": [
        "pl", "pl%", "pl (%)", "pl(%)",
        "plastic limit", "plastic limit %", "plastic limit (%)",
        "atterberg pl", "wp",
    ],

    # ── Plasticity index ──────────────────────────────────────────────────
    "plasticity_index": [
        "pi", "pi%", "pi (%)", "pi(%)",
        "plasticity index", "plasticity index %",
        "ip", "index of plasticity",
        "atterberg pi",
    ],

    # ── Elevation / Reduced Level ─────────────────────────────────────────
    "elevation_m": [
        "elevation", "elevation m", "elevation (m)", "elev", "elev.",
        "reduced level", "reduced level m", "reduced level (m)", "rl",
        "rl (m)", "rl(m)", "r.l.", "r.l. (m)",
        "level", "level (m)", "level m",
        "ground level", "ground elevation",
        "e", "e (m)",    # short column label
        "z level",
        "depth level",
    ],

    # ── Remarks ───────────────────────────────────────────────────────────
    "remarks": [
        "remarks", "remark", "note", "notes",
        "comment", "comments", "observation", "observations",
        "additional info", "additional information",
        "field note", "field notes",
        "log note",
    ],

    # ── Visual profile / lithology log ────────────────────────────────────
    "profile": [
        "profile", "log", "visual log", "graphic log",
        "lithological log", "soil profile", "strata log",
        "borehole profile", "core log",
    ],

    # ── Legend / graphic symbol ───────────────────────────────────────────
    "legend": [
        "legend", "graphic", "symbol", "pattern",
        "hatch", "graphic symbol",
    ],

    # ── Raw N-value string (e.g. "50/7", "R") ─────────────────────────────
    "n_value_raw": [
        "n raw", "spt raw", "n/value", "spt string",
        "n notation", "blow notation",
        "50/n notation", "refusal notation",
    ],
}

# ---------------------------------------------------------------------------
# Metadata patterns (document-level, not column-level)
# ---------------------------------------------------------------------------
# These regex patterns are applied to the full page text to extract
# borehole-level metadata.

METADATA_PATTERNS: Dict[str, List[str]] = {
    "borehole_id": [
        # Specific "BH NO." / "HOLE NO." labels (most reliable)
        r"(?:BH NO\.?|HOLE NO\.?|BOREHOLE NO\.?)\s*[:\.]?\s*([A-Z]{1,3}[-\d][A-Z0-9\-]*)",
        # "BH-01", "BH 07", "B-07" patterns
        r"\bBH[-\s]?(\d+[A-Z]?)\b",
        r"\b(BH[-\s]?\d+[A-Z]?)\b",
        # Generic borehole/boring + id number
        r"(?:BOREHOLE|BORING)\s+(?:NO\.?|NUMBER|#)\s*:?\s*([A-Z0-9][-A-Z0-9]*)",
    ],
    "project_name": [
        r"(?:PROJECT|PROJECT NAME|PROJECT NO|PROJECT NUMBER)[:\s]+(.+?)(?:\n|CLIENT|$)",
        r"PROJECT\s*:\s*(.+?)(?:\n|$)",
    ],
    "client": [
        r"(?:CLIENT|CLIENT NAME)[:\s]+(.+?)(?:\n|$)",
        r"CLIENT\s*:\s*(.+?)(?:\n|$)",
    ],
    "location": [
        r"(?:LOCATION|SITE|AREA|SITE LOCATION)[:\s]+(.+?)(?:\n|$)",
    ],
    "date_started": [
        r"(?:DATE STARTED|START DATE|DATE OF DRILLING|DATE FROM|DATE BEGIN)[:\s]+(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        r"(?:DATE STARTED)[:\s]+(\d{2}-\d{2}-\d{2,4})",
    ],
    "date_completed": [
        r"(?:DATE COMPLETED|END DATE|COMPLETION DATE|DATE TO|DATE FINISH)[:\s]+(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
    ],
    "ground_water_depth_m": [
        # "WATER DEPTH: 2.00"
        r"WATER\s+DEPTH[:\s]+(\d+\.?\d*)\s*m?",
        # "GWD 26.22" / "G.W.D 26.22"
        r"(?:GWD|G\.W\.D|G\.W\.L|GWL|GWT)\s*[:\s]+(\d+\.?\d*)\s*m?",
        # "GROUND WATER DEPTH/LEVEL: X"
        r"GROUND\s+WATER\s+(?:DEPTH|LEVEL)[:\s]+(\d+\.?\d*)\s*m?",
        # "G.W.D  26.22" or "G.W.L  5.00"
        r"G\.?W\.?[DL]\.?\s+(\d+\.?\d*)\s*m?",
    ],
    "elevation_m": [
        r"(?:ELEVATION|ELEV\.?)[:\s]+([+-]?\d+\.?\d*)\s*(?:m|M)?",
        r"(?:REDUCED LEVEL|RL|R\.L\.)[:\s]+([+-]?\d+\.?\d*)\s*(?:m|M)?",
        # "GROUND-LEVEL  +637.29  M"
        r"GROUND[-\s]?LEVEL\s+([+-]?\d+\.?\d*)\s*(?:m|M)?",
        r"GROUND[-\s]?ELEVATION\s+([+-]?\d+\.?\d*)\s*(?:m|M)?",
    ],
    "total_depth_m": [
        r"(?:TOTAL DEPTH|BOREHOLE DEPTH|BORING DEPTH|HOLE DEPTH)[:\s(M)]+(\d+\.?\d*)",
        r"(?:DEPTH\s*\(M\))[:\s]+(\d+\.?\d*)",
    ],
    "coordinates_north": [
        r"(?:NORTH|N)[:\s]+(\d+\.?\d+)",
        r"N\s+(\d{7,}\.?\d*)",
    ],
    "coordinates_east": [
        r"(?:EAST|E)[:\s]+(\d+\.?\d+)",
        r"E\s+(\d{6,}\.?\d*)",
    ],
    "drilling_method": [
        r"(?:DRILLING METHOD|METHOD|BORING METHOD)[:\s]+(.+?)(?:\n|$)",
    ],
}

# ---------------------------------------------------------------------------
# Value-pattern validators (used to confirm column mapping by data)
# ---------------------------------------------------------------------------
# After a column is tentatively mapped, its values are checked against these.

VALUE_PATTERNS: Dict[str, str] = {
    # SPT first – more specific patterns including refusal indicators
    "spt_n_value":     r"^\d+$|^R$|^R\.$|^REF$|^REFUSAL$|^50/\d+$|^>?\d+$|\d+/\d+/\d+",
    "rqd_pct":         r"^\d+\.?\d*$",
    "tcr_pct":         r"^\d+\.?\d*$",
    "scr_pct":         r"^\d+\.?\d*$",
    "water_content_pct": r"^\d+\.?\d*$",
    "liquid_limit_pct":  r"^\d+\.?\d*$",
    "plastic_limit_pct": r"^\d+\.?\d*$",
    "plasticity_index":  r"^\d+\.?\d*$",
    "depth_from_m":    r"^\d+\.?\d*$",
    "depth_to_m":      r"^\d+\.?\d*$",
    "depth_m":         r"^\d+\.?\d*$",
    "elevation_m":     r"^[+-]?\d+\.?\d*$",
}

# ---------------------------------------------------------------------------
# Value-range validators (additional numeric sanity checks)
# ---------------------------------------------------------------------------
VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    # SPT N-value: 50 is the conventional refusal limit; 100 allows for
    # exceptional dense material or rock testing without rejecting valid data.
    "spt_n_value":      (0, 100),
    "rqd_pct":          (0, 100),
    "tcr_pct":          (0, 100),
    "scr_pct":          (0, 100),
    "water_content_pct":(0, 200),
    "liquid_limit_pct": (0, 200),
    "plastic_limit_pct":(0, 100),
    "plasticity_index": (0, 150),
    "depth_from_m":     (0, 2000),
    "depth_to_m":       (0, 2000),
    "depth_m":          (0, 2000),
    "elevation_m":      (-1000, 3000),
}


def normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace and remove common punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)   # underscores/hyphens → space
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%./()]", "", text)
    return text.strip()
