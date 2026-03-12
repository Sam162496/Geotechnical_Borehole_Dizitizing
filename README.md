# Geotechnical Borehole Log Parser

A **format-agnostic**, human-like pipeline for extracting structured data from
any geotechnical borehole log report — PDFs, scanned images, CSV, Excel, or
plain text — regardless of column naming conventions, report layout, or
regional standards.

---

## The Problem

Real-world geotechnical borehole log reports come in dozens of formats.
The same field (e.g. SPT N-value) may be labelled:

| Report Style | Column Label |
|---|---|
| Riyadh Geotechnique & Foundations | `PENET BU/15CM` / `SPT-N` |
| Applus Arabia | `N-Value` |
| International standard | `N (blows/ft)` / `Blow Count` |
| Some reports | `Standard Penetration` |

A human geotechnical engineer reads any of these and instantly knows what
they mean.  This pipeline replicates that understanding.

---

## How It Works — "Human-Like Thinking"

The pipeline uses a **cascading intelligence strategy** for every column:

```
1. Exact match       → "N-Value"     exactly matches alias → spt_n_value   ✓
2. Contains match    → "PENET BU/15CM" contains "penet"   → spt_n_value   ✓
3. Fuzzy match       → "DESCRIPTOIN"  ~= "description"    → soil_description ✓
4. Value-pattern     → column has ["22","R","50/7"] → spt_n_value (inferred) ✓
5. Unresolved        → keeps original name, data preserved
```

The **geo-vocabulary knowledge base** (`field_aliases.py`) contains hundreds
of aliases per standard field, covering:
- Official column names and common abbreviations
- Regional/international naming variants (Saudi, European, American)
- Typos and word-order variations

---

## Installation

```bash
pip install -r requirements.txt
```

### Optional: OCR for scanned PDFs

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Python binding
pip install pytesseract
```

---

## Quick Start

### Python API

```python
from borehole_parser import BoreholePipeline

pipeline = BoreholePipeline(verbose=True)
result = pipeline.run(
    "path/to/borehole_report.pdf",   # or .csv, .xlsx, .png, .txt
    output_dir="output/",
    fmt="csv",                        # csv | excel | json | parquet
)

# Standardised DataFrame — same column names regardless of input format
print(result.df)

# Borehole-level metadata (ID, date, location, GWL, …)
print(result.metadata)

# Human-readable summary
print(result.summary())
```

### Command-Line Interface

```bash
python -m borehole_parser.pipeline \
    --input  report.pdf \
    --output output/ \
    --format csv \
    --verbose
```

---

## Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Text-based PDF | `.pdf` | Full table extraction via pdfplumber |
| Scanned PDF | `.pdf` | OCR via Tesseract (if installed) |
| Image | `.png .jpg .tiff` | OCR via Tesseract (if installed) |
| CSV | `.csv` | Direct column mapping |
| Excel | `.xlsx .xls` | All sheets processed |
| Plain text | `.txt` | Heuristic table detection |

---

## Standard Output Schema

All output DataFrames use these standardised column names:

| Field | Description |
|-------|-------------|
| `depth_from_m` | Top of layer (m) |
| `depth_to_m` | Bottom of layer (m) |
| `depth_m` | Single depth measurement (m) |
| `spt_n_value` | SPT N-value (integer, refusal → 50) |
| `blow_counts` | Raw interval blow counts (e.g. "5/4/7") |
| `soil_description` | Original free-text description |
| `soil_description_clean` | Cleaned + abbreviations expanded |
| `soil_type` | Classified soil type |
| `uscs_symbol` | USCS classification (CL, SW, GM, …) |
| `sample_id` | Sample identifier |
| `sample_type` | Sample type (SS, UT, RB, …) |
| `color` | Soil colour |
| `moisture` | Moisture state |
| `consistency_density` | Consistency / relative density descriptor |
| `rqd_pct` | Rock Quality Designation (%) |
| `tcr_pct` | Total Core Recovery (%) |
| `scr_pct` | Solid Core Recovery (%) |
| `fracture_index` | Fracture index |
| `water_content_pct` | Water content w (%) |
| `liquid_limit_pct` | Liquid Limit LL (%) |
| `plastic_limit_pct` | Plastic Limit PL (%) |
| `plasticity_index` | Plasticity Index PI |
| `elevation_m` | Elevation / Reduced Level (m) |
| `layer_thickness_m` | Computed: depth_to_m − depth_from_m |
| `remarks` | Remarks / notes |

---

## Architecture

```
borehole_parser/
├── pipeline.py               ← Main entry point (BoreholePipeline)
├── field_aliases.py          ← Geo-vocabulary knowledge base
├── column_mapper.py          ← 4-strategy column mapping (the "AI" layer)
├── metadata_detector.py      ← Document-level metadata extraction
├── ingestion/
│   ├── pdf_reader.py         ← PDF text + table extraction (pdfplumber/PyMuPDF)
│   └── ocr_engine.py         ← OCR wrapper (Tesseract, graceful fallback)
├── extraction/
│   ├── table_detector.py     ← Table detection (structured + text heuristic)
│   └── data_extractor.py     ← DataFrame building, SPT parsing, cleaning
└── output/
    └── formatter.py          ← CSV / Excel / JSON / Parquet writer
```

---

## Running Tests

```bash
pytest tests/ -v
```

92 tests cover:
- Column mapping (exact, contains, fuzzy, value-based strategies)
- SPT string parsing (`"50/7"` → 50, `"5/4/7"` → 16, `"R"` → 50)
- Table detection from structured and free-text sources
- Metadata extraction from two contrasting report formats
- Full pipeline integration for both Format 1 and Format 2 reports
- All three output formats (CSV, Excel, JSON)
