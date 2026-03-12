# Geotechnical Borehole Digitizing – AI Extraction Pipeline

An end-to-end **PySpark + NLP** pipeline that automatically extracts structured
geotechnical data from raw borehole log descriptions.

## Background

Borehole logs contain rich free-text descriptions of soil and rock layers (e.g.
*"Stiff grey sandy CLAY, moist"*) alongside tabular measurements such as SPT
N-values and depth intervals.  This tool digitises those descriptions into a
machine-readable format suitable for geotechnical analysis, reporting, and
machine-learning model training.

---

## Architecture

```
Raw CSV borehole log
        │
        ▼
┌──────────────────┐
│  1. Ingest       │  Load CSV → Spark DataFrame  (BoreholeSchema.RAW)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Validate     │  Check required columns; drop rows with critical nulls
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  3. Preprocess  (TextPreprocessor)        │
│     • Upper-case normalisation            │
│     • Abbreviation expansion             │
│       (e.g. SS → SPLIT SPOON SAMPLE)     │
│     • Stop-word removal                  │
│     ⟶ cleaned_description column         │
└────────┬─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  4. NLP Entity Extraction                            │
│     (GeotechnicalEntityExtractor)                    │
│     • soil_type           – CLAY / SAND / GRAVEL … │
│     • consistency_density – STIFF / DENSE …         │
│     • color               – BROWN / GREY …          │
│     • moisture            – MOIST / WET …           │
│     • plasticity          – LOW / HIGH PLASTICITY … │
│     • material_class      – COARSE / FINE / ROCK    │
│     • uscs_symbol         – CL / CH / SW / GP …     │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  5. Enrich                            │
│     • layer_thickness_m (computed)   │
│     • spt_n_value → integer (50+→50) │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  6. Export       │  Parquet / CSV / JSON
└──────────────────┘
```

---

## Project Structure

```
.
├── data/
│   └── sample/
│       └── sample_borehole_log.csv   ← sample input data
├── src/
│   ├── nlp/
│   │   ├── preprocessor.py           ← TextPreprocessor (stage 3)
│   │   └── entity_extractor.py       ← GeotechnicalEntityExtractor (stage 4)
│   ├── pipeline/
│   │   └── borehole_pipeline.py      ← BoreholePipeline orchestrator
│   └── utils/
│       └── schema.py                 ← Spark schemas & domain vocabulary
├── tests/
│   ├── test_nlp.py                   ← unit tests (no Spark required)
│   └── test_pipeline.py              ← integration tests (local Spark)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Run the pipeline (CLI)

```bash
python -m src.pipeline.borehole_pipeline \
    --input  data/sample/sample_borehole_log.csv \
    --output output/processed_borehole_log \
    --format parquet
```

Supported `--format` values: `parquet` (default), `csv`, `json`.

### 3 – Use as a library

```python
from src.pipeline.borehole_pipeline import BoreholePipeline

pipeline = BoreholePipeline(app_name="MySite_BH01")
df = pipeline.run(
    input_path="data/sample/sample_borehole_log.csv",
    output_path="output/result",
    output_format="csv",
)
df.show(truncate=False)
pipeline.stop()
```

---

## Input Format

The pipeline expects a CSV file with the following columns:

| Column          | Type   | Required | Description                           |
|-----------------|--------|----------|---------------------------------------|
| `borehole_id`   | string | ✅       | Unique borehole identifier            |
| `depth_from_m`  | double | ✅       | Top of layer (metres)                 |
| `depth_to_m`    | double | ✅       | Bottom of layer (metres)              |
| `sample_type`   | string |          | SS / CS / RC / …                     |
| `spt_n_value`   | string |          | SPT blow count (e.g. `28`, `50+`)     |
| `description`   | string |          | Free-text soil/rock description       |
| `water_table_m` | double |          | Depth to water table (metres)         |
| `remarks`       | string |          | Additional field notes                |

See `data/sample/sample_borehole_log.csv` for a worked example.

---

## Output Columns

The enriched DataFrame adds the following columns on top of the input:

| Column                | Description                                          |
|-----------------------|------------------------------------------------------|
| `cleaned_description` | Normalised description (upper-case, abbreviations expanded) |
| `soil_type`           | Primary material (e.g. `CLAY`, `SAND`, `GRAVEL`)    |
| `consistency_density` | Strength / density descriptor (e.g. `STIFF`, `DENSE`) |
| `color`               | Colour term (e.g. `BROWN`, `GREY`)                  |
| `moisture`            | Moisture state (e.g. `MOIST`, `SATURATED`)          |
| `plasticity`          | Plasticity descriptor for fine soils                |
| `material_class`      | Broad class: `COARSE`, `FINE`, `ORGANIC`, or `ROCK` |
| `uscs_symbol`         | USCS symbol inferred from NLP entities               |
| `layer_thickness_m`   | `depth_to_m − depth_from_m`                         |
| `spt_n_value`         | SPT N-value as integer (`50+` → `50`)               |

---

## Running Tests

```bash
# Fast unit tests (no Spark required – run pure-Python NLP functions)
pytest tests/test_nlp.py -v

# Full integration tests (starts a local Spark session)
pytest tests/test_pipeline.py -v

# All tests with coverage
pytest --cov=src tests/
```

---

## Domain Knowledge – NLP Rules

The NLP extractor uses a vocabulary-driven, rule-based approach modelled on
the **Unified Soil Classification System (USCS / ASTM D2487)**:

* **Soil types**: GRAVEL, SAND, SILT, CLAY, PEAT, COBBLES, BOULDERS, plus
  common rock types (SANDSTONE, GRANITE, …)
* **Consistency (fine soils)**: VERY SOFT → SOFT → FIRM → STIFF → VERY STIFF → HARD
* **Density (coarse soils)**: VERY LOOSE → LOOSE → MEDIUM DENSE → DENSE → VERY DENSE
* **USCS symbols**: CL, CH, ML, MH, SW, SP, SM, SC, GW, GP, GM, GC, PT, OL, OH
