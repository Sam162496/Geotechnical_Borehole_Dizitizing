# Geotechnical Borehole Digitizing

Extract structured geotechnical data from borehole log reports using a PySpark + NLP pipeline.

## Quick Start

### 1. Open the VS Code Workspace

```
File > Open Workspace from File… > Geotechnical_Borehole_Digitizing.code-workspace
```

This workspace configures Python settings, recommended extensions, launch configurations, and tasks for the project.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the built-in VS Code task: **Terminal > Run Task… > Install dependencies**

### 3. Run the Pipeline

```bash
python -m src.pipeline.borehole_pipeline \
    --input  data/sample/sample_borehole_log.csv \
    --output output/processed \
    --format csv
```

Or use the **Run Pipeline (CSV output)** launch configuration in VS Code (F5).

### 4. Run Tests

```bash
pytest tests/ -v
```

Or use the **Run Tests** launch configuration in VS Code.

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| Ingest | Load raw CSV borehole log into a Spark DataFrame |
| Validate | Check required columns; drop rows with critical nulls |
| Preprocess | Normalise text, expand abbreviations, remove stop words |
| Extract | Rule-based NLP entity extraction (soil type, USCS, colour, …) |
| Enrich | Compute layer thickness; parse SPT N-values to integers |
| Export | Write enriched DataFrame to Parquet / CSV / JSON |

## Output Columns

`borehole_id`, `depth_from_m`, `depth_to_m`, `layer_thickness_m`, `sample_type`,
`spt_n_value`, `description`, `cleaned_description`, `soil_type`,
`consistency_density`, `color`, `moisture`, `plasticity`, `material_class`,
`uscs_symbol`, `water_table_m`, `remarks`

## Project Structure

```
.
├── Geotechnical_Borehole_Digitizing.code-workspace  ← VS Code workspace
├── requirements.txt
├── data/
│   └── sample/
│       └── sample_borehole_log.csv
├── src/
│   ├── nlp/
│   │   ├── preprocessor.py       ← text normalisation
│   │   └── entity_extractor.py   ← rule-based NLP extractor
│   ├── pipeline/
│   │   └── borehole_pipeline.py  ← end-to-end pipeline + CLI
│   └── utils/
│       └── schema.py             ← Spark schema definitions
└── tests/
```
