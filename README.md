# Geotechnical Borehole Log Digitization Pipeline

Automatically extract structured data from scanned or digital borehole report PDFs using OCR, regex-based NLP, and rule-based geotechnical logic.

---

## What the pipeline does

For every PDF placed in the `input_reports/` folder the pipeline:

1. **Opens the PDF** with PyMuPDF (`fitz`).
2. **OCR-scans each page** at multiple resolutions using Tesseract via `pytesseract`:
   - Full page (300 dpi) for body text.
   - Top header strip (350 dpi) for borehole ID and coordinates.
   - Right-side coordinate region (500 dpi) as a precision fallback.
3. **Extracts metadata** per borehole page:
   - Borehole ID (`BH-01`, `BH-02`, …)
   - Easting & Northing (with OCR-error correction)
   - Elevation / RL
4. **Parses SPT rows** (Standard Penetration Test):
   - Depth, N-value, and up to 6 individual blow-count increments.
   - Handles the 15 cm and 7.5 cm penetration-increment conventions.
   - Detects refusal (`R`).
5. **Classifies soil type** from description text using a weighted scoring
   system that prefers uppercase tokens, modifier→head patterns (e.g.
   "silty sand" → SAND), and a geotechnical soil-term dictionary.
6. **Extracts rock-quality indices**: RQD, TCR, SCR.
7. **Detects water depth** from `Water Level`, `WL`, etc.
8. **Exports** results to `output/borehole_data.xlsx` and
   `output/borehole_data.csv`.

---

## Libraries used

| Library | Purpose |
|---|---|
| `PyMuPDF` (`fitz`) | Open PDF, render pages to pixel maps, clip regions |
| `pytesseract` | Python wrapper for Tesseract OCR engine |
| `opencv-python-headless` (`cv2`) | Image pre-processing: colour conversion, Otsu binarisation |
| `numpy` | Pixel array manipulation |
| `pandas` | Tabular data aggregation, Excel / CSV export |
| `openpyxl` | Excel file writing backend for pandas |
| `re`, `os` | Pattern matching and file system access (stdlib) |

---

## Output columns

| Column | Description |
|---|---|
| `Borehole_ID` | e.g. `BH-03` |
| `Easting` / `Northing` | Projected coordinates |
| `Elevation_m` | Ground-level elevation (RL) |
| `Depth_from` / `Depth_to` | SPT test interval (m) |
| `Soil_Type` | Classified dominant soil/rock component |
| `N_value` | SPT N-value (integer or `"R"` for refusal) |
| `Blow_Count_Raw` | Raw blow-count string, e.g. `"5, 8, 9"` |
| `Blow_Count_1` … `Blow_Count_6` | Individual increment values |
| `RQD` / `TCR` / `SCR` | Rock quality indices (%) |
| `Water_Depth` | Groundwater depth (m) |
| `Notes` | OCR context snippet around the SPT row |
| `Page_No` | Source page number |
| `Source_File` | Name of the source PDF |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract** must also be installed at the OS level:
> - Ubuntu/Debian: `sudo apt install tesseract-ocr`
> - macOS: `brew install tesseract`
> - Windows: download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Place PDF reports

```
input_reports/
    BH-01_report.pdf
    BH-02_report.pdf
    ...
```

### 3. Run

```bash
python borehole_pipeline.py
```

Results are written to `output/borehole_data.xlsx` and `output/borehole_data.csv`.

---

## Configuration

All tuneable parameters are at the top of `borehole_pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `INPUT_DIR` | `"input_reports"` | Folder containing source PDFs |
| `OUTPUT_DIR` | `"output"` | Folder for extracted results |
| `EASTING_MIN/MAX` | 300 000 – 330 000 | Valid Easting range for your project |
| `NORTHING_MIN/MAX` | 3 000 000 – 3 100 000 | Valid Northing range for your project |
| `ELEV_MIN/MAX` | −100 – 1 000 m | Valid elevation range |
| `MANUAL_BH_METADATA` | `{}` | Per-borehole override dict for unreliable OCR pages |

---

## Running tests

```bash
pip install pytest
python -m pytest tests/test_pipeline.py -v
```
