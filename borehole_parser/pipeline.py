"""
pipeline.py
-----------
BoreholePipeline – the single entry-point for the generic borehole log parser.

Usage (Python API):
    from borehole_parser import BoreholePipeline

    pipeline = BoreholePipeline()
    result = pipeline.run("path/to/report.pdf", output_dir="output/", fmt="csv")
    print(result.df)
    print(result.metadata)

Usage (CLI):
    python -m borehole_parser.pipeline --input report.pdf --output output/ --format csv

Supported input formats:
  • PDF  (.pdf)  – text-based (pdfplumber) and scanned (OCR via Tesseract)
  • Image (.png, .jpg, .jpeg, .tiff, .tif, .bmp)  – OCR
  • CSV  (.csv)
  • Excel (.xlsx, .xls)
  • Text  (.txt)  – raw text, heuristic table parsing

Pipeline stages:
  1. Ingest      – load the file into page text + tables
  2. OCR         – run OCR on scanned pages (if Tesseract available)
  3. Metadata    – extract borehole-level metadata from page headers
  4. Detect      – detect and parse table structures
  5. Map         – map arbitrary column headers to standard field names
  6. Clean       – clean, type-cast, post-process values
  7. Export      – write output file

"Human-like thinking" is implemented in the ColumnMapper (step 5):
  It uses a comprehensive geo-vocabulary knowledge base, fuzzy string
  matching, and data-value pattern checks to correctly identify columns
  regardless of their naming convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from borehole_parser.column_mapper import map_columns
from borehole_parser.extraction.data_extractor import merge_tables, table_to_dataframe
from borehole_parser.extraction.table_detector import extract_tables
from borehole_parser.ingestion.ocr_engine import ocr_image, ocr_available
from borehole_parser.ingestion.pdf_reader import PageData, read_pdf
from borehole_parser.metadata_detector import extract_metadata, format_metadata_summary
from borehole_parser.output.formatter import print_summary, save


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Holds all artefacts produced by BoreholePipeline.run()."""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict = field(default_factory=dict)
    column_mappings: Dict = field(default_factory=dict)
    output_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.df.empty

    def summary(self) -> str:
        lines = [format_metadata_summary(self.metadata)]
        lines.append(f"\nExtracted {len(self.df)} data rows with {len(self.df.columns)} columns.")
        if self.output_path:
            lines.append(f"Output saved to: {self.output_path}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BoreholePipeline:
    """
    Format-agnostic geotechnical borehole log extraction pipeline.

    Parameters
    ----------
    verbose : bool
        Print progress and column-mapping details to stdout.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        fmt: str = "csv",
    ) -> ExtractionResult:
        """
        End-to-end extraction.

        Parameters
        ----------
        input_path : str
        output_dir : str, optional
            Directory for output file.  Defaults to same directory as input.
        fmt : str
            Output format: "csv" | "excel" | "json" | "parquet".

        Returns
        -------
        ExtractionResult
        """
        result = ExtractionResult()
        input_path = str(input_path)

        self._log(f"\n{'='*60}")
        self._log(f"BoreholePipeline: processing  {input_path}")
        self._log(f"{'='*60}")

        # ── 1. Ingest ────────────────────────────────────────────────────
        pages = self._ingest(input_path, result)
        if not pages:
            result.warnings.append("No content could be extracted from the file.")
            return result

        # ── 2. OCR (scanned pages) ───────────────────────────────────────
        pages = self._apply_ocr(pages, result)

        # ── 3. Metadata ──────────────────────────────────────────────────
        all_text = "\n".join(p.raw_text for p in pages if p.raw_text)
        result.metadata = extract_metadata(all_text)

        # ── 4. Detect tables ─────────────────────────────────────────────
        col_dicts = []
        for pg in pages:
            extracted = extract_tables(pg.raw_text, pg.tables)
            col_dicts.extend(extracted)

        if not col_dicts:
            result.warnings.append(
                "No tables detected. Ensure the file contains a borehole log table."
            )
            return result

        # ── 5 + 6. Map columns & clean data ─────────────────────────────
        dfs: List[pd.DataFrame] = []
        for cd in col_dicts:
            df_i = table_to_dataframe(cd, verbose=self.verbose)
            if not df_i.empty:
                dfs.append(df_i)

        if not dfs:
            result.warnings.append("Tables detected but no usable data rows extracted.")
            return result

        result.df = merge_tables(dfs)

        # Store column mapping details (informational)
        for col in result.df.columns:
            result.column_mappings[col] = col   # already renamed

        # ── 7. Export ────────────────────────────────────────────────────
        if output_dir is not None:
            stem = Path(input_path).stem
            out_path = os.path.join(output_dir, stem)
            result.output_path = save(result.df, result.metadata, out_path, fmt=fmt)
            self._log(f"Output saved to: {result.output_path}")

        if self.verbose:
            print_summary(result.df, result.metadata)

        return result

    # ── Stage helpers ───────────────────────────────────────────────────────

    def _ingest(self, path: str, result: ExtractionResult) -> List[PageData]:
        suffix = Path(path).suffix.lower()

        if suffix == ".pdf":
            return self._ingest_pdf(path, result)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
            return self._ingest_image(path, result)
        elif suffix == ".csv":
            return self._ingest_csv(path, result)
        elif suffix in (".xlsx", ".xls"):
            return self._ingest_excel(path, result)
        elif suffix == ".txt":
            return self._ingest_text(path, result)
        else:
            result.warnings.append(f"Unknown file extension {suffix!r}; trying PDF reader.")
            try:
                return self._ingest_pdf(path, result)
            except Exception:
                result.warnings.append("PDF reader failed for unknown file type.")
                return []

    def _ingest_pdf(self, path: str, result: ExtractionResult) -> List[PageData]:
        self._log("Stage 1: Ingesting PDF …")
        try:
            pages = read_pdf(path)
            self._log(f"  → {len(pages)} page(s) read.")
            return pages
        except Exception as exc:
            result.warnings.append(f"PDF read error: {exc}")
            return []

    def _ingest_image(self, path: str, result: ExtractionResult) -> List[PageData]:
        self._log("Stage 1: Ingesting image file …")
        pg = PageData(page_num=1, image_path=path)
        return [pg]

    def _ingest_csv(self, path: str, result: ExtractionResult) -> List[PageData]:
        self._log("Stage 1: Ingesting CSV …")
        try:
            df = pd.read_csv(path, dtype=str)
            return [self._df_to_page(df)]
        except Exception as exc:
            result.warnings.append(f"CSV read error: {exc}")
            return []

    def _ingest_excel(self, path: str, result: ExtractionResult) -> List[PageData]:
        self._log("Stage 1: Ingesting Excel …")
        try:
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
            pages = []
            for sheet_name, df in sheets.items():
                self._log(f"  → Sheet: {sheet_name}")
                pages.append(self._df_to_page(df))
            return pages
        except Exception as exc:
            result.warnings.append(f"Excel read error: {exc}")
            return []

    def _ingest_text(self, path: str, result: ExtractionResult) -> List[PageData]:
        self._log("Stage 1: Ingesting text file …")
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            return [PageData(page_num=1, raw_text=text)]
        except Exception as exc:
            result.warnings.append(f"Text read error: {exc}")
            return []

    @staticmethod
    def _df_to_page(df: pd.DataFrame) -> PageData:
        """Convert a DataFrame to a PageData object with a synthetic table."""
        rows = [df.columns.tolist()]
        for _, row in df.iterrows():
            rows.append([str(v) if pd.notna(v) else "" for v in row])
        return PageData(page_num=1, tables=[rows])

    def _apply_ocr(self, pages: List[PageData], result: ExtractionResult) -> List[PageData]:
        """Run OCR on pages without text (if Tesseract is available)."""
        scanned = [p for p in pages if not p.has_text and p.image_path]
        if not scanned:
            return pages

        if not ocr_available():
            result.warnings.append(
                f"{len(scanned)} page(s) appear to be scanned images but Tesseract OCR "
                "is not installed.  Install it with:  "
                "sudo apt-get install tesseract-ocr && pip install pytesseract"
            )
            return pages

        self._log(f"Stage 2: Running OCR on {len(scanned)} scanned page(s) …")
        for pg in scanned:
            text = ocr_image(pg.image_path)
            pg.raw_text = text
            self._log(f"  → Page {pg.page_num}: {len(text)} chars extracted via OCR.")

        return pages

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generic Geotechnical Borehole Log Parser\n"
            "Extracts structured data from any borehole log report format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Path to input file (PDF, image, CSV, Excel, TXT).")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory.  Defaults to same directory as input.")
    p.add_argument("--format", "-f", default="csv",
                   choices=["csv", "excel", "json", "parquet"],
                   help="Output file format (default: csv).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print detailed progress and column mapping info.")
    return p


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pipeline = BoreholePipeline(verbose=args.verbose or True)
    result = pipeline.run(
        input_path=args.input,
        output_dir=args.output or str(Path(args.input).parent),
        fmt=args.format,
    )

    print(result.summary())

    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
