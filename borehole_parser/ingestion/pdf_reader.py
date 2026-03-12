"""
pdf_reader.py
-------------
Ingests PDF documents and converts them into a list of page objects, each
containing:

  - raw_text   : str  – all text extracted from the page
  - tables     : list[list[list[str]]] – tables detected by pdfplumber
  - page_num   : int
  - image_path : str | None – path to a rendered PNG of the page (used for OCR)

Two backends are tried in order of preference:
  1. pdfplumber  – the best option for digital / text-based PDFs; it can
                   extract both text AND table structures natively.
  2. PyMuPDF    – used as fallback for page rendering (needed for OCR on
                   scanned documents).

If pdfplumber cannot extract any meaningful text from a page (e.g. it is a
scanned image), the page is rendered to a temporary PNG so that the OCR
engine can process it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Optional heavy deps – import lazily so the package loads without them.
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


@dataclass
class PageData:
    page_num: int
    raw_text: str = ""
    tables: List[List[List[Optional[str]]]] = field(default_factory=list)
    image_path: Optional[str] = None   # path to rendered PNG for OCR

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text and self.raw_text.strip())

    @property
    def has_tables(self) -> bool:
        return bool(self.tables)


def read_pdf(path: str, render_dpi: int = 200) -> List[PageData]:
    """
    Read all pages of a PDF and return a list of PageData objects.

    Parameters
    ----------
    path : str
        Absolute path to the PDF file.
    render_dpi : int
        Resolution for rendering scanned pages to image for OCR.

    Returns
    -------
    list of PageData
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: List[PageData] = []

    if _HAS_PDFPLUMBER:
        pages = _read_with_pdfplumber(path)
    elif _HAS_FITZ:
        pages = _read_with_fitz_text(path)
    else:
        raise ImportError(
            "Neither pdfplumber nor PyMuPDF (fitz) is installed.  "
            "Install at least one with:  pip install pdfplumber  OR  pip install PyMuPDF"
        )

    # For pages with no text (scanned), render to image for OCR fallback
    if _HAS_FITZ:
        _render_scanned_pages(path, pages, dpi=render_dpi)

    return pages


def _read_with_pdfplumber(path: str) -> List[PageData]:
    """Extract text and tables using pdfplumber."""
    result: List[PageData] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            raw_tables = page.extract_tables() or []
            # Normalise: each cell is str or None
            tables = []
            for tbl in raw_tables:
                norm_tbl = [
                    [str(cell).strip() if cell is not None else "" for cell in row]
                    for row in tbl
                ]
                tables.append(norm_tbl)
            result.append(PageData(page_num=i + 1, raw_text=text, tables=tables))
    return result


def _read_with_fitz_text(path: str) -> List[PageData]:
    """Extract text using PyMuPDF (no table structure)."""
    result: List[PageData] = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        result.append(PageData(page_num=i + 1, raw_text=text))
    doc.close()
    return result


def _render_scanned_pages(path: str, pages: List[PageData], dpi: int) -> None:
    """
    For pages with no extracted text, render them to PNG images so the OCR
    engine can process them.  Images are written to a temp directory.
    """
    if not _HAS_FITZ:
        return

    doc = fitz.open(path)
    tmp_dir = tempfile.mkdtemp(prefix="borehole_ocr_")

    for pd_page in pages:
        if not pd_page.has_text:
            fitz_page = doc[pd_page.page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = fitz_page.get_pixmap(matrix=mat)
            img_path = os.path.join(tmp_dir, f"page_{pd_page.page_num:03d}.png")
            pix.save(img_path)
            pd_page.image_path = img_path

    doc.close()
