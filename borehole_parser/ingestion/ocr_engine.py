"""
ocr_engine.py
-------------
OCR engine wrapper.  Uses Tesseract via pytesseract when available;
falls back to a no-op (returns empty string) when Tesseract is not installed.

Tesseract setup:
  Ubuntu / Debian:  sudo apt-get install tesseract-ocr
  Python binding:   pip install pytesseract Pillow
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    _HAS_TESSERACT = True
    # Quick check – will raise if the binary is missing
    pytesseract.get_tesseract_version()
except Exception:
    _HAS_TESSERACT = False


def ocr_image(image_path: str, lang: str = "eng") -> str:
    """
    Run OCR on the given image file and return extracted text.

    Parameters
    ----------
    image_path : str
    lang : str
        Tesseract language code (default: "eng").

    Returns
    -------
    str  – extracted text, or "" if OCR is unavailable.
    """
    if not _HAS_TESSERACT:
        return ""
    if not os.path.isfile(image_path):
        return ""

    try:
        img = Image.open(image_path).convert("L")   # grayscale
        text = pytesseract.image_to_string(
            img,
            lang=lang,
            config="--psm 6 --oem 3",   # assume a single uniform block
        )
        return text or ""
    except Exception:
        return ""


def ocr_available() -> bool:
    """Return True if Tesseract OCR is available on this system."""
    return _HAS_TESSERACT
