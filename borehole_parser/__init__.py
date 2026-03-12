"""
Geotechnical Borehole Parser
-----------------------------
A format-agnostic pipeline for extracting structured data from any
geotechnical borehole log report.  Works on PDFs (text-based or scanned),
images, CSV/Excel, and raw text – regardless of column naming conventions
or report layout.
"""

from borehole_parser.pipeline import BoreholePipeline

__all__ = ["BoreholePipeline"]
__version__ = "1.0.0"
