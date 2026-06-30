"""
PDF Processor
==============
Loads a PDF, renders page images, extracts native text layer, and
detects correction/addendum pages using an expanded keyword set.

DPI strategy:
  - Clean/native PDFs: rendered at 150 DPI (faster, enough for LLM)
  - Scanned/degraded PDFs: rendered at 200 DPI (more detail for OCR)
"""

import re
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from .models import PageAnalysis


_CORRECTION_KEYWORDS = [
    "addendum", "revised", "revision", "correction", "corrected",
    "amendment", "amended", "per county", "reinspection",
    "updated schedule", "erratum", "errata", "notice of change",
    "as-corrected", "as corrected", "schedule b", "supplement",
    "modification", "change order", "supersedes", "supercedes",
    "overrides", "replaces", "pursuant to notice", "post-cd",
    "post cd", "fee schedule update", "updated fee", "fee revision",
    "value revision", "value revised", "appraised value revised",
    "revised downward", "revised upward", "adjusted value",
    "restated", "restatement", "updated disclosure", "updated appraisal",
    "stale appraisal", "new appraisal", "reconsidered value",
    "reconsideration of value", "revised opinion", "rov ",
    "change of circumstance", "changed circumstance",
    "notice of revised", "notice of updated", "final revised",
    "corrected closing disclosure", "cd correction", "re-disclosure",
    "redisclosure", "tolerance cure", "cure payment",
    "value update", "value correction", "field correction",
    "field update", "data correction", "data update",
    "second addendum", "third addendum", "additional addendum",
    "page correction", "corrected page", "replacement page",
    "see addendum", "refer to addendum", "pursuant to addendum",
]

_CORRECTION_REGEX = re.compile(
    "|".join(re.escape(kw) for kw in _CORRECTION_KEYWORDS),
    re.IGNORECASE,
)

_CORRECTION_VALUE_PATTERN = re.compile(
    r"(?:revised|corrected|updated|new|correct)\s+(?:value|amount|fee|total|cost|rate|ltv)"
    r".*?\$[\d,]+(?:\.\d{2})?",
    re.IGNORECASE,
)


def _detect_correction_page(text: str) -> bool:
    if not text:
        return False
    if _CORRECTION_REGEX.search(text):
        return True
    if _CORRECTION_VALUE_PATTERN.search(text):
        return True
    return False


def choose_dpi(text: str) -> int:
    """
    Returns the render DPI appropriate for this page.
    Exposed (no leading underscore) so orchestrator.py can call it.
      - Sparse or empty text → likely scanned → 200 DPI
      - Dense native text    → 150 DPI (faster, sufficient for LLM)
    """
    if not text or len(text.strip()) < 80:
        return 200
    return 150


def render_page_as_image(page: fitz.Page, dpi: int = 150) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _extract_pdfplumber_tables(pdf_path: str) -> Dict[int, List]:
    tables: Dict[int, List] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                extracted = page.extract_tables()
                if extracted:
                    cleaned_tables = []
                    for table in extracted:
                        cleaned = [
                            [cell if cell is not None else "" for cell in row]
                            for row in table
                            if any(cell for cell in row if cell)
                        ]
                        if len(cleaned) > 1:
                            cleaned_tables.append(cleaned)
                    if cleaned_tables:
                        tables[page_num] = cleaned_tables
    except Exception:
        pass
    return tables


def load_pdf(pdf_path: str) -> Tuple[fitz.Document, List[PageAnalysis], Dict[int, List]]:
    doc = fitz.open(pdf_path)
    pdfplumber_tables = _extract_pdfplumber_tables(pdf_path)
    page_analyses: List[PageAnalysis] = []

    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1

        raw_text = page.get_text("text") or ""
        raw_text = raw_text.strip()

        is_correction = _detect_correction_page(raw_text)

        pa = PageAnalysis(
            page_number=page_num,
            raw_text=raw_text,
            is_correction_page=is_correction,
            text_quality="clean" if len(raw_text) > 200 else (
                "degraded" if len(raw_text) > 0 else "unreadable"
            ),
            has_degraded_text=len(raw_text) < 100,
            correction_targets=[],
        )
        page_analyses.append(pa)

    return doc, page_analyses, pdfplumber_tables