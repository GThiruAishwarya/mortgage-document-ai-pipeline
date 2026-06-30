"""
Document Orchestrator
======================
Central pipeline coordinator for mortgage document extraction.

Stages (in order):
  1. PDF load  — pdfplumber text extraction + page image rendering (adaptive DPI)
  2. Page LLM  — concurrent Groq Vision calls (one per page, max 3 at once)
  3. Merge     — collect fields, tables, visual elements, degraded blocks from all pages
  4. Filter    — drop visual elements that describe absences ("no visible watermark")
  5. Resolve   — apply correction/addendum rules, pick winning field values
  6. Normalize — unit/format normalization for every resolved field
  7. Cross-val — annotate table rows whose values differ from resolved fields (digit-read guard)
  8. Confidence— bump/dock confidence based on source and text quality signals
  9. Mismatches— arithmetic checks (A+B+C=D, D+I=total, LTV, etc.)
 10. Classify  — rule-based + LLM document type classification

Outputs:  ExtractionResult (see models.py)

Fixes applied over original:
  • Adaptive DPI: scanned/sparse pages rendered at 200 DPI instead of 72
  • choose_dpi() imported correctly from pdf_processor
  • Visual element filter: drops elements whose description is a negation
    ("no visible watermark", "not present", etc.) — only real detections stored
  • Table cross-validation: compares every table monetary cell against the
    resolved field value; annotates mismatching rows with [TABLE READ ERROR]
    so UI users know the table cell may be an LLM digit-read error.
    Field values are NOT changed — only the table display is annotated.
  • Field merge fix: changed from last-page-wins to highest-confidence,
    first-page-preferred merge so page 1 borrower/loan fields are preserved
    alongside page 2 escrow/correction fields. The correction_resolver still
    controls any explicitly corrected fields.
  • No process_multiple_pdfs(): Streamlit callers must loop themselves
    (avoids NoSessionContext errors from nested threads)
"""

from __future__ import annotations

import os
import re
import time
import uuid
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .models import (
    DegradedBlock,
    DetectedElement,
    ExtractedTable,
    ExtractionResult,
    FieldValue,
    PageAnalysis,
)
from .pdf_processor import load_pdf, render_page_as_image, choose_dpi
from .vision_llm import analyze_page, classify_document
from .document_classifier import classify_from_text
from .correction_resolver import resolve_corrections
from .mismatch_detector import detect_mismatches
from .confidence_engine import apply_confidence_adjustments
from .text_cleaner import clean_text, normalize_field_value


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TEXT_QUALITY: frozenset = frozenset({"clean", "noisy", "degraded", "unreadable"})
_VALID_ELEMENT_TYPES: frozenset = frozenset(
    {"signature", "stamp", "watermark", "logo", "image", "handwriting", "checkbox"}
)
_VALID_CONFIDENCE: frozenset = frozenset({"high", "medium", "low"})

# Phrases that mean the element is ABSENT — filter these out before storing.
# The LLM sometimes reports "No visible watermark" as a detection; we drop those.
_NEGATION_PHRASES: Tuple[str, ...] = (
    "no visible",
    "not visible",
    "not present",
    "none visible",
    "no watermark",
    "no stamp",
    "no signature",
    "not found",
    "absent",
    "none detected",
    "not detected",
    "n/a",
    "does not appear",
    "cannot be seen",
    "could not be found",
    "is not present",
    "are not present",
)

# Closing-cost table label → resolved field name mapping
# Used by _cross_validate_tables() to match table rows to field values.
_LABEL_TO_FIELD: Dict[str, str] = {
    # Origination
    "origination charges": "origination_charges",
    "origination charge": "origination_charges",
    "origination fee": "origination_charges",
    "lender fees": "origination_charges",
    "lender origination fee": "origination_charges",
    # B — services not shopped
    "services borrower did not shop for": "services_borrower_did_not_shop",
    "services you cannot shop for": "services_borrower_did_not_shop",
    "services you did not shop for": "services_borrower_did_not_shop",
    "required services (no shop)": "services_borrower_did_not_shop",
    # C — services shopped
    "services borrower did shop for": "services_borrower_shopped",
    "services you can shop for": "services_borrower_shopped",
    "services you did shop for": "services_borrower_shopped",
    "optional services": "services_borrower_shopped",
    # D — total loan costs
    "total loan costs": "total_loan_costs",
    "total loan costs (a+b+c)": "total_loan_costs",
    "total loan costs (a + b + c)": "total_loan_costs",
    "d. total loan costs": "total_loan_costs",
    # E — taxes
    "taxes and govt fees": "taxes_and_govt_fees",
    "taxes and other govt fees": "taxes_and_govt_fees",
    "taxes and government fees": "taxes_and_govt_fees",
    "taxes, and other government fees": "taxes_and_govt_fees",
    "e. taxes and other government fees": "taxes_and_govt_fees",
    # F — prepaids
    "prepaids": "prepaids",
    "prepaid items": "prepaids",
    "f. prepaids": "prepaids",
    # G — initial escrow
    "initial escrow payment": "escrow_deposit",
    "initial escrow payment at closing": "escrow_deposit",
    "g. initial escrow payment at closing": "escrow_deposit",
    # H — other
    "other": "total_other_costs",
    "other costs": "total_other_costs",
    # I — total other costs
    "total other costs": "total_other_costs",
    "total other costs (e+f+g+h)": "total_other_costs",
    "total other costs (e + f + g + h)": "total_other_costs",
    "i. total other costs": "total_other_costs",
    # J — total closing costs
    "total closing costs": "total_closing_costs",
    "total closing costs (d+i)": "total_closing_costs",
    "total closing costs (d + i)": "total_closing_costs",
    "j. total closing costs (d + i)": "total_closing_costs",
    # Escrow
    "monthly escrow payment": "monthly_escrow",
    "monthly escrow": "monthly_escrow",
    "aggregate adjustment": "aggregate_adjustment",
}

# Monetary value pattern
_MONEY_RE = re.compile(r"^\(?[\$]?[\d,]+(?:\.\d{1,2})?\)?$")

# Confidence rank used during field merge (lower = better)
_CONFIDENCE_RANK: Dict[str, int] = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_text_quality(val: str, default: str = "clean") -> str:
    return val if val in _VALID_TEXT_QUALITY else default


def _safe_element_type(val: str, default: str = "image") -> str:
    return val if val in _VALID_ELEMENT_TYPES else default


def _safe_confidence(val: str, default: str = "medium") -> str:
    return val if val in _VALID_CONFIDENCE else default


def _is_negation_element(description: str) -> bool:
    """
    Return True if the element description reports that something is NOT present.

    The vision LLM sometimes writes entries like:
        { "type": "watermark", "description": "No visible watermark on this page" }
    These are not real detections and must be filtered out.
    """
    lower = description.lower().strip()
    return any(phrase in lower for phrase in _NEGATION_PHRASES)


def _parse_monetary(val: str) -> Optional[float]:
    """
    Parse a monetary string to float.

    Handles:  "$24,950.00"  "($140.00)"  "-$25,500"  "24950"
    Returns None if the string cannot be parsed as a number.
    """
    if not val:
        return None
    s = val.replace(",", "").replace("$", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _cross_validate_tables(
    tables: List[ExtractedTable],
    resolved_fields: Dict[str, FieldValue],
) -> List[ExtractedTable]:
    """
    Compare every monetary table cell against its matching resolved field value.

    If the table cell and the resolved field differ by more than $1 it means the
    LLM misread a digit while extracting the table (e.g. read $74,950 instead of
    $24,950).  We annotate the offending row's 'notes' key with:

        [TABLE READ ERROR: field value is $24,950.00]

    This does NOT change the resolved field value — it only flags the table row
    so the Streamlit UI can highlight it for the reviewer.

    Why not fix the table value?
    The resolved field went through 4-layer resolution + correction logic and is
    the authoritative value.  The table is purely a display artifact; overwriting
    it would hide the fact that the LLM made an OCR error, which is useful
    diagnostic information for the reviewer.
    """
    corrected_tables: List[ExtractedTable] = []

    for tbl in tables:
        new_rows: List[Dict] = []

        for row in tbl.rows:
            new_row = dict(row)

            label_raw = ""
            for key in ("label", "Label", "description", "Description", "item", "Item"):
                if row.get(key):
                    label_raw = str(row[key]).lower().strip()
                    break

            amount_raw = ""
            for key in ("amount", "Amount", "value", "Value", "cost", "Cost"):
                if row.get(key):
                    amount_raw = str(row[key]).strip()
                    break

            field_name = _LABEL_TO_FIELD.get(label_raw)

            if field_name and field_name in resolved_fields and amount_raw:
                table_val = _parse_monetary(amount_raw)
                field_obj = resolved_fields[field_name]
                field_val = _parse_monetary(field_obj.raw_value)

                if (
                    table_val is not None
                    and field_val is not None
                    and abs(table_val - field_val) > 1.0
                ):
                    note = f"[TABLE READ ERROR: field value is ${field_val:,.2f}]"
                    existing = new_row.get("notes", "") or ""
                    new_row["notes"] = f"{existing} {note}".strip() if existing else note

            new_rows.append(new_row)

        corrected_tables.append(
            ExtractedTable(
                name=tbl.name,
                source_page=tbl.source_page,
                rows=new_rows,
                confidence=tbl.confidence,
            )
        )

    return corrected_tables


def _merge_pdfplumber_tables(
    pdfplumber_tables: Dict[int, List[List]],
    page_number: int,
    existing_table_names: set,
) -> List[ExtractedTable]:
    """
    Convert raw pdfplumber table data (list-of-lists) into ExtractedTable objects.
    Only adds tables that don't already exist (by name) for this page.
    """
    result: List[ExtractedTable] = []
    page_tables = pdfplumber_tables.get(page_number, [])

    for j, raw_table in enumerate(page_tables):
        name = f"Parsed Table {j + 1} (p{page_number})"
        if name in existing_table_names or len(raw_table) < 2:
            continue

        header = raw_table[0]
        rows: List[Dict] = []

        for raw_row in raw_table[1:]:
            row_dict: Dict[str, str] = {}
            for k, cell in enumerate(raw_row):
                col_name = (
                    str(header[k]).strip()
                    if k < len(header) and header[k]
                    else f"col_{k}"
                )
                row_dict[col_name] = str(cell).strip() if cell is not None else ""
            rows.append(row_dict)

        if rows:
            result.append(
                ExtractedTable(
                    name=name,
                    source_page=page_number,
                    rows=rows,
                    confidence="high",
                )
            )

    return result


def _merge_all_fields(
    llm_page_results: List[Optional[Dict]],
    page_analyses: List,
) -> Dict[str, List[Dict]]:
    """
    Collect every field occurrence from every page into a dict of lists.

    FIX (was last-page-wins):
    Previously the loop did:
        fields[field_name] = field   # page 2 silently overwrote page 1

    Now we accumulate ALL occurrences per field. The correction_resolver
    handles fields that appear on a correction/addendum page. For all other
    fields, _pick_best_occurrence() selects the highest-confidence,
    lowest-page-number value so page 1 data is never silently discarded.

    Returns
    -------
    Dict[str, List[Dict]]
        Maps each field name to a list of occurrence dicts, each containing:
        value, page, confidence, confidence_reason, evidence.
    """
    field_occurrences: Dict[str, List[Dict]] = defaultdict(list)

    for i, gresult in enumerate(llm_page_results):
        if not gresult:
            continue
        pa = page_analyses[i]
        for field_name, field_data in (gresult.get("extracted_fields") or {}).items():
            if not field_name or not isinstance(field_data, dict):
                continue
            raw_val = str(field_data.get("value", "")).strip()
            if not raw_val or raw_val.lower() in (
                "null", "none", "n/a", "not applicable", ""
            ):
                continue

            entry = {
                "value": raw_val,
                "page": pa.page_number,
                "confidence": _safe_confidence(
                    str(field_data.get("confidence", "medium")), "medium"
                ),
                "confidence_reason": field_data.get("confidence_reason"),
                "evidence": field_data.get("evidence"),
            }
            field_occurrences[field_name].append(entry)

    return dict(field_occurrences)


def _pick_best_occurrences(
    field_occurrences: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    For each field, sort occurrences so that the best candidate is first.

    Sort key (primary → secondary):
      1. Confidence rank  — high (0) < medium (1) < low (2)
      2. Page number      — lower page preferred (original source takes priority)

    The correction_resolver will later override values for fields that appear
    on a correction/addendum page, so this ordering only matters for fields
    that are NOT corrected.

    Returns the same structure (dict of lists) with occurrences sorted.
    """
    sorted_occurrences: Dict[str, List[Dict]] = {}
    for field_name, occurrences in field_occurrences.items():
        sorted_occurrences[field_name] = sorted(
            occurrences,
            key=lambda x: (
                _CONFIDENCE_RANK.get(x.get("confidence", "low"), 2),
                x.get("page", 999),
            ),
        )
    return sorted_occurrences


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DocumentOrchestrator:
    """
    Main pipeline coordinator.

    Parameters
    ----------
    api_key : str
        Groq API key.
    max_workers : int
        Maximum concurrent vision LLM calls per document (default 3).
        Keep at <= 3 to stay within Groq rate limits on the free tier.
    model : str
        Vision model to use for page analysis.
        Automatically falls back to secondary vision models on 404 / rate limit.
    """

    def __init__(
        self,
        api_key: str,
        max_workers: int = 3,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty Groq API key string")
        self.api_key = api_key
        self.max_workers = max_workers
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_single_pdf(
        self,
        pdf_path: str,
        filename: str,
        progress_callback=None,
    ) -> ExtractionResult:
        """
        Run the full extraction pipeline on one PDF file.

        Parameters
        ----------
        pdf_path : str
            Absolute or relative path to the PDF on disk.
        filename : str
            Display filename (shown in the UI and result object).
        progress_callback : callable, optional
            Called as progress_callback(fraction: float, message: str)
            where fraction is in [0, 1].  Safe to pass a Streamlit
            progress-bar update function.

        Returns
        -------
        ExtractionResult
            Fully populated result object (see models.py).
        """
        start_time = time.time()
        doc_id = str(uuid.uuid4())[:8]

        # ------------------------------------------------------------------
        # Stage 1 — Load PDF
        # ------------------------------------------------------------------
        _progress(progress_callback, 0.02, f"Loading {filename}...")

        try:
            doc, page_analyses, pdfplumber_tables = load_pdf(pdf_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load PDF '{filename}': {exc}") from exc

        total_pages = len(doc)
        _progress(progress_callback, 0.10, f"Loaded {total_pages} page(s)")

        # ------------------------------------------------------------------
        # Stage 2 — Render page images (adaptive DPI)
        # ------------------------------------------------------------------
        _progress(progress_callback, 0.12, "Rendering page images...")

        page_images = []
        for i, pa in enumerate(page_analyses):
            page = doc[i]
            dpi = choose_dpi(pa.raw_text)          # 72 DPI digital, 200 DPI scanned
            img = render_page_as_image(page, dpi=dpi)
            page_images.append(img)
            pa.raw_text = clean_text(pa.raw_text)  # strip noise before LLM

        _progress(progress_callback, 0.18, "Sending pages to Groq Vision...")

        # ------------------------------------------------------------------
        # Stage 3 — Concurrent Groq Vision analysis
        # ------------------------------------------------------------------
        llm_page_results: List[Optional[Dict]] = [None] * total_pages

        def _analyze_one(idx: int) -> Tuple[int, Dict]:
            pa = page_analyses[idx]
            try:
                result = analyze_page(
                    image=page_images[idx],
                    page_num=pa.page_number,
                    total_pages=total_pages,
                    raw_text=pa.raw_text,
                    api_key=self.api_key,
                    model=self.model,
                )
                result = result or {}
            except Exception as exc:
                result = {"_error": str(exc), "_traceback": traceback.format_exc()}
            result["_page_num"] = pa.page_number
            return idx, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_analyze_one, i): i for i in range(total_pages)}
            done = 0
            for future in as_completed(futures):
                idx, result = future.result()
                llm_page_results[idx] = result
                done += 1
                pct = 0.18 + 0.44 * (done / total_pages)
                _progress(progress_callback, pct, f"Vision: page {done}/{total_pages} done")

        _progress(progress_callback, 0.64, "Merging page results...")

        # ------------------------------------------------------------------
        # Stage 4 — Merge: collect all fields, elements, tables, blocks
        # ------------------------------------------------------------------
        # FIX: use _merge_all_fields() + _pick_best_occurrences() instead of
        # the old last-page-wins loop.  Page 1 borrower/loan fields are now
        # preserved; the correction_resolver still controls corrected fields.
        all_detected_elements: List[DetectedElement] = []
        all_tables: List[ExtractedTable] = []
        all_degraded_blocks: List[DegradedBlock] = []

        for i, gresult in enumerate(llm_page_results):
            if not gresult:
                continue

            pa = page_analyses[i]

            # -- Update PageAnalysis in-place from LLM response --------
            raw_tq = gresult.get("text_quality", pa.text_quality)
            pa.text_quality = _safe_text_quality(str(raw_tq), pa.text_quality)
            pa.is_correction_page = bool(gresult.get("is_correction_page", pa.is_correction_page))
            pa.has_degraded_text = bool(gresult.get("has_degraded_text", pa.has_degraded_text))
            raw_targets = gresult.get("correction_targets", [])
            pa.correction_targets = raw_targets if isinstance(raw_targets, list) else []

            # -- Visual elements (with negation filter) -----------------
            for elem in gresult.get("detected_elements", []) or []:
                if not isinstance(elem, dict):
                    continue
                description = str(elem.get("description", "")).strip()
                if _is_negation_element(description):
                    # LLM reported an absence — not a real detection, skip it
                    continue
                try:
                    all_detected_elements.append(
                        DetectedElement(
                            element_type=_safe_element_type(
                                str(elem.get("type", "image")), "image"
                            ),
                            page=pa.page_number,
                            description=description,
                            confidence=_safe_confidence(
                                str(elem.get("confidence", "medium")), "medium"
                            ),
                        )
                    )
                except Exception:
                    pass

            # -- Degraded text blocks -----------------------------------
            for blk in gresult.get("degraded_text_blocks", []) or []:
                if not isinstance(blk, dict):
                    continue
                partial = str(blk.get("partial_text", "")).strip()
                location = str(blk.get("location", "unknown")).strip()
                if not partial and not location:
                    continue
                conf = str(blk.get("confidence", "low"))
                if conf not in ("medium", "low"):
                    conf = "low"
                try:
                    all_degraded_blocks.append(
                        DegradedBlock(
                            page=pa.page_number,
                            location=location,
                            partial_text=partial,
                            reason=str(blk.get("reason", "degraded scan")),
                            confidence=conf,
                        )
                    )
                except Exception:
                    pass

            # -- Tables (LLM-extracted) ---------------------------------
            for tbl in gresult.get("tables", []) or []:
                if not isinstance(tbl, dict):
                    continue
                raw_rows = tbl.get("rows") or []
                if not raw_rows:
                    continue
                safe_rows: List[Dict[str, str]] = []
                for row in raw_rows:
                    if isinstance(row, dict):
                        safe_rows.append(
                            {
                                str(k): (str(v).strip() if v is not None else "")
                                for k, v in row.items()
                            }
                        )
                if not safe_rows:
                    continue
                try:
                    all_tables.append(
                        ExtractedTable(
                            name=str(tbl.get("name", f"Table (p{pa.page_number})")),
                            source_page=pa.page_number,
                            rows=safe_rows,
                            confidence=(
                                "high" if pa.text_quality == "clean" else "medium"
                            ),
                        )
                    )
                except Exception:
                    pass

            # -- Tables (pdfplumber-parsed, high-fidelity fallback) ----
            existing_names = {
                t.name for t in all_tables if t.source_page == pa.page_number
            }
            pb_tables = _merge_pdfplumber_tables(
                pdfplumber_tables, pa.page_number, existing_names
            )
            all_tables.extend(pb_tables)

        # FIX: collect all field occurrences across pages, then sort so the
        # best (highest-confidence, lowest-page) occurrence is first per field.
        # This replaces the old last-page-wins loop and ensures page 1 fields
        # (borrower_name, loan_amount, origination_charges, etc.) survive even
        # when page 2 also returns some of the same field names.
        raw_field_occurrences = _merge_all_fields(llm_page_results, page_analyses)
        all_fields = _pick_best_occurrences(raw_field_occurrences)

        _progress(progress_callback, 0.70, "Resolving corrections...")

        # ------------------------------------------------------------------
        # Stage 5 — Correction resolution
        # ------------------------------------------------------------------
        resolved_fields, corrections_applied = resolve_corrections(
            page_analyses, all_fields
        )

        # ------------------------------------------------------------------
        # Stage 6 — Normalize resolved field values
        # ------------------------------------------------------------------
        for field_name, fv in list(resolved_fields.items()):
            norm = normalize_field_value(field_name, fv.raw_value)
            resolved_fields[field_name] = fv.model_copy(
                update={"normalized_value": norm}
            )

        _progress(progress_callback, 0.76, "Cross-validating table values...")

        # ------------------------------------------------------------------
        # Stage 7 — Table cross-validation
        # Annotate table rows where the LLM digit-read differs from the
        # resolved field value.  Does NOT change field values.
        # ------------------------------------------------------------------
        all_tables = _cross_validate_tables(all_tables, resolved_fields)

        _progress(progress_callback, 0.80, "Applying confidence adjustments...")

        # ------------------------------------------------------------------
        # Stage 8 — Confidence adjustments
        # ------------------------------------------------------------------
        resolved_fields = apply_confidence_adjustments(resolved_fields, page_analyses)

        _progress(progress_callback, 0.86, "Detecting arithmetic mismatches...")

        # ------------------------------------------------------------------
        # Stage 9 — Mismatch detection
        # ------------------------------------------------------------------
        mismatches = detect_mismatches(resolved_fields)

        _progress(progress_callback, 0.92, "Classifying document type...")

        # ------------------------------------------------------------------
        # Stage 10 — Document classification
        # ------------------------------------------------------------------
        page_summaries = [
            {
                "page_num": r.get("_page_num", i + 1),
                "page_type": r.get("page_type", "unknown"),
                "extracted_fields": list((r.get("extracted_fields") or {}).keys()),
                "is_correction_page": r.get("is_correction_page", False),
            }
            for i, r in enumerate(llm_page_results)
            if r
        ]
        llm_cls = classify_document(
            page_summaries,
            self.api_key,
            model=self.model,
        )
        doc_type, doc_confidence, doc_reason = classify_from_text(
            page_analyses, llm_cls
        )

        elapsed = round(time.time() - start_time, 2)
        _progress(progress_callback, 1.0, f"Done — {elapsed}s")

        # ------------------------------------------------------------------
        # Assemble and return result
        # ------------------------------------------------------------------
        return ExtractionResult(
            document_id=doc_id,
            filename=filename,
            document_type=doc_type,
            document_type_confidence=doc_confidence,
            document_type_reason=doc_reason,
            fields=resolved_fields,
            tables=all_tables,
            detected_elements=all_detected_elements,
            degraded_text_blocks=all_degraded_blocks,
            corrections_applied=corrections_applied,
            mismatches=mismatches,
            page_analyses=page_analyses,
            processing_time_seconds=elapsed,
            pages_processed=total_pages,
        )


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------

def _progress(
    callback,
    fraction: float,
    message: str,
) -> None:
    """Call the progress callback safely; silently ignore errors."""
    if callback is None:
        return
    try:
        callback(fraction, message)
    except Exception:
        pass