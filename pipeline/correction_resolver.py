"""
Correction Resolver
====================
Resolves field values when the same field appears on multiple pages,
including corrections/addenda.

Field name resolution uses a 4-layer approach:
  Layer 1 — Exact alias match (hand-coded synonyms, fastest)
  Layer 2 — Exact schema match
  Layer 3 — Semantic similarity via sentence-transformers (understands meaning,
             not just spelling — e.g. "county recorder fee" → "taxes_and_govt_fees")
  Layer 4 — rapidfuzz string similarity fallback

Correction resolution rules:
  1. A page explicitly labeled as a correction/addendum supersedes the original.
     Detection is via LLM flags AND text-pattern fallback (for appraisal-style
     language like "REVISED DOWNWARD", "REINSPECTION CONDITION ADJUSTMENT").
  2. Among multiple correction pages, the one with the latest EXPLICIT DATE wins.
  3. If no explicit date is present on any correction, HIGHER PAGE NUMBER wins.
     The correction is still applied but noted as undated in CorrectionRecord.
  4. Non-correction pages: first (lowest page number) occurrence wins.
  5. Immutable historical fields (e.g. ltv_original) are never overwritten.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Semantic embeddings (sentence-transformers) ───────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None
    np = None

# ── Fuzzy matching fallback ──────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz, process as fuzz_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from .models import FieldValue, PageAnalysis, CorrectionRecord

def _resolve_original_value(
    target: dict,
    canonical_field: str,
    non_correction_occurrences: dict,
) -> str:
    """Resolve original value preferring the LLM-provided original_value."""
    llm_original=(target.get("original_value") or "").strip()
    if llm_original and llm_original.lower() not in ("","null","none","unknown","n/a"):
        return llm_original
    occurrences=non_correction_occurrences.get(canonical_field,[])
    if occurrences:
        return occurrences[0]["value"]
    return "unknown"



# ── Schema fields (canonical names) ──────────────────────────────────────────
_SCHEMA_FIELDS = [
    "loan_number", "borrower_name", "co_borrower_name", "property_address",
    "loan_amount", "interest_rate", "loan_term", "loan_purpose", "loan_type",
    "origination_charges", "services_borrower_did_not_shop", "services_borrower_shopped",
    "total_loan_costs", "taxes_and_govt_fees", "prepaids",
    "total_other_costs", "total_closing_costs",
    "appraised_value", "property_type", "year_built", "gross_living_area",
    "appraiser_name", "appraiser_cert", "inspection_date",
    "file_ref", "associated_loan_number",
    "ltv_original", "ltv_revised", "adjustment_amount", "adjustment_basis",
    "escrow_deposit", "monthly_escrow", "aggregate_adjustment",
    "insurance_annual", "property_tax_annual", "flood_zone_surcharge",
    "comparable_sale_1", "comparable_sale_2", "comparable_sale_3",
    "lender_name", "nmls", "date_issued",
    "borrower_signature_present", "loan_number_on_signature_page",
    "correction_date",
]

# Human-readable versions used for semantic embedding.
_SCHEMA_FIELDS_READABLE = [
    "loan number", "borrower name", "co-borrower name", "property address",
    "loan amount", "interest rate", "loan term", "loan purpose", "loan type",
    "origination charges", "services borrower did not shop for", "services borrower shopped for",
    "total loan costs", "taxes and government fees", "prepaids",
    "total other costs", "total closing costs",
    "appraised value", "property type", "year built", "gross living area",
    "appraiser name", "appraiser certification number", "inspection date",
    "file reference number", "associated loan number",
    "original loan to value ratio", "revised loan to value ratio",
    "adjustment amount", "adjustment basis",
    "escrow deposit", "monthly escrow payment", "aggregate adjustment",
    "annual hazard insurance", "annual property tax", "flood zone surcharge",
    "comparable sale 1", "comparable sale 2", "comparable sale 3",
    "lender name", "NMLS number", "date issued",
    "borrower signature present", "loan number on signature page",
    "correction date",
]

_FUZZY_THRESHOLD = 65
_SEMANTIC_THRESHOLD = 0.50   # cosine similarity — lower = more permissive

# ── Exact alias dictionary ────────────────────────────────────────────────────
_FIELD_ALIASES: Dict[str, str] = {
    # Taxes / govt fees
    "recording_fee": "taxes_and_govt_fees",
    "recording_fees": "taxes_and_govt_fees",
    "govt_recording_fee": "taxes_and_govt_fees",
    "government_recording_fee": "taxes_and_govt_fees",
    "county_tax": "taxes_and_govt_fees",
    "county_taxes": "taxes_and_govt_fees",
    "transfer_taxes": "taxes_and_govt_fees",
    "transfer_tax": "taxes_and_govt_fees",
    "state_tax": "taxes_and_govt_fees",
    "state_taxes": "taxes_and_govt_fees",
    "govt_fees": "taxes_and_govt_fees",
    "government_fees": "taxes_and_govt_fees",
    # Closing costs
    "updated_closing_total": "total_closing_costs",
    "revised_closing_total": "total_closing_costs",
    "total_closing": "total_closing_costs",
    "closing_costs_total": "total_closing_costs",
    # Appraised value
    "revised_value": "appraised_value",
    "updated_value": "appraised_value",
    "corrected_value": "appraised_value",
    "market_value": "appraised_value",
    "estimated_value": "appraised_value",
    "opinion_of_value": "appraised_value",
    "property_value": "appraised_value",
    # LTV
    "ltv": "ltv_original",
    "loan_to_value": "ltv_original",
    "loan_to_value_ratio": "ltv_original",
    "ltv_ratio": "ltv_original",
    "revised_ltv": "ltv_revised",
    "updated_ltv": "ltv_revised",
    "corrected_ltv": "ltv_revised",
    # Loan identifiers
    "loan_id": "loan_number",
    "loan_no": "loan_number",
    "file_number": "file_ref",
    "file_no": "file_ref",
    # Services
    "services_not_shopped": "services_borrower_did_not_shop",
    "services_borrower_did_not_shop_for": "services_borrower_did_not_shop",
    "lender_required_services": "services_borrower_did_not_shop",
    "services_shopped": "services_borrower_shopped",
    "borrower_shopped_services": "services_borrower_shopped",
    # Origination
    "origination_fee": "origination_charges",
    "origination_fees": "origination_charges",
    "lender_fees": "origination_charges",
    # Prepaids
    "prepaid_items": "prepaids",
    "prepaid_interest": "prepaids",
    "prepaid_insurance": "prepaids",
    # Taxes annual
    "property_taxes_annual": "property_tax_annual",
    "annual_taxes": "property_tax_annual",
    "annual_property_tax": "property_tax_annual",
    # Insurance
    "hazard_insurance": "insurance_annual",
    "homeowners_insurance": "insurance_annual",
    "annual_insurance": "insurance_annual",
    # Living area
    "gross_area": "gross_living_area",
    "gla": "gross_living_area",
    "living_area": "gross_living_area",
    "sq_ft": "gross_living_area",
    "square_footage": "gross_living_area",
    # Appraiser
    "appraiser": "appraiser_name",
    "appraiser_license": "appraiser_cert",
    "appraiser_certification": "appraiser_cert",
    # Comparables
    "sale_1": "comparable_sale_1",
    "comp_1": "comparable_sale_1",
    "comparable_1": "comparable_sale_1",
    "sale_2": "comparable_sale_2",
    "comp_2": "comparable_sale_2",
    "comparable_2": "comparable_sale_2",
    "sale_3": "comparable_sale_3",
    "comp_3": "comparable_sale_3",
    "comparable_3": "comparable_sale_3",
    # Escrow
    "escrow": "escrow_deposit",
    "initial_escrow": "escrow_deposit",
    "monthly_payment_escrow": "monthly_escrow",
    # Adjustments
    "net_adjustment": "adjustment_amount",
    "total_adjustment": "adjustment_amount",
    # Lender / NMLS
    "lender": "lender_name",
    "mortgage_lender": "lender_name",
    "nmls_id": "nmls",
    "nmls_number": "nmls",
    # Dates
    "issue_date": "date_issued",
    "issuance_date": "date_issued",
    "document_date": "date_issued",
    "effective_date": "date_issued",
    # Signatures
    "signed": "borrower_signature_present",
    "signature_present": "borrower_signature_present",
    # Borrowers
    "borrower": "borrower_name",
    "co_borrower": "co_borrower_name",
    "coborrower": "co_borrower_name",
    # Property
    "address": "property_address",
    "subject_property": "property_address",
    "property": "property_address",
    # Loan terms
    "rate": "interest_rate",
    "note_rate": "interest_rate",
    "loan_rate": "interest_rate",
    "term": "loan_term",
    "loan_term_months": "loan_term",
    "amortization_term": "loan_term",
    "purpose": "loan_purpose",
    "loan_purpose_type": "loan_purpose",
    "type": "loan_type",
    "loan_product": "loan_type",
    # Year built
    "year": "year_built",
    "year_construction": "year_built",
}

_IMMUTABLE_HISTORICAL_FIELDS = frozenset({
    "ltv_original",
    "appraised_value_original",
})

# ── Semantic model (lazy-loaded on first use) ─────────────────────────────────
_ST_MODEL: Optional[object] = None
_SCHEMA_EMBEDDINGS: Optional[object] = None


def _get_semantic_model():
    """Load sentence-transformers model once and reuse."""
    global _ST_MODEL, _SCHEMA_EMBEDDINGS
    if not _SEMANTIC_AVAILABLE:
        return None, None
    if _ST_MODEL is None:
        try:
            logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            _SCHEMA_EMBEDDINGS = _ST_MODEL.encode(
                _SCHEMA_FIELDS_READABLE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            logger.info("Semantic model loaded — field name resolution is now semantic.")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers model: {e}")
            _ST_MODEL = None
            _SCHEMA_EMBEDDINGS = None
    return _ST_MODEL, _SCHEMA_EMBEDDINGS


# ── Text-pattern correction detection ────────────────────────────────────────
_CORRECTION_PAGE_TRIGGER = re.compile(
    r'\b(ADDENDUM|ADDENDA|CORRECTION|REVISED|REVISION|REINSPECTION|ERRATA|AMENDMENT)\b',
    re.IGNORECASE,
)

_FIELD_REVISION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r'appraised\s+value\b.{0,120}?'
            r'(?:REVISED|ADJUSTED|CORRECTED|revised|adjusted|corrected)\b.{0,50}?'
            r'\$\s*([\d,]+(?:\.\d{2})?)',
            re.IGNORECASE | re.DOTALL,
        ),
        'appraised_value',
    ),
    (
        re.compile(
            r'(?:REVISED|revised|ADJUSTED|adjusted)\s+'
            r'(?:DOWNWARD|UPWARD|downward|upward)\s+to\s+\$\s*([\d,]+(?:\.\d{2})?)',
            re.IGNORECASE,
        ),
        'appraised_value',
    ),
    (
        re.compile(
            r'(?:Revised|REVISED|Updated|UPDATED)\s+TOTAL\s+CLOSING\s+COSTS\s*[:\-]?\s*'
            r'\$\s*([\d,]+(?:\.\d{2})?)',
            re.IGNORECASE,
        ),
        'total_closing_costs',
    ),
    (
        re.compile(
            r'(?:recording\s+fee|govt\s+fee|government\s+fee)\b.{0,80}?'
            r'(?:REVISED|revised|CORRECTED|corrected)\s+to\s+\$\s*([\d,]+(?:\.\d{2})?)',
            re.IGNORECASE | re.DOTALL,
        ),
        'taxes_and_govt_fees',
    ),
    (
        re.compile(
            r'(?:Revised|REVISED)\s+LTV\s*[:\(]?\s*([\d.]+)\s*%',
            re.IGNORECASE,
        ),
        'ltv_revised',
    ),
]

_DATE_IN_TEXT = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})')


# ── Core helpers ──────────────────────────────────────────────────────────────

def _normalize_key(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_").replace(".", "")


def _resolve_field_name(raw_name: str) -> str:
    """
    Map any LLM-extracted field name to a canonical schema field using:
      1. Exact alias lookup  (O(1), handles common synonyms)
      2. Exact schema match
      3. Semantic cosine similarity via sentence-transformers
      4. rapidfuzz string similarity fallback
      5. Return as-is if nothing matches well enough
    """
    key = _normalize_key(raw_name)

    # Layer 1: exact alias
    if key in _FIELD_ALIASES:
        return _FIELD_ALIASES[key]

    # Layer 2: exact schema match
    if key in _SCHEMA_FIELDS:
        return key

    # Layer 3: semantic similarity
    model, schema_embs = _get_semantic_model()
    if model is not None and schema_embs is not None:
        try:
            query_text = raw_name.replace("_", " ").strip()
            query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
            scores = cosine_similarity(query_emb, schema_embs)[0]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= _SEMANTIC_THRESHOLD:
                matched = _SCHEMA_FIELDS[best_idx]
                logger.debug(
                    f"Semantic match: '{raw_name}' → '{matched}' "
                    f"(score={best_score:.3f}, readable='{_SCHEMA_FIELDS_READABLE[best_idx]}')"
                )
                return matched
        except Exception as e:
            logger.warning(f"Semantic matching failed for '{raw_name}': {e}")

    # Layer 4: rapidfuzz string similarity fallback
    if _RAPIDFUZZ_AVAILABLE:
        match = fuzz_process.extractOne(
            key,
            _SCHEMA_FIELDS,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=_FUZZY_THRESHOLD,
        )
        if match:
            return match[0]

    # Layer 5: return as-is (unknown field — still stored, just not canonicalized)
    return key


def _parse_date(date_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Parse MM/DD/YYYY (or variants) into (year, month, day) for comparison."""
    if not date_str:
        return None
    m = re.search(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})", date_str)
    if not m:
        return None
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    return (year, month, day)


def _scan_raw_text_for_corrections(
    page_analyses: List[PageAnalysis],
) -> Dict[str, List[Tuple[str, int, Optional[str]]]]:
    """
    Text-pattern fallback: detect correction language the LLM may miss.
    Handles appraisal-style language like 'REVISED DOWNWARD to $586,500'.

    Captured monetary amounts are prefixed with '$' so that FieldValue.raw_value
    displays as '$586,500.00' rather than the bare numeric string '586500.00'.
    """
    extra: Dict[str, List[Tuple[str, int, Optional[str]]]] = {}

    for pa in page_analyses:
        text = pa.raw_text or ""
        if not text or not _CORRECTION_PAGE_TRIGGER.search(text):
            continue

        dates_found = _DATE_IN_TEXT.findall(text)
        page_date = dates_found[-1] if dates_found else None

        for pattern, canonical_field in _FIELD_REVISION_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            raw_matched = m.group(1).strip()   # e.g. "586,500.00" or "83.5" for LTV
            if not raw_matched:
                continue
            # Prefix monetary fields with '$' for correct raw display.
            # LTV pattern captures a percentage — no '$' needed there.
            if canonical_field == "ltv_revised":
                display_val = f"{raw_matched}%"
            else:
                display_val = f"${raw_matched}"
            extra.setdefault(canonical_field, []).append(
                (display_val, pa.page_number, page_date)
            )

    return extra


# ── Main resolver ──────────────────────────────────────────────────────────────

def resolve_corrections(
    page_analyses: List[PageAnalysis],
    all_fields: Dict[str, List[Dict]],
) -> Tuple[Dict[str, FieldValue], List[CorrectionRecord]]:
    """
    Apply correction resolution rules across all pages.
    Returns (resolved_fields, corrections_applied).

    Tie-breaking priority (highest to lowest):
      1. Correction with the latest explicit date wins.
      2. If no correction has an explicit date, the one on the highest page number wins.
      3. For non-correction pages: first occurrence (lowest page number) wins.
    """
    corrections_applied: List[CorrectionRecord] = []

    # ── Step 1: Collect LLM-identified corrections ────────────────────────────
    correction_index: Dict[str, List[Tuple[str, int, Optional[str]]]] = {}

    for pa in page_analyses:
        if not pa.is_correction_page:
            continue
        for target in pa.correction_targets:
            raw_field = target.get("field", "")
            if not raw_field:
                continue
            canonical = _resolve_field_name(raw_field)
            corrected_val = str(target.get("corrected_value", "")).strip()
            if not corrected_val:
                continue
            correction_date = target.get("correction_date")
            correction_index.setdefault(canonical, []).append(
                (corrected_val, pa.page_number, correction_date)
            )

    # ── Step 2: Text-pattern fallback ────────────────────────────────────────
    text_corrections = _scan_raw_text_for_corrections(page_analyses)
    for canonical_field, entries in text_corrections.items():
        existing_pages = {page for _, page, _ in correction_index.get(canonical_field, [])}
        for entry in entries:
            _, page, _ = entry
            if page not in existing_pages:
                correction_index.setdefault(canonical_field, []).append(entry)

    # ── Step 3: Normalise all extracted field names to canonical names ─────────
    normalised_all_fields: Dict[str, List[Dict]] = {}
    for raw_name, occurrences in all_fields.items():
        canonical = _resolve_field_name(raw_name)
        existing = normalised_all_fields.get(canonical, [])
        existing.extend(occurrences)
        normalised_all_fields[canonical] = existing

    # ── Step 4: Resolve each field ────────────────────────────────────────────
    resolved: Dict[str, FieldValue] = {}
    all_canonical_fields = set(normalised_all_fields.keys()) | set(correction_index.keys())

    for canonical_field in all_canonical_fields:
        occurrences = normalised_all_fields.get(canonical_field, [])
        corrections = correction_index.get(canonical_field, [])

        if not occurrences and not corrections:
            continue

        # Immutable historical fields — always take first occurrence, never overwrite
        if canonical_field in _IMMUTABLE_HISTORICAL_FIELDS:
            if occurrences:
                first = sorted(occurrences, key=lambda x: x["page"])[0]
                resolved[canonical_field] = FieldValue(
                    raw_value=first["value"],
                    source_page=first["page"],
                    confidence=first.get("confidence", "medium"),
                    confidence_reason=first.get("confidence_reason"),
                    evidence=first.get("evidence"),
                    corrected=False,
                    original_value=None,
                    correction_page=None,
                    correction_date=None,
                )
            continue

        if corrections:
            def correction_sort_key(c: Tuple[str, int, Optional[str]]) -> Tuple:
                """
                Sort key for choosing the winning correction.

                Returns a tuple that sorts higher for more-authoritative corrections:
                  - (1, year, month, day, page) for dated corrections   → dated always beats undated
                  - (0, 0, 0, 0, page)          for undated corrections → higher page wins among undated
                """
                _val, page_num, date_str = c
                parsed_date = _parse_date(date_str)
                if parsed_date:
                    return (1, parsed_date[0], parsed_date[1], parsed_date[2], page_num)
                else:
                    return (0, 0, 0, 0, page_num)

            best = sorted(corrections, key=correction_sort_key, reverse=True)[0]
            corrected_val, correction_page, correction_date = best

            # Original value = first occurrence on a non-correction page
            original_occurrences = [
                o for o in occurrences
                if not any(
                    pa.page_number == o["page"] and pa.is_correction_page
                    for pa in page_analyses
                )
            ]
            if original_occurrences:
                first_orig = sorted(original_occurrences, key=lambda x: x["page"])[0]
                original_val = first_orig["value"]
                original_page = first_orig["page"]
            elif occurrences:
                first_orig = sorted(occurrences, key=lambda x: x["page"])[0]
                original_val = first_orig["value"]
                original_page = first_orig["page"]
            else:
                original_val = "unknown"
                original_page = 1

            if correction_date:
                rule = "Explicit correction/addendum page detected; latest explicit date wins"
            else:
                rule = "Explicit correction/addendum page detected; no date — higher page number wins"

            corrections_applied.append(CorrectionRecord(
                field=canonical_field,
                original_value=original_val,
                original_page=original_page,
                corrected_value=corrected_val,
                correction_page=correction_page,
                correction_date=correction_date,
                resolution_rule=rule,
            ))

            resolved[canonical_field] = FieldValue(
                raw_value=corrected_val,
                source_page=correction_page,
                confidence="high",
                confidence_reason="Value superseded by explicit correction on addendum/correction page",
                evidence=f"Correction page {correction_page} overrides original '{original_val}'",
                corrected=True,
                original_value=original_val,
                correction_page=correction_page,
                correction_date=correction_date,
            )

        else:
            # No correction found — first occurrence wins (lowest page number = primary source)
            sorted_occs = sorted(occurrences, key=lambda x: x["page"])
            best = sorted_occs[0]
            resolved[canonical_field] = FieldValue(
                raw_value=best["value"],
                source_page=best["page"],
                confidence=best.get("confidence", "medium"),
                confidence_reason=best.get("confidence_reason"),
                evidence=best.get("evidence"),
                corrected=False,
                original_value=None,
                correction_page=None,
                correction_date=None,
            )

    return resolved, corrections_applied