"""
Confidence Engine
==================
Adjusts field confidence scores after initial extraction based on:
  - Format validation (does the value look like what this field should be?)
  - Degraded text flags from the vision model

Design rule: NEVER downgrade a value that is correctly formatted but uses a
non-standard presentation style. Specifically:
  - Parenthetical negatives  ($140.00) or ($2,500.00) are valid monetary values
  - Signed values  -$140.00 or +$500  are valid monetary values
  - Percentage values  83.5  (without %) are valid when the field expects a percentage
"""

import re
from typing import Dict, List, Optional

from .models import FieldValue, PageAnalysis


_MONETARY_RE = re.compile(
    r"^"
    r"[\+\-]?"                # optional leading sign
    r"\(?"                    # optional opening paren (parenthetical negative)
    r"\$?"                    # optional dollar sign (inside or outside paren)
    r"[\d,]+"                 # integer digits with optional thousands separators
    r"(?:\.\d{1,2})?"         # optional decimal part
    r"\)?"                    # optional closing paren
    r"\s*$"
)

_PERCENTAGE_RE = re.compile(
    r"^"
    r"[\+\-]?"
    r"\d{1,3}"
    r"(?:\.\d{1,6})?"
    r"\s*%?"
    r"\s*$"
)

_DATE_RE = re.compile(
    r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$"
)

_MONETARY_FIELDS = {
    "loan_amount", "origination_charges", "services_borrower_did_not_shop",
    "services_borrower_shopped", "total_loan_costs", "taxes_and_govt_fees",
    "prepaids", "total_other_costs", "total_closing_costs",
    "appraised_value", "escrow_deposit", "monthly_escrow",
    "aggregate_adjustment", "insurance_annual", "property_tax_annual",
    "flood_zone_surcharge", "comparable_sale_1", "comparable_sale_2",
    "comparable_sale_3", "adjustment_amount",
}

_PERCENTAGE_FIELDS = {
    "interest_rate", "ltv_original", "ltv_revised", "ltv_ratio",
}

_DATE_FIELDS = {
    "inspection_date", "date_issued", "correction_date",
}


def _is_valid_monetary(value: str) -> bool:
    """Accept any monetary string including signed, parenthetical, or bare numbers."""
    v = value.strip().replace(",", "")
    return bool(_MONETARY_RE.match(v))


def _is_valid_percentage(value: str) -> bool:
    return bool(_PERCENTAGE_RE.match(value.strip()))


def _is_valid_date(value: str) -> bool:
    return bool(_DATE_RE.match(value.strip()))


def _format_check(field_name: str, value: str) -> Optional[str]:
    """
    Return a failure reason string if the value fails format validation,
    or None if valid (including unknown fields where we make no assertion).
    """
    fn = field_name.lower()

    if fn in _MONETARY_FIELDS:
        if not _is_valid_monetary(value):
            return f"Value '{value}' does not match expected monetary format"
        return None

    if fn in _PERCENTAGE_FIELDS:
        if not _is_valid_percentage(value):
            return f"Value '{value}' does not match expected percentage format"
        return None

    if fn in _DATE_FIELDS:
        if not _is_valid_date(value):
            return f"Value '{value}' does not match expected date format (MM/DD/YYYY)"
        return None

    return None


def apply_confidence_adjustments(
    resolved_fields: Dict[str, FieldValue],
    page_analyses: List[PageAnalysis],
) -> Dict[str, FieldValue]:
    """
    Walk every resolved field and adjust its confidence level based on:
      1. Format validation failure → downgrade one level
      2. Degraded source page → downgrade to medium if currently high

    Confidence levels: high → medium → low (one step per rule violation).
    Multiple violations can compound downward but never go below 'low'.
    """
    degraded_pages = {
        pa.page_number
        for pa in page_analyses
        if pa.has_degraded_text or pa.text_quality in ("degraded", "unreadable")
    }

    _LEVELS = {"high": 2, "medium": 1, "low": 0}
    _NAMES = {2: "high", 1: "medium", 0: "low"}

    for field_name, fv in resolved_fields.items():
        current_level = _LEVELS.get(fv.confidence, 1)
        reasons: List[str] = []

        if fv.confidence_reason:
            reasons.append(fv.confidence_reason)

        # Rule 1: format check
        format_failure = _format_check(field_name, fv.raw_value)
        if format_failure:
            current_level = max(0, current_level - 1)
            reasons.append(format_failure)

        # Rule 2: degraded source page
        if fv.source_page in degraded_pages and current_level == 2:
            current_level = 1
            reasons.append(f"Source page {fv.source_page} flagged as degraded")

        new_confidence = _NAMES[current_level]
        new_reason = "; ".join(reasons) if reasons else fv.confidence_reason

        resolved_fields[field_name] = fv.model_copy(update={
            "confidence": new_confidence,
            "confidence_reason": new_reason if new_confidence != "high" else None,
        })

    return resolved_fields