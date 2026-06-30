"""
Text Cleaner
=============
Cleans raw PDF text-layer content before it is sent to the vision LLM.

Layered approach:
  1. ftfy  — fixes Unicode/encoding artifacts (â€œ → ", etc.)
  2. Control-character and decorative noise removal
  3. Context-aware numeric OCR substitutions — only fire adjacent to digits/$ signs
  4. Word-level OCR confusion pairs — only fire inside word tokens
  5. Field-type normalisation per field after extraction

Key rule: uncertain substitution is worse than none — every rule has a
context guard to prevent false positives.
"""

import re
from typing import Optional

try:
    import ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False


_NOISE_PATTERNS = [
    r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]",
    r"[▪◦□■●◆▶►→←↑↓©®™°•]",
]

_NUMERIC_CONTEXT_RULES = [
    # O ↔ 0  (fixed-width lookbehind only — Python re requires fixed-width)
    (r"(?<=\d)O(?=\d)", "0"),
    (r"(?<=\d)O(?=\b)", "0"),
    (r"\bO(?=\d)", "0"),
    # Monetary O fix: use capture group instead of variable-width lookbehind
    (r"(\$\s{0,2})O", r"\g<1>0"),

    # I / l / | ↔ 1
    (r"(?<=\d)[Il|](?=\d)", "1"),
    (r"(?<=\d)[Il|](?=\b)", "1"),
    (r"\b[Il](?=\d)", "1"),

    # S ↔ 5
    (r"(?<=\d)S(?=\d)", "5"),
    (r"\bS(?=\d{2,})", "5"),

    # Z ↔ 2
    (r"(?<=\d)Z(?=\d)", "2"),
    (r"\bZ(?=\d{3,})", "2"),

    # B ↔ 8
    (r"(?<=\d)B(?=\d)", "8"),

    # G ↔ 6
    (r"(?<=\d)G(?=\d)", "6"),

    # Remove stray spaces after $
    (r"(?<=\$)\s+", ""),
]

_WORD_LEVEL_RULES = [
    (r"(?<=[a-z])rn(?=[a-z])", "m"),
    (r"(?<=[a-z])cl(?=[a-z])", "d"),
    (r"(?<=[a-z])li(?=[a-z])", "h"),
    (r"(?<=[a-z])vv(?=[a-z])", "w"),
    (r"(?<=[a-z])ii(?=[a-z])", "u"),
]

_MONETARY_CLEAN = re.compile(r"[^\d.\-()]")
_PERCENTAGE_CLEAN = re.compile(r"[^\d.]")
_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b")


def clean_text(text: str) -> str:
    if not text:
        return text

    if _FTFY_AVAILABLE:
        text = ftfy.fix_text(text)

    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"\s{3,}", "  ", text)

    for pattern, replacement in _NUMERIC_CONTEXT_RULES:
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in _WORD_LEVEL_RULES:
        text = re.sub(pattern, replacement, text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def normalize_monetary(value: str) -> Optional[float]:
    if not value:
        return None
    v = value
    v = re.sub(r"(?<=\d)O(?=\d)", "0", v)
    v = re.sub(r"(?<=\d)[Il](?=\d)", "1", v)
    negative = "(" in v and ")" in v or v.lstrip().startswith("-")
    cleaned = _MONETARY_CLEAN.sub("", v).replace(",", "")
    try:
        result = float(cleaned)
        return -abs(result) if negative else result
    except ValueError:
        return None


def normalize_percentage(value: str) -> Optional[float]:
    if not value:
        return None
    v = value.split("%")[0]
    v = re.sub(r"O", "0", v)
    v = re.sub(r"[Il]", "1", v)
    cleaned = _PERCENTAGE_CLEAN.sub("", v)
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_date(value: str) -> Optional[str]:
    if not value:
        return None
    v = re.sub(r"(?<=\d)O(?=\d)", "0", value)
    match = _DATE_PATTERN.search(v)
    if match:
        m, d, y = match.group(1), match.group(2), match.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{m.zfill(2)}/{d.zfill(2)}/{y}"
    return value.strip()


_MONETARY_FIELD_NAMES = {
    "loan_amount", "origination_charges", "total_loan_costs",
    "total_closing_costs", "taxes_and_govt_fees", "prepaids",
    "total_other_costs", "appraised_value", "escrow_deposit",
    "monthly_escrow", "aggregate_adjustment", "insurance_annual",
    "property_tax_annual", "flood_zone_surcharge",
    "comparable_sale_1", "comparable_sale_2", "comparable_sale_3",
    "services_borrower_shopped", "services_borrower_did_not_shop",
    "adjustment_amount",
}

_PERCENTAGE_FIELD_NAMES = {
    "interest_rate", "ltv_ratio", "ltv_original", "ltv_revised",
}

_DATE_FIELD_NAMES = {
    "date_issued", "inspection_date", "correction_date",
}


def normalize_field_value(field_name: str, raw_value: str) -> Optional[object]:
    if not raw_value:
        return None
    fn = field_name.lower()
    if fn in _MONETARY_FIELD_NAMES or "$" in raw_value:
        return normalize_monetary(raw_value)
    if fn in _PERCENTAGE_FIELD_NAMES or raw_value.endswith("%"):
        return normalize_percentage(raw_value)
    if fn in _DATE_FIELD_NAMES:
        return normalize_date(raw_value)
    return raw_value.strip()