"""
Vision LLM
===========
Groq vision and text model calls with:
  - Exponential backoff with jitter + model fallback
  - Semaphore (max 3 concurrent calls)
  - diskcache: results cached by page-image hash (re-runs cost $0)
  - Robust JSON recovery for truncated responses

Design principle: the page-analysis prompt is DOCUMENT-TYPE AGNOSTIC.
It does NOT hardcode a specific field list for a specific document type.
Instead it:
  1. Provides a comprehensive hint catalog covering all common mortgage
     document types (Loan Estimate, Closing Disclosure, Appraisal, Note,
     Deed of Trust, HUD-1, etc.)
  2. Explicitly instructs the LLM to extract EVERY field it can read —
     not just the ones in the catalog.
  3. Uses the catalog only as naming guidance so field names stay
     consistent across runs, not as a restriction on what gets extracted.
"""

import base64
import hashlib
import io
import json
import random
import re
import threading
import time
from typing import Optional

from groq import Groq, RateLimitError, APIStatusError
from PIL import Image

try:
    import diskcache
    _CACHE = diskcache.Cache(".llm_cache")
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE = None
    _CACHE_AVAILABLE = False

_SEMAPHORE = threading.Semaphore(3)

_VISION_MODEL_FALLBACK = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
]

_TEXT_MODEL_DEFAULT = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Comprehensive field hint catalog — covers ALL common mortgage document types.
#
# Purpose: give the LLM consistent, canonical key names when it extracts a
# field.  This is NOT a restriction — the LLM is explicitly told to extract
# fields that do NOT appear here and name them descriptively.
# ---------------------------------------------------------------------------
_FIELD_HINT_CATALOG = """
PREFERRED FIELD NAMES (use these canonical names when the field matches):

IDENTITY / PARTIES
  borrower_name              — primary borrower full name
  co_borrower_name           — co-borrower full name (if any)
  lender_name                — name of lender / lending institution
  nmls                       — NMLS ID of lender or loan officer
  appraiser_name             — appraiser full name
  appraiser_cert             — appraiser license / certification number
  settlement_agent           — name of settlement / closing agent
  trustee                    — trustee name (Deed of Trust)
  grantor                    — grantor name (Deed of Trust / Mortgage)

PROPERTY
  property_address           — full subject property address
  property_type              — property type (e.g. Single Family, Condo)
  year_built                 — year property was built
  gross_living_area          — gross living area in sq ft
  site_area                  — lot/site area
  neighborhood               — neighborhood name or description

LOAN IDENTIFICATION
  loan_number                — primary loan number / identifier
  loan_number_on_signature_page — loan number as shown on signature/ack page
  file_ref                   — appraisal or lender file reference number
  associated_loan_number     — linked/associated loan number
  loan_amount                — total loan amount in dollars
  loan_purpose               — Purchase / Refinance / Construction / Other
  loan_type                  — loan product type (e.g. Conventional, FHA, VA)
  loan_term                  — loan term (e.g. "30 Years", "24 Months")
  interest_rate              — stated note/interest rate (e.g. "9.875%")
  date_issued                — date document was issued or effective
  inspection_date            — date of property inspection (appraisal)
  consummation_date          — closing/consummation date (Closing Disclosure)
  maturity_date              — loan maturity date (Note)

LOAN ESTIMATE / CLOSING DISCLOSURE — CLOSING COST SECTIONS
  origination_charges              — Section A: origination / lender charges
  services_borrower_did_not_shop   — Section B: required services, no shopping
  services_borrower_shopped        — Section C: services borrower can shop for
  total_loan_costs                 — Total Loan Costs (A+B+C)
  taxes_and_govt_fees              — Section E: taxes and government fees
  prepaids                         — Section F/G: prepaid items
  initial_escrow_payment           — Section G/H: initial escrow payment at closing
  total_other_costs                — Total Other Costs (E+F+G+H or similar)
  total_closing_costs              — Grand total closing costs (D+I or all sections)
  cash_to_close                    — cash to close amount (Closing Disclosure)
  aggregate_adjustment             — aggregate adjustment (may be negative)

ESCROW ACCOUNT
  escrow_deposit             — initial escrow deposit amount
  monthly_escrow             — monthly escrow payment amount
  insurance_annual           — annual hazard / homeowners insurance premium
  property_tax_annual        — annual property tax amount
  flood_zone_surcharge       — flood zone surcharge amount

APPRAISAL
  appraised_value            — final appraised / opinion of value
  ltv_original               — original loan-to-value ratio (before any revision)
  ltv_revised                — revised LTV (after correction/addendum)
  adjustment_amount          — net adjustment amount (may be negative)
  adjustment_basis           — reason / description for appraisal adjustment
  comparable_sale_1          — comparable sale 1 price
  comparable_sale_2          — comparable sale 2 price
  comparable_sale_3          — comparable sale 3 price

HUD-1 / SETTLEMENT STATEMENT
  gross_amount_due_from_borrower   — HUD-1 gross amount due from borrower
  gross_amount_due_to_seller       — HUD-1 gross amount due to seller
  total_settlement_charges         — HUD-1 total settlement charges
  poc_amount                       — paid outside closing amount

PROMISSORY NOTE
  note_amount                — principal amount (Note)
  note_rate                  — interest rate stated in Note
  first_payment_date         — date of first payment
  place_of_payment           — where payments are sent

SIGNATURES / ACKNOWLEDGMENT
  borrower_signature_present — true if actual ink/scanned signature visible (not a blank line)
  correction_date            — date shown on correction / addendum page
"""

# ---------------------------------------------------------------------------
# Page-analysis prompt — fully document-type agnostic.
# ---------------------------------------------------------------------------
_PAGE_ANALYSIS_PROMPT = """You are an expert mortgage and real-estate document analyst. \
Analyze page {page_num} of {total_pages} in the attached image.

Extracted text layer (may be empty or noisy for scanned pages — use the IMAGE as the primary source):
---
{text}
---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL NUMBER READING RULES — apply to every digit you read
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Read each digit carefully. Common OCR confusions:
   • O (letter) ↔ 0 (zero): "2O26" → "2026", "OO417" → "00417"
   • I (letter) ↔ 1 (one):  "447I" → "4471"
   • l (lowercase L) ↔ 1, S ↔ 5, Z ↔ 2, B ↔ 8, G ↔ 6
2. Double-check every dollar amount — misread leading digits are the #1 error.
3. Preserve signs: ($140.00) and -$140.00 are negative; keep the sign in raw_value.
4. For OCR-noisy identifiers (loan numbers, cert numbers), clean obvious confusions
   and return your best reading with confidence "low" if uncertain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract EVERY piece of structured information on this page — including fields
NOT listed in the hint catalog below. Never skip a field because it is not in
the list.

Naming guidance (use these canonical names when the field matches):
{field_hint_catalog}

If you find a field that does NOT match any name in the catalog, create a
descriptive snake_case key for it (e.g. "prepayment_penalty_term",
"note_holder_address", "va_entitlement_amount"). Do NOT skip it.

Extract from the IMAGE independently — do not assume a field will appear on
another page. If a field is on THIS page, extract it here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL ELEMENTS RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only add an entry to detected_elements if the element is ACTUALLY VISIBLE.
• Do NOT add "no watermark present" or similar — if absent, return [].
• Valid: actual ink signature, rubber stamp impression, watermark image/text,
  logo graphic, handwritten text, checked/unchecked checkbox.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECTION / ADDENDUM PAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the page contains keywords ADDENDUM, CORRECTION, REVISED, REVISION,
REINSPECTION, AMENDMENT, ERRATA — set is_correction_page: true.
Populate correction_targets with the EXACT field, original value, corrected
value, and any date shown. This is how the pipeline knows to apply the
correction rule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNATURE PAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• borrower_signature_present: true ONLY if an actual ink/scanned signature
  or signed mark is visible — NOT a blank line or placeholder underline.
• loan_number_on_signature_page: extract the loan number shown, cleaning
  OCR noise (O→0, I→1), confidence "low" if uncertain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEGRADED TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For rotated stamps, partial text, or unreadable blocks — report in both
detected_elements (as "stamp" or appropriate type) AND degraded_text_blocks.
Still attempt to extract the partial text; set confidence "low" with the reason.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract ALL tables. Each row should have "label" and "amount" (or equivalent
column names). Also extract each table row as an individual field entry in
extracted_fields (using the canonical name from the catalog if it matches).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETURN FORMAT — output ONLY valid JSON, no markdown fences, no explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "page_type": "main|addendum|correction|signature|table|cover|other",
  "is_correction_page": false,
  "text_quality": "clean|noisy|degraded|unreadable",
  "has_degraded_text": false,
  "correction_targets": [
    {{
      "field": "canonical_field_name",
      "original_value": "value before correction",
      "corrected_value": "new corrected value",
      "correction_date": "MM/DD/YYYY or null",
      "reference_page": 1
    }}
  ],
  "detected_elements": [
    {{
      "type": "signature|stamp|watermark|logo|image|handwriting|checkbox",
      "description": "brief factual description of what is present",
      "confidence": "high|medium|low"
    }}
  ],
  "extracted_fields": {{
    "field_name": {{
      "value": "extracted value as a string",
      "confidence": "high|medium|low",
      "confidence_reason": "one-line reason when not high, else null",
      "evidence": "exact quote or precise description from the page"
    }}
  }},
  "tables": [
    {{
      "name": "descriptive table name",
      "rows": [
        {{"label": "row label", "amount": "dollar amount or value", "notes": "any notes"}}
      ]
    }}
  ],
  "degraded_text_blocks": [
    {{
      "location": "where on the page (e.g. bottom-right corner)",
      "partial_text": "best attempt at reading the degraded content",
      "reason": "smudge | ink bleed | scan artifact | rotation | low contrast | partial stamp",
      "confidence": "medium|low"
    }}
  ]
}}

Return ONLY the JSON object."""


_CLASSIFICATION_PROMPT = """You are an expert mortgage document analyst.

Below is a summary of all pages from a document:
{page_summaries}

Classify this document. Key distinctions:

LOAN ESTIMATE (LE):
- Issued BEFORE closing, at application time
- Contains "Loan Estimate" header or "Good Faith Estimate"
- Has "Projected Payments" and "Costs at Closing" sections
- Shows ESTIMATED fees, not final figures
- Typically 3 pages; issued within 3 business days of application

CLOSING DISCLOSURE (CD):
- Issued AT OR AFTER closing
- Contains "Closing Disclosure" header
- Has "Summaries of Transactions" and "Due From Borrower at Closing"
- Shows FINAL, locked figures
- Has "consummation date" and "cash to close" figures

PROPERTY APPRAISAL SUMMARY:
- Contains appraised value, comparable sales, gross living area
- Appraiser certification and signature
- Property condition ratings
- May include reinspection addendum / reconsideration of value

PROMISSORY NOTE:
- Contains "I promise to pay" language
- Maturity date, place of payment, note holder

DEED OF TRUST / MORTGAGE:
- Contains trustee, security instrument, covenants
- Foreclosure and acceleration clauses

HUD-1 SETTLEMENT STATEMENT:
- Contains "HUD-1" or "Settlement Statement" header
- Paid outside closing (POC) entries

Return ONLY a JSON object:
{{
  "document_type": "Loan Estimate|Property Appraisal Summary|Closing Disclosure|Promissory Note|Deed of Trust|HUD-1 Settlement Statement|Unknown",
  "confidence": "high|medium|low",
  "reason": "one-line explanation based on specific fields found"
}}"""


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def _cache_key(image: Image.Image, prompt: str, model: str) -> str:
    return f"v4:{model}:{_image_hash(image)}:{hashlib.md5(prompt.encode()).hexdigest()}"


def _backoff(attempt: int) -> None:
    sleep_time = min(30.0, (2.0 ** attempt)) * (0.5 + random.random() * 0.5)
    time.sleep(sleep_time)


def _repair_truncated_json(text: str) -> Optional[dict]:
    for candidate in [
        text + "}}",
        text + "}",
        text.rsplit(",", 1)[0] + "}",
        text.rsplit(",", 1)[0] + "}}",
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_json_response(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(match.group())
            if repaired and isinstance(repaired, dict):
                return repaired
    repaired = _repair_truncated_json(text)
    if repaired and isinstance(repaired, dict):
        return repaired
    return {}


def _call_groq_vision(
    client: Groq,
    image: Image.Image,
    prompt: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    retries: int = 4,
) -> Optional[str]:
    if _CACHE_AVAILABLE:
        key = _cache_key(image, prompt, model)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    img_b64 = _pil_to_b64(image)
    models_to_try = [model] + [m for m in _VISION_MODEL_FALLBACK if m != model]

    for model_id in models_to_try:
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens=8000,
                    temperature=0.0,
                )
                result = response.choices[0].message.content
                if _CACHE_AVAILABLE:
                    _CACHE.set(key, result, expire=86400)
                return result

            except RateLimitError:
                if attempt < retries - 1:
                    _backoff(attempt + 2)
                else:
                    break

            except APIStatusError as e:
                if e.status_code == 404:
                    break
                if e.status_code in (429, 503):
                    if attempt < retries - 1:
                        _backoff(attempt + 2)
                    else:
                        break
                elif attempt < retries - 1:
                    _backoff(attempt)
                else:
                    raise

            except Exception:
                if attempt < retries - 1:
                    _backoff(attempt)
                else:
                    raise

    return None


def _call_groq_text(
    client: Groq,
    prompt: str,
    model: str = _TEXT_MODEL_DEFAULT,
    retries: int = 4,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content

        except RateLimitError:
            if attempt < retries - 1:
                _backoff(attempt + 2)
            else:
                return None

        except APIStatusError as e:
            if e.status_code in (429, 503):
                if attempt < retries - 1:
                    _backoff(attempt + 2)
                else:
                    return None
            elif attempt < retries - 1:
                _backoff(attempt)
            else:
                raise

        except Exception:
            if attempt < retries - 1:
                _backoff(attempt)
            else:
                raise

    return None


def analyze_page(
    image: Image.Image,
    page_num: int,
    total_pages: int,
    raw_text: str,
    api_key: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> dict:
    """
    Analyze one page image with the vision LLM and return structured JSON.

    The prompt is document-type agnostic: it instructs the LLM to extract
    ALL fields it can find, using the canonical field catalog only as naming
    guidance, not as a restriction.
    """
    with _SEMAPHORE:
        client = Groq(api_key=api_key)
        prompt = _PAGE_ANALYSIS_PROMPT.format(
            page_num=page_num,
            total_pages=total_pages,
            text=raw_text[:3000] if raw_text else "(scanned — no text layer)",
            field_hint_catalog=_FIELD_HINT_CATALOG,
        )
        raw = _call_groq_vision(client, image, prompt, model=model)
        if not raw:
            return {}
        return _parse_json_response(raw)


def classify_document(
    page_summaries: list,
    api_key: str,
    model: str = _TEXT_MODEL_DEFAULT,
) -> dict:
    """
    Classify the document type using a text-only LLM call.
    Uses the caller-supplied model.
    """
    with _SEMAPHORE:
        client = Groq(api_key=api_key)
        summary_parts = []
        for i, s in enumerate(page_summaries):
            ef = s.get("extracted_fields", [])
            field_names = list(ef.keys()) if isinstance(ef, dict) else list(ef)
            summary_parts.append(
                f"Page {s.get('page_num', i+1)}: type={s.get('page_type','?')}, "
                f"fields={field_names}, "
                f"correction_page={s.get('is_correction_page', False)}"
            )
        summary_text = "\n\n".join(summary_parts)
        prompt = _CLASSIFICATION_PROMPT.format(page_summaries=summary_text)
        try:
            raw = _call_groq_text(client, prompt, model=model)
            return _parse_json_response(raw) if raw else {}
        except Exception:
            return {}