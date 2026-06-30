"""
Document Classifier
====================
Two-stage classification:
  1. Rule-based scoring against known document type signals (fast, free).
  2. If rule-based confidence is low, defer to the Groq LLM classification.

Key fix: Loan Estimate and Closing Disclosure shared too many signals,
causing LLM disagreement and MEDIUM confidence. Each type now has
signals that are unique to it and not shared with the other.

Covers 6 document types: Loan Estimate, Property Appraisal Summary,
Closing Disclosure, Promissory Note, Deed of Trust, HUD-1 Settlement Statement.
"""

from typing import Dict, List, Optional, Tuple

from .models import PageAnalysis


_SIGNAL_MAP: Dict[str, List[str]] = {
    "Loan Estimate": [
        # Strong unique identifiers — not shared with Closing Disclosure
        "loan estimate",
        "this is not a commitment to make a loan",
        "before you close",
        "good faith estimate",
        "intent to proceed",
        "rate lock",
        "application received",
        # Section names unique to LE (TRID format)
        "projected payments",
        "estimated total monthly payment",
        "costs at closing",
        "estimated closing costs",
        "estimated cash to close",
        # Field names
        "origination charges",
        "services you cannot shop for",
        "services you can shop for",
        "services borrower did not shop for",
        "services borrower did shop for",
        "loan costs",
        "other costs",
        "prepaids",
        "initial escrow payment",
        "loan terms",
        "nmls",
        # These appear on LE but not CD
        "comparisons use this chart",
        "other considerations",
        "this form is a statement of final loan terms",
    ],
    "Property Appraisal Summary": [
        "appraisal",
        "appraised value",
        "uniform residential appraisal",
        "urar",
        "comparable sale",
        "market conditions",
        "gross living area",
        "appraiser certification",
        "subject property",
        "neighborhood",
        "site area",
        "above grade",
        "below grade",
        "condition rating",
        "quality rating",
        "reconciliation",
        "opinion of value",
        "effective date of appraisal",
        "sales comparison approach",
        "cost approach",
        "income approach",
        "reinspection",
        "reconsideration of value",
        "adjusted value",
    ],
    "Closing Disclosure": [
        # Strong unique identifiers
        "closing disclosure",
        "three business days before",
        "consummation date",
        "cash to close",
        # Section names unique to CD
        "summaries of transactions",
        "due from borrower at closing",
        "adjustments for items paid by seller",
        "paid already by or on behalf of borrower",
        "due to seller at closing",
        # CD-specific fields
        "loan costs paid",
        "other costs paid",
        "total closing costs paid",
        "loan disclosures",
        "assumption",
        "demand feature",
        "late payment",
        "negative amortization",
        "partial payment",
        "security interest",
        "escrow account",
        "cd form",
    ],
    "Promissory Note": [
        "promissory note",
        "note rate",
        "i promise to pay",
        "monthly payment",
        "maturity date",
        "prepayment",
        "borrower's right to prepay",
        "place of payment",
        "default",
        "notice of default",
        "note holder",
        "failure to pay as required",
        "giving of notices",
    ],
    "Deed of Trust": [
        "deed of trust",
        "mortgage",
        "trustee",
        "grantor",
        "grantee",
        "security instrument",
        "covenants",
        "hazard insurance",
        "preservation and maintenance of property",
        "protection of lender's interest",
        "uniform covenants",
        "acceleration",
        "remedies",
        "foreclosure",
        "this security instrument",
        "lien",
    ],
    "HUD-1 Settlement Statement": [
        "hud-1",
        "hud 1",
        "settlement statement",
        "settlement charges",
        "gross amount due from borrower",
        "gross amount due to seller",
        "total settlement charges",
        "poc",
        "paid outside closing",
        "settlement agent",
        "place of settlement",
    ],
}

# Signals that are EXCLUSIVE to one type and should boost confidence
# significantly when found (weighted 3x in scoring).
_EXCLUSIVE_SIGNALS: Dict[str, List[str]] = {
    "Loan Estimate": [
        "loan estimate",
        "this is not a commitment to make a loan",
        "before you close",
        "projected payments",
        "estimated total monthly payment",
    ],
    "Closing Disclosure": [
        "closing disclosure",
        "consummation date",
        "summaries of transactions",
        "due from borrower at closing",
        "three business days before",
    ],
    "Property Appraisal Summary": [
        "uniform residential appraisal",
        "urar",
        "appraiser certification",
        "sales comparison approach",
        "opinion of value",
    ],
    "Promissory Note": [
        "i promise to pay",
        "note holder",
        "borrower's right to prepay",
    ],
    "Deed of Trust": [
        "this security instrument",
        "foreclosure",
        "trustee",
    ],
    "HUD-1 Settlement Statement": [
        "hud-1",
        "hud 1",
        "settlement statement",
        "paid outside closing",
    ],
}


def _score_page(text: str, signals: List[str], exclusive: List[str]) -> int:
    """
    Score a document type against page text.
    Exclusive signals count 3x to differentiate closely related types.
    """
    lower = text.lower()
    score = 0
    for s in signals:
        if s in lower:
            score += 3 if s in exclusive else 1
    return score


def classify_from_text(
    page_analyses: List[PageAnalysis],
    llm_result: Optional[Dict] = None,
) -> Tuple[str, str, str]:
    """
    Returns (document_type, confidence, reason).

    Logic:
      - Score each document type against all page texts combined.
        Exclusive signals count 3x to separate LE from CD.
      - If the top scorer has a clear lead, return high confidence.
      - If LLM and rules agree on a high-signal type, return high confidence.
      - If they disagree, return medium and prefer the higher-scoring one.
    """
    combined_text = " ".join(pa.raw_text or "" for pa in page_analyses)

    scores: Dict[str, int] = {}
    for doc_type, signals in _SIGNAL_MAP.items():
        exclusive = _EXCLUSIVE_SIGNALS.get(doc_type, [])
        scores[doc_type] = _score_page(combined_text, signals, exclusive)

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    # Clear lead: best score >= 3 AND at least 1.5x the second-best
    # (lowered from 2x so weighted scoring can create a clear lead more easily)
    rule_confident = best_score >= 3 and (second_score == 0 or best_score >= 1.5 * second_score)

    if llm_result:
        llm_type = llm_result.get("document_type", "Unknown")
        llm_conf = llm_result.get("confidence", "low")
        llm_reason = llm_result.get("reason", "LLM classification")

        if rule_confident:
            if llm_type == best_type:
                return (
                    best_type,
                    "high",
                    f"Rule-based ({best_score} pts) and LLM agree: {llm_reason}",
                )
            else:
                # Rules are confident — trust them, but note disagreement
                return (
                    best_type,
                    "medium",
                    f"Rule-based favours '{best_type}' ({best_score} pts); "
                    f"LLM favours '{llm_type}'. Defaulting to rule-based.",
                )
        elif llm_conf == "high":
            return (
                llm_type,
                "high",
                f"LLM confident ({llm_reason}); rule score too low to override ({best_score} pts)",
            )
        elif llm_conf == "medium" and best_score >= 2:
            chosen = llm_type if llm_type == best_type else best_type
            return (
                chosen,
                "medium",
                f"Partial agreement: rules={best_type}({best_score}), LLM={llm_type}",
            )
        else:
            if best_score > 0:
                return (
                    best_type,
                    "low",
                    f"Low confidence: best rule score is {best_score} pts, LLM: {llm_reason}",
                )
            return (
                llm_type if llm_type != "Unknown" else "Unknown",
                "low",
                f"Both signals weak. LLM: {llm_reason}",
            )

    if rule_confident:
        return (
            best_type,
            "high",
            f"{best_score} pts matching signals for '{best_type}'",
        )
    if best_score >= 2:
        return (
            best_type,
            "medium",
            f"Only {best_score} pts for '{best_type}'; second-best={second_score}",
        )
    return (
        "Unknown",
        "low",
        f"Insufficient signals to classify (best score: {best_score})",
    )