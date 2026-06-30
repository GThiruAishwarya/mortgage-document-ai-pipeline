"""
Mismatch Detector
==================
Checks whether extracted numeric totals are internally consistent.

Policy: FLAG mismatches and preserve values as-stated. Never silently recompute.
"""

import re
from typing import Dict, List, Optional

from .models import FieldValue, MismatchFlag


_TOLERANCE = 1.01


def _to_float(fv: Optional[FieldValue]) -> Optional[float]:
    if fv is None:
        return None
    v = fv.raw_value.strip()
    negative = v.startswith("(") and v.endswith(")")
    v = re.sub(r"[^\d.\-]", "", v)
    if not v:
        return None
    try:
        result = float(v)
        return -abs(result) if negative else result
    except ValueError:
        return None


def _check(
    total_field: str,
    component_fields: List[str],
    fields: Dict[str, FieldValue],
    label: str,
) -> Optional[MismatchFlag]:
    total_fv = fields.get(total_field)
    total = _to_float(total_fv)
    if total is None:
        return None

    component_values = []
    missing_components = []
    for cf in component_fields:
        fv = fields.get(cf)
        val = _to_float(fv)
        if val is not None:
            component_values.append(val)
        else:
            missing_components.append(cf)

    if not component_values:
        return None

    computed = sum(component_values)
    diff = abs(computed - total)

    if diff > _TOLERANCE:
        return MismatchFlag(
            field=total_field,
            stated_value=str(total),
            computed_value=str(round(computed, 2)),
            difference=round(diff, 2),
            description=(
                f"{label}: stated {total_field}={total:.2f}, "
                f"components sum to {computed:.2f} (diff ${diff:.2f}). "
                + (f"Missing components: {missing_components}." if missing_components else "")
                + " Values preserved as-stated per pipeline policy."
            ),
            component_fields=component_fields,
            missing_components=missing_components,
            severity="error",
            action="flagged",
        )
    return None


def _generic_total_check(fields: Dict[str, FieldValue]) -> List[MismatchFlag]:
    results = []

    total_field_names = [
        name for name in fields
        if name.startswith("total_") or name.endswith("_total")
    ]

    known_totals = {"total_loan_costs", "total_other_costs", "total_closing_costs"}

    numeric_non_total = {
        name: _to_float(fv)
        for name, fv in fields.items()
        if not (name.startswith("total_") or name.endswith("_total"))
        and _to_float(fv) is not None
        and name not in ("ltv_original", "ltv_revised", "interest_rate",
                         "loan_term", "year_built")
    }

    for total_name in total_field_names:
        if total_name in known_totals:
            continue
        total = _to_float(fields.get(total_name))
        if total is None or total <= 0:
            continue
        matching_components = [
            name for name, val in numeric_non_total.items()
            if val is not None and abs(val) < total and val > 0
        ]
        if 2 <= len(matching_components) <= 6:
            computed = sum(numeric_non_total[n] for n in matching_components)
            diff = abs(computed - total)
            if diff > _TOLERANCE:
                results.append(MismatchFlag(
                    field=total_name,
                    stated_value=str(total),
                    computed_value=str(round(computed, 2)),
                    difference=round(diff, 2),
                    description=(
                        f"Generic total check: {total_name}={total:.2f} does not match "
                        f"sum of candidate components {matching_components} = {computed:.2f} "
                        f"(diff ${diff:.2f}). Values preserved as-stated."
                    ),
                    component_fields=matching_components,
                    missing_components=[],
                    severity="warning",
                    action="flagged",
                ))
    return results


def detect_mismatches(fields: Dict[str, FieldValue]) -> List[MismatchFlag]:
    flags: List[MismatchFlag] = []

    m = _check(
        total_field="total_loan_costs",
        component_fields=[
            "origination_charges",
            "services_borrower_did_not_shop",
            "services_borrower_shopped",
        ],
        fields=fields,
        label="Section A+B+C = total_loan_costs",
    )
    if m:
        flags.append(m)

    m = _check(
        total_field="total_other_costs",
        component_fields=["taxes_and_govt_fees", "prepaids"],
        fields=fields,
        label="Section E+F = total_other_costs",
    )
    if m:
        flags.append(m)

    m = _check(
        total_field="total_closing_costs",
        component_fields=["total_loan_costs", "total_other_costs"],
        fields=fields,
        label="D+I = total_closing_costs",
    )
    if m:
        flags.append(m)

    m = _check(
        total_field="total_closing_costs",
        component_fields=[
            "origination_charges",
            "services_borrower_did_not_shop",
            "services_borrower_shopped",
            "taxes_and_govt_fees",
            "prepaids",
        ],
        fields=fields,
        label="CD total check (all components)",
    )
    if m and not any(f.field == "total_closing_costs" for f in flags):
        flags.append(m)

    loan_amount = _to_float(fields.get("loan_amount"))
    appraised = _to_float(fields.get("appraised_value"))
    ltv_field = fields.get("ltv_revised") or fields.get("ltv_original")
    ltv = _to_float(ltv_field)

    if loan_amount and appraised and appraised > 0 and ltv is not None:
        ltv_pct = ltv if ltv > 1 else ltv * 100
        computed_ltv = (loan_amount / appraised) * 100
        diff = abs(computed_ltv - ltv_pct)
        if diff > 0.6:
            flags.append(MismatchFlag(
                field="ltv_original" if "ltv_original" in fields else "ltv_revised",
                stated_value=f"{ltv_pct:.3f}%",
                computed_value=f"{computed_ltv:.3f}%",
                difference=round(diff, 3),
                description=(
                    f"LTV identity: loan_amount({loan_amount:.2f}) / appraised({appraised:.2f}) "
                    f"= {computed_ltv:.3f}% but stated LTV is {ltv_pct:.3f}% "
                    f"(diff {diff:.3f}pp). Values preserved as-stated."
                ),
                component_fields=["loan_amount", "appraised_value"],
                missing_components=[],
                severity="error",
                action="flagged",
            ))

    generic_flags = _generic_total_check(fields)
    existing_total_fields = {f.field for f in flags}
    for gf in generic_flags:
        if gf.field not in existing_total_fields:
            flags.append(gf)

    return flags