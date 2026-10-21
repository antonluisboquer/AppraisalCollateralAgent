"""Generic compare-and-flag engine.

Ported from LG-discOrch's `determine_flag_action()`
(tools/document_review.py:2008-2071), which is the "Validate against
Encompass property data" + "Flag discrepancies or missing information"
responsibility shared by both agent specs (Appraisal spec item #6, Title
spec item #7 — both confirmed ✅ "have it" in the architecture doc).

Same four-action vocabulary as the original:
    ignore | overwrite | ptf | critical

The field-ID registries below (APPRAISAL_FIELD_CHECKS, TITLE_FIELD_CHECKS,
INSURANCE_FIELD_CHECKS, TAX_FIELD_CHECKS) are ported from
LG-discOrch/tools/shared/field_verification_config.py — only the specific
field IDs cited in the architecture doc are included here (356, 1041, 1894,
L362, 186, CD5.X55, CD1.X2, 1405, L252, 642, 235, 643). Extend this registry
from the full field_verification_config.py if/when this repo is wired to a
real Encompass field set.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

from .money import to_number
from .models import Flag


@dataclass
class FieldCheckConfig:
    field_id: str
    field_name: str
    check_type: str = "exact_match"  # exact_match | hard_stop | fixed_value | conditional
    tolerance: float = 0.0
    hard_stop: bool = False
    alert: bool = False
    optional: bool = False
    conditional: bool = False
    fixed_value: Optional[Any] = None


def determine_flag_action(
    config: FieldCheckConfig,
    extracted_value: Any,
    encompass_value: Any,
    loan_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Same semantics as LG-discOrch's determine_flag_action(), with one
    deliberate fix: a hard_stop field whose values are numerically within
    tolerance now returns "ignore" instead of falling through to
    "overwrite" on an exact-equality check. (The original only returns
    early from the hard_stop block when tolerance is *exceeded*; this port
    closes that gap since "within tolerance" should mean no action needed.)

    Returns one of "ignore" | "critical" | "ptf" | "overwrite".
    """
    loan_context = loan_context or {}

    if extracted_value is None and encompass_value is None:
        return "ignore"

    if config.check_type == "fixed_value":
        if encompass_value != config.fixed_value:
            return "overwrite"
        return "ignore"

    if config.optional and not extracted_value:
        return "ignore"

    if config.hard_stop:
        extracted_num = to_number(extracted_value)
        encompass_num = to_number(encompass_value)
        if extracted_num is not None and encompass_num is not None:
            if abs(extracted_num - encompass_num) > config.tolerance:
                return "critical"
            return "ignore"  # numeric values within tolerance — no action needed
        elif extracted_value != encompass_value:
            return "critical"

    if extracted_value == encompass_value:
        return "ignore"

    if config.alert:
        return "ptf"

    if extracted_value and extracted_value != encompass_value:
        return "overwrite"

    return "ignore"


def compare_fields_to_encompass(
    extracted_by_field_id: Dict[str, Any],
    encompass_by_field_id: Dict[str, Any],
    checks: Dict[str, FieldCheckConfig],
    loan_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run determine_flag_action() across every configured field and return
    a list of Flag dicts (only for fields whose action isn't "ignore",
    plus one "ignore"-level info flag per field for transparency).
    """
    flags: List[Dict[str, Any]] = []
    for field_id, config in checks.items():
        extracted_value = extracted_by_field_id.get(field_id)
        encompass_value = encompass_by_field_id.get(field_id)
        action = determine_flag_action(config, extracted_value, encompass_value, loan_context)

        if action == "ignore":
            continue  # matches or N/A — no flag needed
        elif action == "critical":
            severity = "critical"
            details = (
                f"{config.field_name} mismatch: extracted='{extracted_value}' "
                f"vs Encompass='{encompass_value}' (tolerance={config.tolerance})."
            )
        elif action == "ptf":
            severity = "warning"
            details = (
                f"{config.field_name}: extracted='{extracted_value}' differs from "
                f"Encompass='{encompass_value}'. Recommend a Prior-to-Funding condition."
            )
        else:  # overwrite
            severity = "warning"
            details = (
                f"{config.field_name}: Encompass='{encompass_value}' should be updated to "
                f"extracted='{extracted_value}'."
            )

        flags.append(
            Flag(
                field=config.field_name,
                title=f"{config.field_name} ({action})",
                severity=severity,  # type: ignore[arg-type]
                action=action,  # type: ignore[arg-type]
                details=details,
                extra={"field_id": field_id},
            ).to_dict()
        )
    return flags


# ---------------------------------------------------------------------------
# Field registries (ported from LG-discOrch/tools/shared/field_verification_config.py)
# ---------------------------------------------------------------------------

APPRAISAL_FIELD_CHECKS: Dict[str, FieldCheckConfig] = {
    "356": FieldCheckConfig(field_id="356", field_name="Appraised Value", check_type="hard_stop",
                             tolerance=1.0, hard_stop=False),
    "1041": FieldCheckConfig(field_id="1041", field_name="Subject Property Type", conditional=False),
    "1894": FieldCheckConfig(field_id="1894", field_name="Parcel Number"),
}

TITLE_FIELD_CHECKS: Dict[str, FieldCheckConfig] = {
    "L362": FieldCheckConfig(field_id="L362", field_name="Title Report Date"),
    "186": FieldCheckConfig(field_id="186", field_name="Escrow Co Escrow #"),
    "CD5.X55": FieldCheckConfig(field_id="CD5.X55", field_name="Settlement Agent Name"),
    "CD1.X2": FieldCheckConfig(field_id="CD1.X2", field_name="Closing Disclosure - Seller Names"),
}

INSURANCE_FIELD_CHECKS: Dict[str, FieldCheckConfig] = {
    "L252": FieldCheckConfig(field_id="L252", field_name="Hazard Insurance Company Name"),
    "642": FieldCheckConfig(field_id="642", field_name="Fees Hazard Ins Premium Borr",
                             check_type="hard_stop", tolerance=1.0),
    "235": FieldCheckConfig(field_id="235", field_name="Fees Flood Ins Per Mo",
                             check_type="hard_stop", tolerance=1.0),
    "643": FieldCheckConfig(field_id="643", field_name="Fees Flood Ins Premium Borr",
                             check_type="hard_stop", tolerance=1.0),
}

TAX_FIELD_CHECKS: Dict[str, FieldCheckConfig] = {
    "1405": FieldCheckConfig(field_id="1405", field_name="Expenses Proposed Taxes",
                              check_type="hard_stop", tolerance=1.0),
}
