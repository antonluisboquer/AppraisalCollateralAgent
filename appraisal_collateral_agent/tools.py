"""Pure, directly-testable implementations of the Appraisal & Collateral
Agent's tools. See appraisal_collateral_agent/agent.py for the LangGraph
tool-calling wrapper that exposes these to an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.booleans import as_bool
from core.dates import days_between, parse_date
from core.extraction import DocumentExtractor
from core.flagging import APPRAISAL_FIELD_CHECKS, compare_fields_to_encompass
from core.los_client import LOSClient
from core.models import Flag
from core.money import format_currency, to_number
from core.schema_loader import load_schema

APPRAISAL_SCHEMA = load_schema("appraisal_report")

# Extracted-field-name -> Encompass field ID.
# Only the 3 fields verified in the architecture doc
# (LG-discOrch/tools/shared/field_verification_config.py:263-287) are mapped;
# the rest of appraisal_report.json's fields (address/zip/appraiser info/etc.)
# don't have a confirmed Encompass field ID counterpart yet.
APPRAISAL_FIELD_TO_ENCOMPASS: Dict[str, str] = {
    "appraised_value": "356",
    "property_type": "1041",
    "parcel_number": "1894",
}


# ---------------------------------------------------------------------------
# Tool 1/3: extract_appraisal_data(document)
# ---------------------------------------------------------------------------

def extract_appraisal_data(document_text: str, extractor: DocumentExtractor) -> Dict[str, Any]:
    """Schema-driven extraction of appraisal data (property value, address,
    parcel #, appraiser info) from an Appraisal Report document.

    Ported pattern: LG-discOrch/tools/tool_schema/appraisal_report.json —
    confirmed ✅ "have it" in the architecture doc, item #1.
    """
    return extractor.extract(document_text, APPRAISAL_SCHEMA)


# ---------------------------------------------------------------------------
# Tool 2/3: get_property_details(loan_guid)
# ---------------------------------------------------------------------------

def get_property_details(loan_guid: str, los: LOSClient) -> Dict[str, Any]:
    """Direct Encompass reads of property fields by loan GUID.

    Confirmed ✅ "have it" in the architecture doc, item #2
    (LG-discOrch/tools/encompass_client.py + tools/document_review.py:2119-2138).
    """
    field_ids = list(APPRAISAL_FIELD_TO_ENCOMPASS.values())
    raw = los.get_fields(loan_guid, field_ids)
    encompass_id_to_name = {v: k for k, v in APPRAISAL_FIELD_TO_ENCOMPASS.items()}
    return {encompass_id_to_name[fid]: val for fid, val in raw.items() if fid in encompass_id_to_name}


# ---------------------------------------------------------------------------
# Tool 3/3: validate_property_value(appraised_value, loan_amount)
# ---------------------------------------------------------------------------

def validate_property_value(
    appraised_value: Any,
    loan_amount: Any,
    program_max_ltv: Optional[float] = None,
) -> Dict[str, Any]:
    """Deterministic LTV check: ltv = loan_amount / appraised_value.

    This was the architecture doc's item A3 (🟡 partial -> extend):
    previously only a self-consistency check existed (extracted appraised
    value vs. Encompass's own stored value for field 356). This function adds
    the actually-missing piece: compare appraised_value against loan_amount.
    """
    av = to_number(appraised_value)
    la = to_number(loan_amount)
    flags: List[Dict[str, Any]] = []

    if av is None or av <= 0:
        flags.append(
            Flag(
                field="appraised_value",
                title="Missing or invalid appraised value",
                severity="warning",
                action="ptf",
                details="Cannot compute LTV without a valid appraised value.",
            ).to_dict()
        )
        return {"appraised_value": av, "loan_amount": la, "ltv": None, "flags": flags}

    if la is None or la < 0:
        flags.append(
            Flag(
                field="loan_amount",
                title="Missing or invalid loan amount",
                severity="warning",
                action="ptf",
                details="Cannot compute LTV without a valid loan amount.",
            ).to_dict()
        )
        return {"appraised_value": av, "loan_amount": la, "ltv": None, "flags": flags}

    ltv = la / av

    if la > av:
        flags.append(
            Flag(
                field="loan_amount",
                title="Loan amount exceeds appraised value",
                severity="critical",
                action="critical",
                details=(
                    f"Loan amount {format_currency(la)} exceeds the appraised value "
                    f"{format_currency(av)} (LTV {ltv:.2%})."
                ),
            ).to_dict()
        )

    if program_max_ltv is not None and ltv > program_max_ltv:
        flags.append(
            Flag(
                field="ltv",
                title="LTV exceeds program maximum",
                severity="critical",
                action="critical",
                details=f"LTV {ltv:.2%} exceeds the program maximum of {program_max_ltv:.2%}.",
            ).to_dict()
        )

    if not flags:
        flags.append(
            Flag(
                field="ltv",
                title="LTV within expected range",
                severity="info",
                action="ignore",
                details=f"LTV {ltv:.2%} (loan {format_currency(la)} / appraised {format_currency(av)}).",
            ).to_dict()
        )

    return {"appraised_value": av, "loan_amount": la, "ltv": round(ltv, 4), "flags": flags}


# ---------------------------------------------------------------------------
# Supporting tool for responsibility #6 ("Validate against Encompass property
# data" + "Flag discrepancies") — confirmed ✅ have-it, generalized from
# LG-discOrch's determine_flag_action().
# ---------------------------------------------------------------------------

def compare_appraisal_to_encompass(
    extracted_fields: Dict[str, Any],
    encompass_fields: Dict[str, Any],
    loan_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Compare extracted appraisal fields against current Encompass values
    and return a list of Flag dicts for anything that doesn't match.

    `extracted_fields` / `encompass_fields` are keyed by the *friendly* field
    names used in appraisal_report.json / APPRAISAL_FIELD_TO_ENCOMPASS (e.g.
    "appraised_value"), not raw Encompass field IDs — this function handles
    the translation internally.
    """
    extracted_by_id = {
        fid: extracted_fields.get(name)
        for name, fid in APPRAISAL_FIELD_TO_ENCOMPASS.items()
        if name in extracted_fields
    }
    encompass_by_id = {
        fid: encompass_fields.get(name)
        for name, fid in APPRAISAL_FIELD_TO_ENCOMPASS.items()
        if name in encompass_fields
    }
    return compare_fields_to_encompass(extracted_by_id, encompass_by_id, APPRAISAL_FIELD_CHECKS, loan_context)


# ---------------------------------------------------------------------------
# Lightweight RM summary generator (architecture doc item #7 — ❌ but
# low-effort/no-new-data, included so the agent can fulfil its "generate
# summary for RM" responsibility end-to-end).
# ---------------------------------------------------------------------------

def generate_appraisal_summary(
    extracted_fields: Dict[str, Any],
    ltv_result: Dict[str, Any],
    comparison_flags: List[Dict[str, Any]],
) -> str:
    """Produce a short RM-facing narrative summarizing this appraisal review."""
    lines: List[str] = ["## Appraisal Review Summary"]

    av = extracted_fields.get("appraised_value")
    addr = extracted_fields.get("property_address", "unknown address")
    lines.append(f"- Property: {addr}")
    lines.append(f"- Appraised value: {format_currency(to_number(av))}")

    ltv = ltv_result.get("ltv")
    if ltv is not None:
        lines.append(f"- LTV: {ltv:.2%}")

    critical = [f for f in comparison_flags + ltv_result.get("flags", []) if f.get("severity") == "critical"]
    warnings = [f for f in comparison_flags + ltv_result.get("flags", []) if f.get("severity") == "warning"]

    if critical:
        lines.append(f"\n**{len(critical)} critical issue(s):**")
        for f in critical:
            lines.append(f"  - {f['title']}: {f['details']}")
    if warnings:
        lines.append(f"\n**{len(warnings)} item(s) to review:**")
        for f in warnings:
            lines.append(f"  - {f['title']}: {f['details']}")
    if not critical and not warnings:
        lines.append("\nNo discrepancies found between the appraisal and Encompass.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# J-A1: Appraisal document expiration vs. closing date
# ---------------------------------------------------------------------------

APPRAISAL_VALIDITY_DAYS = 120  # standard conventional appraisal validity window


def validate_appraisal_expiration(
    appraisal_effective_date: Any,
    estimated_closing_date: Any,
    validity_days: int = APPRAISAL_VALIDITY_DAYS,
) -> Dict[str, Any]:
    """Flag if the appraisal will be stale (expired) by the closing date.

    RM Job Aid p.28 ("NonQM - Document Expirations"): Appraisal is one of the
    items that must not expire before the closing date. This is a *separate*
    check from `validate_property_value`'s LTV math — it's about document
    staleness relative to closing, not a value comparison (architecture doc
    item J-A1).
    """
    effective = parse_date(appraisal_effective_date)
    closing = parse_date(estimated_closing_date)

    if not effective or not closing:
        return {
            "expired": None,
            "flags": [
                Flag(field="appraisal_expiration", title="Could not verify appraisal expiration",
                     severity="warning", action="ptf",
                     details="Appraisal effective date and/or estimated closing date missing; "
                             "cannot verify the document won't expire before closing.").to_dict()
            ],
        }

    age_at_closing = days_between(effective, closing)
    expired = age_at_closing > validity_days

    if expired:
        flag = Flag(
            field="appraisal_expiration", title="Appraisal will expire before closing",
            severity="critical", action="critical",
            details=f"The appraisal will be {age_at_closing} days old at the estimated closing "
                    f"date, exceeding the {validity_days}-day validity window.",
            suggestion="Request an appraisal update/recertification of value before closing.",
        ).to_dict()
    else:
        flag = Flag(
            field="appraisal_expiration", title="Appraisal valid through closing",
            severity="info", action="ignore",
            details=f"Appraisal will be {age_at_closing} days old at closing (<= {validity_days}).",
        ).to_dict()

    return {"expired": expired, "age_at_closing_days": age_at_closing, "flags": [flag]}


# ---------------------------------------------------------------------------
# J-A3: File Contacts <-> Appraisal Report cross-check
# ---------------------------------------------------------------------------

def cross_check_file_contacts_appraiser(
    file_contacts_appraiser: Dict[str, Any],
    appraisal_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Cross-check the Appraiser entry in Encompass File Contacts against the
    extracted Appraisal Report.

    RM Job Aid p.28 (File Contacts section): the Appraiser entry (name,
    license #, company name) must match the appraisal report. Architecture
    doc item J-A3 — refines the previously-partial "compare vs Encompass"
    item with the exact 3 fields to check.

    `file_contacts_appraiser` expects: name, license_number, company_name
    (as entered in Encompass File Contacts).
    """
    pairs = [
        ("name", "appraiser_name", "Appraiser name"),
        ("license_number", "appraiser_license", "Appraiser license number"),
        ("company_name", "appraiser_company", "Appraiser company name"),
    ]
    flags: List[Dict[str, Any]] = []
    for contact_key, appraisal_key, label in pairs:
        contact_value = file_contacts_appraiser.get(contact_key)
        appraisal_value = appraisal_fields.get(appraisal_key)
        if not contact_value or not appraisal_value:
            flags.append(
                Flag(field=appraisal_key, title=f"{label} not available for cross-check",
                     severity="warning", action="ptf",
                     details=f"File Contacts {label.lower()} ({contact_value!r}) or appraisal "
                             f"{label.lower()} ({appraisal_value!r}) is missing.").to_dict()
            )
        elif str(contact_value).strip().lower() != str(appraisal_value).strip().lower():
            flags.append(
                Flag(field=appraisal_key, title=f"{label} mismatch",
                     severity="warning", action="ptf",
                     details=f"File Contacts has '{contact_value}' but the appraisal report "
                             f"shows '{appraisal_value}'.",
                     suggestion="Update File Contacts to match the appraisal report.").to_dict()
            )
        else:
            flags.append(
                Flag(field=appraisal_key, title=f"{label} matches",
                     severity="info", action="ignore",
                     details=f"'{contact_value}' matches between File Contacts and the appraisal.").to_dict()
            )
    return {"flags": flags}


# ---------------------------------------------------------------------------
# J-A4: "Appraisal = immediate resubmit" priority rule
# ---------------------------------------------------------------------------

def appraisal_report_conditions(appraisal_fields: Dict[str, Any]) -> List[str]:
    """Derive condition-type tags directly from the appraisal report's own
    "subject to" statements (URAR Reconciliation section), rather than
    relying solely on separately-tracked UW condition list entries.

    Modeled on fields present in LG-docsOrch's production Appraisal Report
    extraction schema (`appraisal_subject_to_completion`,
    `appraisal_subject_to_repairs`, `appraisal_subject_to_conditions`) that
    this repo's schema previously didn't have at all. If the appraiser's own
    certification says the value opinion is subject to something, that's
    exactly the kind of signal `determine_resubmission_priority` should
    treat as "an appraisal condition is present" -- feed the result of this
    function into that function's `open_condition_types`, merged with
    whatever UW has separately logged.
    """
    appraisal_fields = appraisal_fields or {}
    tags: List[str] = []
    if as_bool(appraisal_fields.get("appraisal_subject_to_completion")):
        tags.append("appraisal: subject to completion")
    if as_bool(appraisal_fields.get("appraisal_subject_to_repairs")):
        tags.append("appraisal: subject to repairs")
    if appraisal_fields.get("appraisal_subject_to_conditions"):
        tags.append(f"appraisal: subject to conditions ({appraisal_fields['appraisal_subject_to_conditions']})")
    return tags


def determine_resubmission_priority(open_condition_types: List[str], condition_count_threshold: int = 5) -> Dict[str, Any]:
    """Decide whether a loan should be resubmitted to UW immediately.

    RM Job Aid p.12 (Conditions Queue): "The Appraisal warrants an immediate
    resubmit to UW, regardless of how many other conditions have been
    received" — this overrides the normal rule of batching ~5 conditions
    before resubmitting (architecture doc item J-A4). This is a workflow
    rule, not a document validation.
    """
    has_appraisal_condition = any("appraisal" in c.lower() for c in open_condition_types)
    if has_appraisal_condition:
        return {
            "resubmit_now": True,
            "reason": "An appraisal condition is present — RM Job Aid p.12 requires immediate "
                      "resubmission regardless of the total condition count.",
        }
    if len(open_condition_types) >= condition_count_threshold:
        return {
            "resubmit_now": True,
            "reason": f"{len(open_condition_types)} conditions received, meets the "
                      f"{condition_count_threshold}-condition batching threshold.",
        }
    return {
        "resubmit_now": False,
        "reason": f"Only {len(open_condition_types)} condition(s) received (no appraisal "
                  f"condition); wait for more before resubmitting unless a deal-breaker applies.",
    }


# ---------------------------------------------------------------------------
# J-A2: Transferred Appraisal workflow + payment-approval gate
# ---------------------------------------------------------------------------

def evaluate_transferred_appraisal(transfer_context: Dict[str, Any]) -> Dict[str, Any]:
    """Determine the required next step(s) for a transferred-appraisal request.

    RM Job Aid p.20-21: distinct from Appraisal Corrections. Wholesale
    brokers must submit a transfer request form to the NQMF Appraisal
    Transfer Desk; if the TPO/Borrower intends to pay for the appraisal at
    closing, the RM must request the invoice and get RM management approval,
    saved to the conversation log, before that can happen (architecture doc
    item J-A2 — a workflow/approval-gate rule, not a data validation).

    `transfer_context` expects: channel ("wholesale" | "correspondent"),
    transfer_request_submitted (bool), transfer_approved (bool),
    tpo_or_borrower_pays_at_closing (bool), management_approval_obtained (bool).
    """
    channel = (transfer_context.get("channel") or "").lower()
    steps: List[str] = []
    blocked = False

    if channel == "correspondent":
        return {
            "applicable": False,
            "steps": ["Correspondent Lenders handle their own appraisal transfers; "
                      "NQMF review is not required."],
            "blocked": False,
        }

    if not as_bool(transfer_context.get("transfer_request_submitted")):
        steps.append("Submit the appraisal transfer request form to appraisaltransfers@nqmf.com.")
        blocked = True
    elif not as_bool(transfer_context.get("transfer_approved")):
        steps.append("Awaiting Appraisal Transfer Desk approval; UW will review as part of normal procedure.")
        blocked = True
    else:
        steps.append("Transfer approved — appraisal documents uploaded; send to UW for final review.")

    if as_bool(transfer_context.get("tpo_or_borrower_pays_at_closing")):
        if not as_bool(transfer_context.get("management_approval_obtained")):
            steps.append(
                "TPO/Borrower intends to pay for the appraisal at closing: RM must request a "
                "copy of the invoice and obtain approval from RM management BEFORE proceeding; "
                "save the approval to the conversation log and loan notes."
            )
            blocked = True
        else:
            steps.append("RM management approval for borrower-paid appraisal at closing is on file.")

    return {"applicable": True, "steps": steps, "blocked": blocked}


# ---------------------------------------------------------------------------
# Disaster Inspections (RM Job Aid p.21 — not in architecture doc's numbered
# items, but called out explicitly in the Appraisals section)
# ---------------------------------------------------------------------------

def check_disaster_inspection_required(
    appraisal_effective_date: Any,
    disaster_declaration_date: Any,
    inspection_condition_present: bool = False,
) -> Dict[str, Any]:
    """Flag if a disaster inspection is needed.

    RM Job Aid p.21: a disaster inspection may be required if a FEMA-declared
    disaster occurred *after* the original appraisal's effective date. Ops
    management circulates FEMA declarations; this function does the date
    comparison and checks whether UW has already added the required
    condition.
    """
    appraisal_date = parse_date(appraisal_effective_date)
    disaster_date = parse_date(disaster_declaration_date)

    if not appraisal_date or not disaster_date:
        return {
            "inspection_required": None,
            "flags": [
                Flag(field="disaster_inspection", title="Cannot evaluate disaster inspection need",
                     severity="warning", action="ptf",
                     details="Appraisal effective date and/or disaster declaration date missing.").to_dict()
            ],
        }

    inspection_required = disaster_date > appraisal_date

    if not inspection_required:
        return {
            "inspection_required": False,
            "flags": [
                Flag(field="disaster_inspection", title="No disaster inspection needed",
                     severity="info", action="ignore",
                     details="The disaster declaration predates the appraisal; the appraisal "
                             "already reflects current conditions.").to_dict()
            ],
        }

    if inspection_condition_present:
        flag = Flag(
            field="disaster_inspection", title="Disaster inspection condition already on file",
            severity="info", action="ignore",
            details="UW has added the disaster-inspection condition as required.",
        ).to_dict()
    else:
        flag = Flag(
            field="disaster_inspection", title="Disaster inspection required but condition missing",
            severity="critical", action="critical",
            details=f"A FEMA disaster was declared ({disaster_declaration_date}) after the "
                    f"appraisal's effective date ({appraisal_effective_date}), but no "
                    f"disaster-inspection condition is on the loan.",
            suggestion="Request the inspection by contacting appraisaldesk@nqmf.com; confirm UW "
                       "adds the corresponding condition.",
        ).to_dict()

    return {"inspection_required": True, "flags": [flag]}


# ---------------------------------------------------------------------------
# Secondary Appraisal Review requirement (CDA / BPO / SSR, RM Job Aid p.21)
# ---------------------------------------------------------------------------

SSR_SCORE_THRESHOLD = 2.5


def determine_secondary_appraisal_requirement(
    ssr_score: Optional[float] = None,
    property_type: Optional[str] = None,
    field_review_received: bool = False,
    second_full_appraisal_received: bool = False,
) -> Dict[str, Any]:
    """Determine whether a secondary appraisal product (CDA/BPO/Field
    Review/2nd appraisal) is required, and which one, per RM Job Aid p.21.

    A secondary product is required on every NonQM loan UNLESS:
      - FNMA SSR score <= 2.5, OR
      - a Field Review was received, OR
      - a Second Full Appraisal was received.

    If required:
      - CDA on a 1-4 unit property when SSR > 2.5 (or no SSR available), OR
      - BPO on Mixed Use / 5-10 unit residential properties.
    """
    property_type_l = (property_type or "").lower()
    is_mixed_or_multi = any(k in property_type_l for k in ("mixed use", "mixed-use", "5-unit",
                                                             "5+", "multi", "5-10"))

    if field_review_received:
        return {"required": False, "product": None,
                "reason": "A Field Review has already been received — no further product needed."}
    if second_full_appraisal_received:
        return {"required": False, "product": None,
                "reason": "A Second Full Appraisal has already been received — no further product needed."}
    if ssr_score is not None and ssr_score <= SSR_SCORE_THRESHOLD:
        return {"required": False, "product": None,
                "reason": f"FNMA SSR score {ssr_score} <= {SSR_SCORE_THRESHOLD} — no secondary "
                          f"product is required."}

    if is_mixed_or_multi:
        return {"required": True, "product": "BPO",
                "reason": "Mixed Use or 5-10 unit residential property requires a BPO."}

    return {
        "required": True,
        "product": "CDA",
        "reason": (f"SSR score {ssr_score} exceeds {SSR_SCORE_THRESHOLD}" if ssr_score is not None
                   else "No FNMA SSR score available") + " on a 1-4 unit property — a Collateral "
                  "Desktop Analysis (CDA) is required.",
    }
