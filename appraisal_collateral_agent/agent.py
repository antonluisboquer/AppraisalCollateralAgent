"""LangGraph tool-calling agent wrapper for the Appraisal & Collateral Agent.

Wires the pure functions in tools.py up as LangChain tools bound to an LLM
via `langgraph.prebuilt.create_react_agent`, matching the spec's
"Model: GPT-5 mini (document extraction)" line (model is configurable; see
DEFAULT_MODEL below — swap in the real "GPT-5 mini" identifier once available
in your account/region, nothing else needs to change).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.extraction import DocumentExtractor, OpenAIDocumentExtractor
from core.los_client import LOSClient

from .tools import (
    appraisal_report_conditions,
    check_disaster_inspection_required,
    compare_appraisal_to_encompass,
    cross_check_file_contacts_appraiser,
    determine_resubmission_priority,
    determine_secondary_appraisal_requirement,
    evaluate_transferred_appraisal,
    extract_appraisal_data,
    generate_appraisal_summary,
    get_property_details,
    validate_appraisal_expiration,
    validate_property_value,
)

DEFAULT_MODEL = os.environ.get("APPRAISAL_AGENT_MODEL", "openai:gpt-4o-mini")

SYSTEM_PROMPT = """You are the Appraisal & Collateral Agent for loan processing.

Role: Extract appraisal data and validate property details.

Triggers (handled by whatever invokes you — appraisal uploaded, property
value discrepancy detected, or an RM requesting review): for each loan you
are given a loan GUID and the appraisal report document text.

Responsibilities, in order:
1. Call extract_appraisal_data to pull structured data from the appraisal
   report (property value, address, parcel #, appraiser info).
2. Call get_property_details to read the loan's current Encompass property
   fields.
3. Call compare_appraisal_to_encompass to flag any discrepancies between the
   extracted appraisal and Encompass (appraised value, property type, parcel
   number).
4. Call validate_property_value with the extracted appraised_value and the
   loan amount (ask for it if not provided) to check the LTV and flag if the
   loan amount exceeds the appraised value.
5. Call validate_appraisal_expiration with the appraisal's effective date and
   the estimated closing date to make sure the appraisal won't go stale
   before closing (RM Job Aid p.28).
6. If File Contacts data is available, call cross_check_file_contacts_appraiser
   to verify the Appraiser entry matches the extracted report (RM Job Aid p.28).
7. Call appraisal_report_conditions on the extracted appraisal fields to pick
   up any "subject to completion/repairs/conditions" statement the appraiser
   made, then call determine_resubmission_priority with that merged into
   whatever open conditions UW has separately logged — an appraisal
   condition always warrants an immediate UW resubmit, regardless of how
   many other conditions are open (RM Job Aid p.12).
8. For a transferred appraisal (ordered externally, brought into this loan),
   call evaluate_transferred_appraisal to get the required next steps,
   including the RM-management approval gate if the TPO/borrower pays at
   closing (RM Job Aid p.20-21).
9. If a FEMA disaster was declared, call check_disaster_inspection_required
   to see whether a disaster inspection condition is needed (RM Job Aid p.21).
10. For NonQM loans, call determine_secondary_appraisal_requirement with the
    FNMA SSR score and property type to determine if a CDA/BPO/Field
    Review/second appraisal is required (RM Job Aid p.21).
11. Call generate_appraisal_summary to produce a final RM-facing summary, and
    present that summary as your final answer.

Known limitations (do not claim to check these): this agent extracts UAD
condition/quality ratings and comparable sales fields when present in the
document, but does not yet apply any pass/fail business rules to them beyond
extraction — flag to the user if a rule-based check on those fields is
needed and say explicitly it is not yet supported, rather than guessing.
"""


def build_appraisal_collateral_agent(
    extractor: Optional[DocumentExtractor] = None,
    los: Optional[LOSClient] = None,
    model: Optional[str] = None,
):
    """Build a runnable LangGraph react-agent for the Appraisal & Collateral Agent.

    Args:
        extractor: DocumentExtractor implementation (defaults to
            OpenAIDocumentExtractor — requires OPENAI_API_KEY).
        los: LOSClient implementation (required — there is no safe default
            since reading real Encompass data needs real credentials; pass a
            MockLOSClient for demos/tests).
        model: chat model identifier understood by
            `langchain.chat_models.init_chat_model` (e.g. "openai:gpt-4o-mini").
    """
    if los is None:
        raise ValueError(
            "An LOSClient is required. Use core.los_client.MockLOSClient for "
            "demos/tests, or core.los_client.CallableLOSClient wired to a real "
            "Encompass reader for production."
        )
    extractor = extractor or OpenAIDocumentExtractor()

    @tool
    def extract_appraisal_data_tool(document_text: str) -> dict:
        """Extract structured appraisal data (appraised value, property
        address/state/zip/type, parcel number, appraisal date, appraiser
        name/license) from the raw text of an Appraisal Report document."""
        return extract_appraisal_data(document_text, extractor)

    @tool
    def get_property_details_tool(loan_guid: str) -> dict:
        """Read the loan's current Encompass property fields (appraised
        value, property type, parcel number) for the given loan GUID."""
        return get_property_details(loan_guid, los)

    @tool
    def compare_appraisal_to_encompass_tool(extracted_fields: dict, encompass_fields: dict) -> list:
        """Compare extracted appraisal fields against current Encompass
        values and return a list of flags for any discrepancies."""
        return compare_appraisal_to_encompass(extracted_fields, encompass_fields)

    @tool
    def validate_property_value_tool(appraised_value: Any, loan_amount: Any,
                                      program_max_ltv: Optional[float] = None) -> dict:
        """Compute LTV = loan_amount / appraised_value and flag if the loan
        amount exceeds the appraised value or the program max LTV."""
        return validate_property_value(appraised_value, loan_amount, program_max_ltv)

    @tool
    def generate_appraisal_summary_tool(extracted_fields: dict, ltv_result: dict,
                                         comparison_flags: list) -> str:
        """Generate the final RM-facing markdown summary of the appraisal review."""
        return generate_appraisal_summary(extracted_fields, ltv_result, comparison_flags)

    @tool
    def validate_appraisal_expiration_tool(appraisal_effective_date: str, estimated_closing_date: str) -> dict:
        """Check whether the appraisal will expire (go stale) before the
        estimated closing date (120-day validity window)."""
        return validate_appraisal_expiration(appraisal_effective_date, estimated_closing_date)

    @tool
    def cross_check_file_contacts_appraiser_tool(file_contacts_appraiser: dict, appraisal_fields: dict) -> dict:
        """Cross-check the Appraiser entry in Encompass File Contacts (name,
        license #, company name) against the extracted appraisal report."""
        return cross_check_file_contacts_appraiser(file_contacts_appraiser, appraisal_fields)

    @tool
    def appraisal_report_conditions_tool(appraisal_fields: dict) -> list:
        """Derive condition-type tags directly from the appraisal report's
        own "subject to completion/repairs/conditions" statements (URAR
        Reconciliation section). Merge the result into the list passed to
        determine_resubmission_priority_tool so a condition-on-value doesn't
        get missed just because UW hasn't separately logged it yet."""
        return appraisal_report_conditions(appraisal_fields)

    @tool
    def determine_resubmission_priority_tool(open_condition_types: list) -> dict:
        """Decide whether the loan should be resubmitted to UW immediately —
        an appraisal condition always triggers immediate resubmit regardless
        of the total condition count."""
        return determine_resubmission_priority(open_condition_types)

    @tool
    def evaluate_transferred_appraisal_tool(transfer_context: dict) -> dict:
        """Determine the required next steps for a transferred-appraisal
        request, including the RM-management approval gate when the
        TPO/borrower pays for the appraisal at closing."""
        return evaluate_transferred_appraisal(transfer_context)

    @tool
    def check_disaster_inspection_required_tool(appraisal_effective_date: str,
                                                 disaster_declaration_date: str,
                                                 inspection_condition_present: bool = False) -> dict:
        """Check whether a disaster inspection is required because a FEMA
        disaster was declared after the appraisal's effective date."""
        return check_disaster_inspection_required(appraisal_effective_date, disaster_declaration_date,
                                                   inspection_condition_present)

    @tool
    def determine_secondary_appraisal_requirement_tool(
        ssr_score: Optional[float] = None,
        property_type: Optional[str] = None,
        field_review_received: bool = False,
        second_full_appraisal_received: bool = False,
    ) -> dict:
        """Determine whether a secondary appraisal product (CDA/BPO) is
        required for this NonQM loan, based on the FNMA SSR score and
        property type."""
        return determine_secondary_appraisal_requirement(ssr_score, property_type,
                                                           field_review_received,
                                                           second_full_appraisal_received)

    tools = [
        extract_appraisal_data_tool,
        get_property_details_tool,
        compare_appraisal_to_encompass_tool,
        validate_property_value_tool,
        validate_appraisal_expiration_tool,
        cross_check_file_contacts_appraiser_tool,
        appraisal_report_conditions_tool,
        determine_resubmission_priority_tool,
        evaluate_transferred_appraisal_tool,
        check_disaster_inspection_required_tool,
        determine_secondary_appraisal_requirement_tool,
        generate_appraisal_summary_tool,
    ]

    return create_react_agent(model=model or DEFAULT_MODEL, tools=tools, prompt=SYSTEM_PROMPT)
