"""Deterministic, no-API-key demo of the Appraisal & Collateral Agent pipeline.

Runs extract -> get_property_details -> compare -> validate_property_value ->
summary using a StaticExtractor (canned appraisal data) and a MockLOSClient
(in-memory Encompass fixture), so this can run anywhere with just Python.

To run the real LLM-orchestrated agent instead (tool-calling react agent),
see the bottom of this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extraction import StaticExtractor
from core.los_client import MockLOSClient

from appraisal_collateral_agent.tools import (
    compare_appraisal_to_encompass,
    determine_secondary_appraisal_requirement,
    extract_appraisal_data,
    generate_appraisal_summary,
    get_property_details,
    validate_appraisal_expiration,
    validate_property_value,
)

LOAN_GUID = "11111111-1111-1111-1111-111111111111"

# Encompass currently has a stale parcel number for this loan.
los = MockLOSClient({LOAN_GUID: {"356": 500_000, "1041": "SFR", "1894": "APN-OLD-123"}})

# Canned "extraction" result standing in for what an LLM would pull from a
# real appraisal report PDF using tool_schema/appraisal_report.json.
extractor = StaticExtractor({
    "appraised_value": 500_000,
    "property_address": "123 Main St, Anytown, CA 90000",
    "property_type": "SFR",
    "parcel_number": "APN-NEW-456",  # mismatch vs. Encompass on purpose
    "appraiser_name": "Jane Appraiser",
    "appraiser_license": "CA-12345",
})


def main():
    print(f"=== Appraisal & Collateral Agent demo — loan {LOAN_GUID} ===\n")

    extracted = extract_appraisal_data("<appraisal report document text>", extractor)
    print("1. Extracted appraisal data:", extracted)

    encompass = get_property_details(LOAN_GUID, los)
    print("2. Current Encompass property details:", encompass)

    comparison_flags = compare_appraisal_to_encompass(extracted, encompass)
    print(f"3. Comparison flags ({len(comparison_flags)}):")
    for f in comparison_flags:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")

    loan_amount = 420_000
    ltv_result = validate_property_value(extracted["appraised_value"], loan_amount)
    print(f"\n4. LTV validation (loan amount {loan_amount}):", ltv_result)

    summary = generate_appraisal_summary(extracted, ltv_result, comparison_flags)
    print("\n5. RM summary:\n")
    print(summary)

    expiration_result = validate_appraisal_expiration("2026-01-01", "2026-06-01")
    print("\n6. Appraisal expiration check (effective 2026-01-01, closing 2026-06-01):")
    for f in expiration_result["flags"]:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")

    secondary_result = determine_secondary_appraisal_requirement(ssr_score=3.1, property_type="SFR")
    print(f"\n7. Secondary appraisal requirement (SSR 3.1, SFR): {secondary_result}")


if __name__ == "__main__":
    main()

# --- To run the real LLM-orchestrated agent instead ---
#
#   export OPENAI_API_KEY=sk-...
#   from appraisal_collateral_agent import build_appraisal_collateral_agent
#   from core.los_client import MockLOSClient
#
#   agent = build_appraisal_collateral_agent(los=MockLOSClient({...}))
#   result = agent.invoke({"messages": [("user",
#       f"Review the appraisal for loan {LOAN_GUID}. Appraisal text: ...")]})
#   print(result["messages"][-1].content)
