"""Appraisal & Collateral Agent.

Spec (source of truth, see architecture doc Part 1):
    Role: Extract appraisal data and validate property details
    Triggers: Appraisal report uploaded / Property value discrepancy detected /
              RM requests appraisal review
    Responsibilities:
      - Extract key data (property value, condition, comparable sales)
      - Validate against Encompass property data
      - Flag discrepancies or missing information
      - Generate summary for RM
    Model: GPT-5 mini (document extraction)
    Tools: extract_appraisal_data(document) · get_property_details(loan_guid) ·
           validate_property_value(appraised_value, loan_amount)

MVP scope (this pass): the 3 named tools + the supporting "validate against
Encompass / flag discrepancies" comparison tool (responsibility #6, already
✅ have-it per the architecture doc) + a lightweight RM summary.

Explicitly OUT of scope for this pass (tracked as backlog in README.md):
  - condition extraction (A1 — needs new schema fields)
  - comparable sales extraction (A2 — needs new schema fields)
  - event-driven triggers (A5 — architecture decision, deferred)
  - J-A1..J-A4 job-aid items (appraisal expiration, transferred-appraisal
    workflow, File Contacts cross-check, priority-resubmit rule)
"""

from .agent import build_appraisal_collateral_agent
from .tools import (
    compare_appraisal_to_encompass,
    extract_appraisal_data,
    generate_appraisal_summary,
    get_property_details,
    validate_property_value,
)

__all__ = [
    "build_appraisal_collateral_agent",
    "extract_appraisal_data",
    "get_property_details",
    "validate_property_value",
    "compare_appraisal_to_encompass",
    "generate_appraisal_summary",
]
