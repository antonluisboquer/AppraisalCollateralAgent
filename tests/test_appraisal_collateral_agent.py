from core.extraction import StaticExtractor
from core.los_client import MockLOSClient

from appraisal_collateral_agent.tools import (
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


# ---------------------------------------------------------------------------
# extract_appraisal_data
# ---------------------------------------------------------------------------

def test_extract_appraisal_data_uses_schema_and_extractor():
    canned = {"appraised_value": 525_000, "property_address": "123 Main St"}
    extractor = StaticExtractor(canned)
    result = extract_appraisal_data("<appraisal report text>", extractor)
    assert result == canned


# ---------------------------------------------------------------------------
# get_property_details
# ---------------------------------------------------------------------------

def test_get_property_details_maps_encompass_ids_to_friendly_names():
    los = MockLOSClient({"guid-1": {"356": 525_000, "1041": "SFR", "1894": "APN-1"}})
    result = get_property_details("guid-1", los)
    assert result == {"appraised_value": 525_000, "property_type": "SFR", "parcel_number": "APN-1"}


def test_get_property_details_unknown_loan_returns_nones():
    los = MockLOSClient({})
    result = get_property_details("unknown-guid", los)
    assert result == {"appraised_value": None, "property_type": None, "parcel_number": None}


# ---------------------------------------------------------------------------
# validate_property_value (architecture doc item A3)
# ---------------------------------------------------------------------------

def test_validate_property_value_normal_ltv_has_no_critical_flags():
    result = validate_property_value(appraised_value=500_000, loan_amount=400_000)
    assert result["ltv"] == 0.8
    assert all(f["severity"] != "critical" for f in result["flags"])


def test_validate_property_value_loan_exceeds_appraised_value_is_critical():
    result = validate_property_value(appraised_value=400_000, loan_amount=450_000)
    assert result["ltv"] == 1.125
    critical = [f for f in result["flags"] if f["severity"] == "critical"]
    assert len(critical) == 1
    assert "exceeds the appraised value" in critical[0]["details"]


def test_validate_property_value_exceeds_program_max_ltv_is_critical():
    result = validate_property_value(appraised_value=500_000, loan_amount=400_000, program_max_ltv=0.75)
    critical = [f for f in result["flags"] if f["severity"] == "critical"]
    assert len(critical) == 1
    assert "program maximum" in critical[0]["details"]


def test_validate_property_value_missing_appraised_value():
    result = validate_property_value(appraised_value=None, loan_amount=400_000)
    assert result["ltv"] is None
    assert result["flags"][0]["action"] == "ptf"


def test_validate_property_value_missing_loan_amount():
    result = validate_property_value(appraised_value=500_000, loan_amount=None)
    assert result["ltv"] is None
    assert result["flags"][0]["action"] == "ptf"


# ---------------------------------------------------------------------------
# compare_appraisal_to_encompass (responsibility #6)
# ---------------------------------------------------------------------------

def test_compare_appraisal_to_encompass_flags_parcel_mismatch():
    extracted = {"appraised_value": 500_000, "property_type": "SFR", "parcel_number": "APN-NEW"}
    encompass = {"appraised_value": 500_000, "property_type": "SFR", "parcel_number": "APN-OLD"}
    flags = compare_appraisal_to_encompass(extracted, encompass)
    assert len(flags) == 1
    assert flags[0]["field"] == "Parcel Number"


def test_compare_appraisal_to_encompass_no_flags_when_matching():
    fields = {"appraised_value": 500_000, "property_type": "SFR", "parcel_number": "APN-1"}
    flags = compare_appraisal_to_encompass(fields, dict(fields))
    assert flags == []


# ---------------------------------------------------------------------------
# generate_appraisal_summary
# ---------------------------------------------------------------------------

def test_generate_appraisal_summary_includes_critical_and_warning_sections():
    extracted = {"appraised_value": 400_000, "property_address": "123 Main St"}
    ltv_result = validate_property_value(appraised_value=400_000, loan_amount=450_000)
    comparison_flags = [{"severity": "warning", "title": "Parcel mismatch", "details": "APN differs"}]
    summary = generate_appraisal_summary(extracted, ltv_result, comparison_flags)
    assert "critical issue" in summary
    assert "item(s) to review" in summary
    assert "123 Main St" in summary


def test_generate_appraisal_summary_clean_case():
    extracted = {"appraised_value": 500_000, "property_address": "123 Main St"}
    ltv_result = validate_property_value(appraised_value=500_000, loan_amount=400_000)
    summary = generate_appraisal_summary(extracted, ltv_result, [])
    assert "No discrepancies" in summary


# ---------------------------------------------------------------------------
# validate_appraisal_expiration (J-A1)
# ---------------------------------------------------------------------------

def test_validate_appraisal_expiration_flags_stale_appraisal():
    result = validate_appraisal_expiration("2025-01-01", "2026-06-01")  # ~ 1.4 years later
    assert result["expired"] is True
    assert result["flags"][0]["severity"] == "critical"


def test_validate_appraisal_expiration_valid_within_window():
    result = validate_appraisal_expiration("2026-01-01", "2026-03-01")  # 59 days
    assert result["expired"] is False
    assert result["flags"][0]["severity"] == "info"


def test_validate_appraisal_expiration_missing_dates():
    result = validate_appraisal_expiration(None, "2026-03-01")
    assert result["expired"] is None
    assert result["flags"][0]["action"] == "ptf"


# ---------------------------------------------------------------------------
# cross_check_file_contacts_appraiser (J-A3)
# ---------------------------------------------------------------------------

def test_cross_check_file_contacts_appraiser_all_match():
    contacts = {"name": "Jane Doe", "license_number": "APR-123", "company_name": "Doe Appraisals"}
    appraisal = {"appraiser_name": "Jane Doe", "appraiser_license": "APR-123", "appraiser_company": "Doe Appraisals"}
    result = cross_check_file_contacts_appraiser(contacts, appraisal)
    assert all(f["severity"] == "info" for f in result["flags"])


def test_cross_check_file_contacts_appraiser_name_mismatch():
    contacts = {"name": "Jane Doe", "license_number": "APR-123", "company_name": "Doe Appraisals"}
    appraisal = {"appraiser_name": "John Smith", "appraiser_license": "APR-123", "appraiser_company": "Doe Appraisals"}
    result = cross_check_file_contacts_appraiser(contacts, appraisal)
    name_flags = [f for f in result["flags"] if f["field"] == "appraiser_name"]
    assert name_flags[0]["severity"] == "warning"


def test_cross_check_file_contacts_appraiser_missing_data():
    result = cross_check_file_contacts_appraiser({}, {})
    assert all(f["action"] == "ptf" for f in result["flags"])


# ---------------------------------------------------------------------------
# appraisal_report_conditions
# ---------------------------------------------------------------------------

def test_appraisal_report_conditions_empty_when_no_subject_to_flags():
    assert appraisal_report_conditions({"appraised_value": 500_000}) == []


def test_appraisal_report_conditions_empty_dict_when_none_passed():
    assert appraisal_report_conditions(None) == []


def test_appraisal_report_conditions_flags_subject_to_completion():
    tags = appraisal_report_conditions({"appraisal_subject_to_completion": True})
    assert any("completion" in t for t in tags)


def test_appraisal_report_conditions_flags_subject_to_repairs():
    tags = appraisal_report_conditions({"appraisal_subject_to_repairs": True})
    assert any("repairs" in t for t in tags)


def test_appraisal_report_conditions_includes_free_text_conditions():
    tags = appraisal_report_conditions({"appraisal_subject_to_conditions": "re-inspection of roof"})
    assert any("re-inspection of roof" in t for t in tags)


def test_appraisal_report_conditions_treats_string_false_as_false():
    """Regression test: LG-docsOrch's real extraction output stores this
    field as the literal string "False", not a Python bool. bool("False")
    is True in Python, so without core.booleans.as_bool() this would have
    been a false positive -- found by replaying docsOrch's golden fixtures
    (see tests/test_golden_fixtures.py)."""
    tags = appraisal_report_conditions({
        "appraisal_subject_to_completion": "False",
        "appraisal_subject_to_repairs": "False",
    })
    assert tags == []


def test_appraisal_report_conditions_treats_string_true_as_true():
    tags = appraisal_report_conditions({"appraisal_subject_to_repairs": "True"})
    assert any("repairs" in t for t in tags)


def test_appraisal_report_conditions_feed_into_resubmission_priority():
    tags = appraisal_report_conditions({"appraisal_subject_to_repairs": True})
    result = determine_resubmission_priority(tags + ["Bank Statement"])
    assert result["resubmit_now"] is True


# ---------------------------------------------------------------------------
# determine_resubmission_priority (J-A4)
# ---------------------------------------------------------------------------

def test_determine_resubmission_priority_appraisal_condition_forces_immediate_resubmit():
    result = determine_resubmission_priority(["Appraisal - updated value needed"])
    assert result["resubmit_now"] is True
    assert "appraisal" in result["reason"].lower()


def test_determine_resubmission_priority_few_conditions_waits():
    result = determine_resubmission_priority(["Bank Statement", "Paystub"])
    assert result["resubmit_now"] is False


def test_determine_resubmission_priority_enough_conditions_resubmits():
    result = determine_resubmission_priority(["A", "B", "C", "D", "E"])
    assert result["resubmit_now"] is True


# ---------------------------------------------------------------------------
# evaluate_transferred_appraisal (J-A2)
# ---------------------------------------------------------------------------

def test_evaluate_transferred_appraisal_correspondent_not_applicable():
    result = evaluate_transferred_appraisal({"channel": "correspondent"})
    assert result["applicable"] is False


def test_evaluate_transferred_appraisal_needs_transfer_request():
    result = evaluate_transferred_appraisal({"channel": "wholesale", "transfer_request_submitted": False})
    assert result["blocked"] is True
    assert any("transfer request" in s.lower() for s in result["steps"])


def test_evaluate_transferred_appraisal_borrower_pays_needs_management_approval():
    result = evaluate_transferred_appraisal({
        "channel": "wholesale",
        "transfer_request_submitted": True,
        "transfer_approved": True,
        "tpo_or_borrower_pays_at_closing": True,
        "management_approval_obtained": False,
    })
    assert result["blocked"] is True
    assert any("rm management" in s.lower() for s in result["steps"])


def test_evaluate_transferred_appraisal_fully_cleared():
    result = evaluate_transferred_appraisal({
        "channel": "wholesale",
        "transfer_request_submitted": True,
        "transfer_approved": True,
        "tpo_or_borrower_pays_at_closing": False,
    })
    assert result["blocked"] is False


def test_evaluate_transferred_appraisal_treats_string_booleans_correctly():
    """Regression test for core.booleans.as_bool(): the same string-boolean
    shape found in LG-docsOrch's real extraction output (see
    test_golden_fixtures.py) could just as easily arrive here from a web
    form/API payload. "False" must not be read as truthy."""
    result = evaluate_transferred_appraisal({
        "channel": "wholesale",
        "transfer_request_submitted": "True",
        "transfer_approved": "True",
        "tpo_or_borrower_pays_at_closing": "False",
    })
    assert result["blocked"] is False
    assert not any("rm management" in s.lower() for s in result["steps"])


def test_evaluate_transferred_appraisal_string_false_for_submitted_still_blocks():
    result = evaluate_transferred_appraisal({
        "channel": "wholesale",
        "transfer_request_submitted": "False",
    })
    assert result["blocked"] is True
    assert any("transfer request" in s.lower() for s in result["steps"])


# ---------------------------------------------------------------------------
# check_disaster_inspection_required
# ---------------------------------------------------------------------------

def test_check_disaster_inspection_required_when_disaster_after_appraisal():
    result = check_disaster_inspection_required("2026-01-01", "2026-02-01", inspection_condition_present=False)
    assert result["inspection_required"] is True
    assert result["flags"][0]["severity"] == "critical"


def test_check_disaster_inspection_not_required_when_disaster_before_appraisal():
    result = check_disaster_inspection_required("2026-03-01", "2026-01-01")
    assert result["inspection_required"] is False


def test_check_disaster_inspection_condition_already_present():
    result = check_disaster_inspection_required("2026-01-01", "2026-02-01", inspection_condition_present=True)
    assert result["flags"][0]["severity"] == "info"


# ---------------------------------------------------------------------------
# determine_secondary_appraisal_requirement
# ---------------------------------------------------------------------------

def test_secondary_appraisal_not_required_with_low_ssr_score():
    result = determine_secondary_appraisal_requirement(ssr_score=2.0, property_type="SFR")
    assert result["required"] is False


def test_secondary_appraisal_not_required_with_field_review():
    result = determine_secondary_appraisal_requirement(ssr_score=4.0, field_review_received=True)
    assert result["required"] is False


def test_secondary_appraisal_requires_cda_for_1_4_unit_high_ssr():
    result = determine_secondary_appraisal_requirement(ssr_score=3.5, property_type="SFR")
    assert result["required"] is True
    assert result["product"] == "CDA"


def test_secondary_appraisal_requires_bpo_for_mixed_use():
    result = determine_secondary_appraisal_requirement(ssr_score=3.5, property_type="Mixed Use")
    assert result["required"] is True
    assert result["product"] == "BPO"


def test_secondary_appraisal_requires_cda_when_no_ssr_available():
    result = determine_secondary_appraisal_requirement(ssr_score=None, property_type="Condo")
    assert result["required"] is True
    assert result["product"] == "CDA"
