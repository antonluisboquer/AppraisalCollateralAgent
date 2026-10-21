from datetime import date

from core.booleans import as_bool
from core.dates import days_between, months_between, parse_date
from core.flagging import (
    APPRAISAL_FIELD_CHECKS,
    FieldCheckConfig,
    compare_fields_to_encompass,
    determine_flag_action,
)
from core.los_client import CallableLOSClient, MockLOSClient
from core.money import format_currency, to_number
from core.schema_loader import load_schema


# ---------------------------------------------------------------------------
# money.py
# ---------------------------------------------------------------------------

def test_to_number_handles_currency_strings():
    assert to_number("$1,234.56") == 1234.56


def test_to_number_handles_percent_strings():
    assert to_number("5%") == 5.0


def test_to_number_handles_plain_numbers():
    assert to_number(42) == 42.0
    assert to_number(42.5) == 42.5


def test_to_number_returns_none_for_missing_or_garbage():
    assert to_number(None) is None
    assert to_number("") is None
    assert to_number("n/a") is None


def test_format_currency():
    assert format_currency(1234.5) == "$1,234.50"
    assert format_currency(None) == "n/a"


# ---------------------------------------------------------------------------
# booleans.py
#
# The string-"False" case is not hypothetical: LG-docsOrch's real extraction
# output stores appraisal_subject_to_completion/_repairs as the literal
# string "False", not a Python bool -- confirmed by replaying its golden
# fixtures (see test_golden_fixtures.py). bool("False") is True in Python,
# which was silently forcing false-positive "subject to repairs" conditions
# before this helper existed.
# ---------------------------------------------------------------------------

def test_as_bool_passes_through_real_bools():
    assert as_bool(True) is True
    assert as_bool(False) is False


def test_as_bool_parses_the_literal_string_false_as_false():
    assert as_bool("False") is False
    assert as_bool("false") is False
    assert as_bool("FALSE") is False


def test_as_bool_parses_the_literal_string_true_as_true():
    assert as_bool("True") is True
    assert as_bool("true") is True


def test_as_bool_parses_yes_no():
    assert as_bool("yes") is True
    assert as_bool("no") is False


def test_as_bool_returns_none_for_missing():
    assert as_bool(None) is None
    assert as_bool("") is None


def test_as_bool_falls_back_to_truthiness_for_other_strings():
    # A free-text value that isn't a recognized true/false spelling is still
    # treated as truthy, matching the pre-existing (pre-bug-fix) behavior for
    # genuinely non-boolean-shaped data.
    assert as_bool("some narrative text") is True


def test_as_bool_handles_non_string_non_bool_values():
    assert as_bool(1) is True
    assert as_bool(0) is False


# ---------------------------------------------------------------------------
# dates.py
# ---------------------------------------------------------------------------

def test_parse_date_handles_common_formats():
    assert parse_date("2026-01-15") == date(2026, 1, 15)
    assert parse_date("01/15/2026") == date(2026, 1, 15)
    assert parse_date(date(2026, 1, 15)) == date(2026, 1, 15)


def test_parse_date_returns_none_for_garbage():
    assert parse_date(None) is None
    assert parse_date("not a date") is None


def test_days_between():
    assert days_between(date(2026, 1, 1), date(2026, 1, 31)) == 30
    assert days_between(date(2026, 1, 31), date(2026, 1, 1)) == 30  # order-independent


def test_months_between_roughly_twelve():
    term = months_between(date(2026, 1, 1), date(2027, 1, 1))
    assert 11.5 <= term <= 12.5


# ---------------------------------------------------------------------------
# schema_loader.py
# ---------------------------------------------------------------------------

def test_load_schema_appraisal_report():
    schema = load_schema("appraisal_report")
    assert schema["document_type"] == "Appraisal Report"
    assert "appraised_value" in schema["properties"]


# ---------------------------------------------------------------------------
# flagging.py
# ---------------------------------------------------------------------------

def test_determine_flag_action_ignore_when_values_match():
    config = FieldCheckConfig(field_id="1", field_name="Test")
    assert determine_flag_action(config, "same", "same") == "ignore"


def test_determine_flag_action_hard_stop_outside_tolerance_is_critical():
    config = FieldCheckConfig(field_id="356", field_name="Appraised Value",
                               check_type="hard_stop", tolerance=1.0, hard_stop=True)
    assert determine_flag_action(config, 500_000, 490_000) == "critical"


def test_determine_flag_action_hard_stop_within_tolerance_is_ignore():
    config = FieldCheckConfig(field_id="356", field_name="Appraised Value",
                               check_type="hard_stop", tolerance=1.0, hard_stop=True)
    assert determine_flag_action(config, 500_000.0, 500_000.4) == "ignore"


def test_determine_flag_action_overwrite_when_values_differ_no_hard_stop():
    config = FieldCheckConfig(field_id="1894", field_name="Parcel Number")
    assert determine_flag_action(config, "APN-123", "APN-999") == "overwrite"


def test_determine_flag_action_alert_is_ptf():
    config = FieldCheckConfig(field_id="x", field_name="Name", alert=True)
    assert determine_flag_action(config, "Jane Doe", "John Doe") == "ptf"


def test_compare_fields_to_encompass_flags_mismatches_only():
    extracted = {"356": 500_000, "1041": "SFR", "1894": "APN-1"}
    encompass = {"356": 500_000, "1041": "Condo", "1894": "APN-1"}
    flags = compare_fields_to_encompass(extracted, encompass, APPRAISAL_FIELD_CHECKS)
    assert len(flags) == 1
    assert flags[0]["field"] == "Subject Property Type"
    assert flags[0]["action"] == "overwrite"


# ---------------------------------------------------------------------------
# los_client.py
# ---------------------------------------------------------------------------

def test_mock_los_client_returns_fixtures():
    los = MockLOSClient({"guid-1": {"356": 500_000, "1041": "SFR"}})
    result = los.get_fields("guid-1", ["356", "1041", "1894"])
    assert result == {"356": 500_000, "1041": "SFR", "1894": None}


def test_mock_los_client_unknown_loan_returns_nones():
    los = MockLOSClient({})
    result = los.get_fields("unknown-guid", ["356"])
    assert result == {"356": None}


def test_callable_los_client_delegates():
    calls = []

    def reader(loan_guid, field_ids):
        calls.append((loan_guid, field_ids))
        return {fid: "value" for fid in field_ids}

    los = CallableLOSClient(reader)
    result = los.get_fields("guid-1", ["356"])
    assert result == {"356": "value"}
    assert calls == [("guid-1", ["356"])]
