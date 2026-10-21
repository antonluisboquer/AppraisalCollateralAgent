"""Golden-fixture-driven tests: exercise this repo's tools against real,
previously-closed loan field data from LG-docsOrch's regression-test
fixtures, instead of only hand-written synthetic data.

Why: hand-written test fixtures tend to be "clean" -- round numbers, ISO
dates, no missing fields. Real extracted data is messier ("$1,890.83" not
1890.83, "3/20/2026" not "2026-03-20", nested contact dicts, fields that are
sometimes just missing) and is a much better test of whether our tools are
actually robust.

This is deliberately NOT a snapshot/expected-output comparison against
docsOrch's own golden_flags.json: this repo checks different (and
additional) business rules than docsOrch does, so there's no meaningful
cross-repo "expected flags" to assert against. Instead these tests assert:
  1. Every tool runs without raising on real-shaped field data.
  2. A handful of outcomes independently verifiable from the raw numbers
     in the fixture (e.g. a known LTV).

Skips automatically if the sibling LG-docsOrch repo / its fixtures aren't
present on disk. Set DOCSORCH_FIXTURES_DIR to point elsewhere if your
checkout layout differs.
"""
import pytest

from appraisal_collateral_agent.tools import appraisal_report_conditions, validate_property_value
from core.booleans import as_bool
from core.money import to_number

from tests.golden_fixtures import (
    available_fixtures,
    build_appraisal_fields,
    build_loan_context,
    load_frozen_state,
)

FIXTURES = available_fixtures()

pytestmark = pytest.mark.skipif(
    not FIXTURES,
    reason=(
        "LG-docsOrch golden fixtures not found on disk (sibling repo not checked "
        "out next to this one, or set DOCSORCH_FIXTURES_DIR to point at "
        "LG-docsOrch/tests/regression/fixtures)."
    ),
)


@pytest.fixture(params=FIXTURES)
def golden_loan(request):
    name = request.param
    state = load_frozen_state(name)
    doc_fields = state.get("doc_fields", {}) or {}
    return {
        "name": name,
        "appraisal": build_appraisal_fields(doc_fields),
        "loan_context": build_loan_context(state),
    }


def test_fixtures_were_discovered():
    # Sanity check on the adapter itself, not the tools -- if this starts
    # failing, DOCSORCH_FIXTURES_DIR / the sibling checkout has moved.
    assert len(FIXTURES) >= 5


def test_validate_property_value_on_every_golden_loan(golden_loan):
    appraised = golden_loan["appraisal"]["appraised_value"]
    loan_amount = golden_loan["loan_context"]["loan_amount"]
    result = validate_property_value(appraised, loan_amount)
    assert "flags" in result and result["flags"]
    if appraised and loan_amount:
        expected_ltv = to_number(loan_amount) / to_number(appraised)
        # validate_property_value rounds ltv for display; compare loosely.
        assert result["ltv"] == pytest.approx(expected_ltv, abs=1e-3)


def test_appraisal_report_conditions_on_every_golden_loan(golden_loan):
    """This is what actually surfaced the core.booleans.as_bool() bug:
    docsOrch's real appraisal_subject_to_completion/_repairs values are the
    literal string "False", not a Python bool. Before as_bool() existed,
    every one of these 8 real loans (all of which are "False"/"False")
    would have produced a false-positive "subject to completion/repairs"
    condition tag."""
    appraisal = golden_loan["appraisal"]
    raw_completion = appraisal.get("appraisal_subject_to_completion")
    raw_repairs = appraisal.get("appraisal_subject_to_repairs")
    if raw_completion is None and raw_repairs is None:
        pytest.skip(f"{golden_loan['name']}: no appraisal_subject_to_* fields in this fixture")

    tags = appraisal_report_conditions(appraisal)

    completion_true = as_bool(raw_completion) is True
    repairs_true = as_bool(raw_repairs) is True
    assert any("completion" in t for t in tags) == completion_true, (
        f"{golden_loan['name']}: raw value {raw_completion!r} (type {type(raw_completion).__name__}) "
        f"-> tags={tags}"
    )
    assert any("repairs" in t for t in tags) == repairs_true, (
        f"{golden_loan['name']}: raw value {raw_repairs!r} (type {type(raw_repairs).__name__}) "
        f"-> tags={tags}"
    )
