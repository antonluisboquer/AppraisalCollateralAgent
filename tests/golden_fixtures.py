"""Adapter for using LG-docsOrch's regression-test golden fixtures as
realistic input data for this repo's tools.

LG-docsOrch (a sibling Fintor repo) maintains a golden-fixture regression
suite at ``tests/regression/fixtures/<loan>/frozen_state.json`` -- a frozen
snapshot of real, previously-closed loan files (extracted document fields +
LOS fields) used there to catch regressions in its own extraction/flagging
pipeline.

That data is genuinely useful here too, just for a different purpose: it's
realistic *input* for our tools -- real currency-string formats
("$1,890.83"), real "M/D/YYYY" dates, nested contact/company dicts, fields
that are sometimes missing -- which is a much better stress test than
hand-written synthetic fixtures alone.

This module maps docsOrch's doc_fields/los_fields key names into the shape
this repo's ``tool_schema/*.json`` extraction schemas (and tool function
signatures) expect. It does NOT attempt to replicate docsOrch's own
golden_flags.json expected output: this repo implements different (and
additional) business rules than docsOrch does, so there's no meaningful
"expected flags" comparison to make across repos. See
``tests/test_golden_fixtures.py`` for how this is used.

If the sibling repo isn't present on disk (e.g. CI, or a machine with only
this repo checked out), ``available_fixtures()`` returns an empty list and
dependent tests skip themselves automatically.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# This repo and LG-docsOrch are expected to be sibling directories, e.g.:
#   .../Fintor/LG-AppraisalCollateralAgent   (this repo)
#   .../Fintor/LG-docsOrch                   (sibling)
_THIS_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_FIXTURES_DIR = (
    _THIS_REPO.parent / "LG-docsOrch" / "tests" / "regression" / "fixtures"
)
FIXTURES_DIR = Path(os.environ.get("DOCSORCH_FIXTURES_DIR", str(_DEFAULT_FIXTURES_DIR)))


def available_fixtures() -> List[str]:
    """Names of golden-fixture loan directories that have a frozen_state.json.
    Returns [] if the sibling repo/fixtures aren't present on disk."""
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in FIXTURES_DIR.iterdir()
        if p.is_dir() and (p / "frozen_state.json").is_file()
    )


def load_frozen_state(name: str) -> Dict[str, Any]:
    path = FIXTURES_DIR / name / "frozen_state.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap(entry: Any) -> Any:
    """doc_fields/los_fields entries look like
    {"value": ..., "source_document"|"field_id": ..., ...}; already-plain
    values pass through unchanged."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def doc_value(doc_fields: Dict[str, Any], *keys: str, source: Optional[str] = None) -> Any:
    """First present, non-None value among `keys` in docsOrch's doc_fields.

    `source`, if given, restricts the match to entries whose
    `source_document` equals it -- docsOrch sometimes reuses a field name
    (e.g. "property_address") across multiple document types, and we want
    the copy that actually came from the document we care about.
    """
    for key in keys:
        entry = doc_fields.get(key)
        if entry is None:
            continue
        if source and isinstance(entry, dict) and entry.get("source_document") != source:
            continue
        value = _unwrap(entry)
        if value is not None:
            return value
    return None


def los_value(los_fields: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        entry = los_fields.get(key)
        if entry is None:
            continue
        value = _unwrap(entry)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Field mappings: this repo's tool_schema field names <- docsOrch doc_fields
# ---------------------------------------------------------------------------

def build_appraisal_fields(doc_fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "appraised_value": doc_value(doc_fields, "appraised_value", source="Appraisal Report"),
        "property_address": doc_value(doc_fields, "property_address"),
        "property_type": doc_value(doc_fields, "property_type", source="Appraisal Report"),
        "appraisal_date": doc_value(doc_fields, "appraisal_date", "effective_date", source="Appraisal Report"),
        "flood_zone": doc_value(doc_fields, "flood_zone", source="Appraisal Report"),
        # These 3 are the exact fields whose real docsOrch shape ("False" as
        # a literal string, not a Python bool) surfaced the as_bool() bug --
        # see core/booleans.py and test_golden_fixtures.py.
        "appraisal_subject_to_completion": doc_value(doc_fields, "appraisal_subject_to_completion"),
        "appraisal_subject_to_repairs": doc_value(doc_fields, "appraisal_subject_to_repairs"),
        "appraisal_subject_to_conditions": doc_value(doc_fields, "appraisal_subject_to_conditions"),
    }


def build_loan_context(frozen_state: Dict[str, Any]) -> Dict[str, Any]:
    los_fields = frozen_state.get("los_fields", {}) or {}
    loan_profile = frozen_state.get("loan_profile", {}) or {}
    purpose = str(loan_profile.get("purpose") or "").lower()
    return {
        "loan_amount": los_value(los_fields, "loan_amount", "base_loan_amount"),
        "property_state": los_value(los_fields, "property_state") or loan_profile.get("state"),
        "estimated_closing_date": los_value(los_fields, "note_date", "closing_date"),
        "transaction_type": "refinance" if "refinance" in purpose else "purchase",
        "loan_type": loan_profile.get("loan_type"),
    }
