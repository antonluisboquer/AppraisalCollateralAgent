# Appraisal & Collateral Agent

Standalone implementation of the Appraisal & Collateral Agent spec documented
in `LG-discOrch/docs/APPRAISAL_TITLE_INSURANCE_TAX_AGENTS_ARCHITECTURE.md`
(a sibling repo's architecture doc — see that file for the full spec text
and the RM Job Aid cross-reference this build is based on).

This repo was split out of the original combined `LG-AppraisalTitleTaxAgents`
repo so the Appraisal/Collateral Agent and the
[Title/Insurance/Taxes Agent](../LG-TitleInsuranceTaxAgent) can be developed,
tested, and deployed independently. The split is clean: this agent has no
import dependency on the other agent, and never did.

This repo is **standalone** — it does not import from or modify
`LG-discOrch`, `LG-docsOrch`, or `lg-RMOrch`. Where it ports business logic
verified to already work in one of those repos, the source is cited directly
above the code (grep for `lg-RMOrch`, `LG-docsOrch`, or `LG-discOrch` in the
`core/` and agent modules).

## What's implemented

Built as a [LangGraph](https://langchain-ai.github.io/langgraph/)
tool-calling react agent, invoked as a step in a sequential pipeline (not
event-driven — see **Deferred** below), with a configurable LLM
("GPT-5 mini" per spec; defaults to `gpt-4o-mini`, swap via env var once that
identifier is available).

**Status**: every job-aid item that could be modeled as a pure, testable
function is implemented — 88 unit tests (86 synthetic + golden-fixture
tests, 2 skipped when the sibling `LG-docsOrch` checkout isn't present), all
deterministic, no API key required.

| Spec tool | Status | Notes |
|---|---|---|
| `extract_appraisal_data(document)` | ✅ | Schema-driven LLM extraction; also pulls UAD condition/quality ratings (A1) and a comparable-sales array (A2) |
| `get_property_details(loan_guid)` | ✅ | Reads Encompass fields 356/1041/1894 via a pluggable `LOSClient` |
| `validate_property_value(appraised_value, loan_amount)` | ✅ | Deterministic LTV calc + "loan > appraised value" flag |
| `validate_appraisal_expiration` | ✅ | J-A1 — flags if the appraisal goes stale (120-day window) before closing |
| `cross_check_file_contacts_appraiser` | ✅ | J-A3 — name/license/company match between File Contacts and the appraisal report |
| `determine_resubmission_priority` | ✅ | J-A4 — an appraisal condition always forces an immediate UW resubmit |
| `evaluate_transferred_appraisal` | ✅ | J-A2 — transfer-request + RM-management payment-approval gate workflow |
| `check_disaster_inspection_required` | ✅ | FEMA-declaration-vs-appraisal-date check (RM Job Aid p.21) |
| `determine_secondary_appraisal_requirement` | ✅ | CDA/BPO/Field-Review/2nd-appraisal logic from SSR score + property type (RM Job Aid p.21) |
| *(supporting)* `compare_appraisal_to_encompass` | ✅ | Generic flag engine ported from `determine_flag_action()` |
| *(supporting)* `generate_appraisal_summary` | ✅ | Lightweight RM-facing markdown summary |

## What's intentionally deferred (backlog)

| Item | Why deferred |
|---|---|
| **A5** Event-driven triggers | All 3 source repos run as sequential batch orchestrators; this repo matches that pattern. Scoping a real event/webhook system is a cross-cutting architecture decision, not a tool |
| **Appraisal corrections request** (p.20) | A one-line email to `appraisaldesk@nqmf.com` — an external action, not a document validation |

### Still worth a second pass (modeled, but with simplifying assumptions)
- The agent extracts UAD condition/quality ratings and comparable sales fields when present, but does not yet apply any pass/fail business rules to them beyond extraction.

## Design notes

- **No duplicated Encompass auth.** `core/los_client.py` defines an
  `LOSClient` protocol with a `MockLOSClient` (tests/demos) and a
  `CallableLOSClient` adapter — wire the adapter around whatever
  already-authenticated Encompass reader your deployment uses (e.g.
  `LG-discOrch/tools/encompass_client.py::get_encompass_client`) rather than
  reimplementing OAuth here.
- **No duplicated extraction pipeline.** `core/extraction.py` defines a
  `DocumentExtractor` protocol with a `StaticExtractor` (tests/demos, no API
  key needed) and an `OpenAIDocumentExtractor` (production, schema-guided
  structured-output extraction against `tool_schema/appraisal_report.json`).
- **Same flag vocabulary as LG-discOrch.** `core/flagging.py` ports
  `determine_flag_action()` (`ignore` / `overwrite` / `ptf` / `critical`) so
  output from this repo merges cleanly into an existing orchestrator's
  review-result shape. One bug fix vs. the original: a `hard_stop` field
  whose values are numerically within tolerance now correctly returns
  `ignore` instead of falling through to `overwrite`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY to run the LLM-orchestrated agent

# Unit tests (88 tests, all deterministic — no API key required)
pytest tests/ -v

# Deterministic pipeline demo (no API key required)
python3 examples/run_appraisal_agent_demo.py
```

### Golden-fixture tests (optional, against real loan data)

`tests/test_golden_fixtures.py` replays real, previously-closed loan field
data from LG-docsOrch's own regression-test fixtures
(`LG-docsOrch/tests/regression/fixtures/*/frozen_state.json`) through this
repo's tools, via the field-name adapter in `tests/golden_fixtures.py`. This
catches things hand-written fixtures don't: real currency-string formats,
real date formats, and fields that are sometimes just missing.

It is NOT a comparison against LG-docsOrch's own `golden_flags.json` —
this repo checks different (and additional) rules than docsOrch does, so
there's no meaningful cross-repo "expected output" to assert against.

**Real bug this caught: string-shaped booleans.** 8 of 9 golden fixtures
have `appraisal_subject_to_completion`/`_repairs` extracted as the *literal
string* `"False"`, not a Python `bool`. `bool("False")` is `True` in
Python, so every `if fields.get("some_flag"):` check was silently broken
for that shape — a document explicitly saying "not subject to repairs"
would have been read as "is subject to repairs," forcing a false-positive
immediate UW resubmit. Replaying the real fixtures confirmed it concretely:
**5 of 7 golden loans with this field would have been false positives**
before the fix. `core/booleans.py::as_bool()` is the fix — it's used at
every boolean-field call site in this agent (`appraisal_subject_to_*`,
`transfer_request_submitted`, `transfer_approved`,
`tpo_or_borrower_pays_at_closing`, `management_approval_obtained`), each
with its own regression test proving the string-`"False"` case now
resolves correctly.

Automatically skipped if `LG-docsOrch` isn't checked out as a sibling
directory. Point `DOCSORCH_FIXTURES_DIR` elsewhere if your layout differs:

```bash
DOCSORCH_FIXTURES_DIR=/path/to/LG-docsOrch/tests/regression/fixtures pytest tests/test_golden_fixtures.py -v
```

To run the full LLM-orchestrated agent (a tool-calling react agent that
decides which tool to call and summarizes the results itself), set
`OPENAI_API_KEY` and see the commented-out snippet at the bottom of
`examples/run_appraisal_agent_demo.py`, or:

```python
from core.los_client import MockLOSClient
from appraisal_collateral_agent import build_appraisal_collateral_agent

agent = build_appraisal_collateral_agent(los=MockLOSClient({...}))
result = agent.invoke({"messages": [("user", "Review the appraisal for loan <guid>. Appraisal text: ...")]})
print(result["messages"][-1].content)
```

## Repo layout

```
core/                           # agent-scoped infrastructure (not shared with the title/tax agent)
  money.py, dates.py            # parsing helpers
  models.py                     # Flag dataclass (ignore/overwrite/ptf/critical)
  schema_loader.py              # loads tool_schema/*.json
  booleans.py                   # as_bool() -- normalizes "True"/"False" strings from real extraction data
  los_client.py                 # LOSClient protocol + Mock/Callable implementations
  extraction.py                 # DocumentExtractor protocol + Static/OpenAI implementations
  flagging.py                   # determine_flag_action() port + field-ID registries

tool_schema/
  appraisal_report.json         # extraction schema, incl. condition/quality ratings, comparable_sales (A1/A2)

appraisal_collateral_agent/
  tools.py                      # pure functions (unit-testable without an LLM)
  agent.py                      # LangGraph react-agent wrapper

tests/                          # 88 tests, all deterministic
examples/                       # runnable demo script
```

## Relationship to the Title/Insurance/Taxes Agent

This repo and [`LG-TitleInsuranceTaxAgent`](../LG-TitleInsuranceTaxAgent) were
split from a single combined repo (`LG-AppraisalTitleTaxAgents`). Each repo
carries its own full copy of the shared `core/` infrastructure (`booleans`,
`dates`, `models`, `money`, `schema_loader`) rather than depending on a
shared package — there was no existing shared-package registry in this org
to depend on, and the two agents' business logic is otherwise fully
independent. If `core/` grows substantially shared surface area in the
future, consider extracting it into its own installable package rather than
hand-syncing bug fixes (like the `as_bool()` fix above) across both repos.
