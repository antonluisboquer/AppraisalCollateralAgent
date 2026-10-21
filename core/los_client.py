"""LOS (Loan Origination System / Encompass) field-read abstraction.

This repo does NOT reimplement Encompass authentication — LG-discOrch already
has a production-grade client (`tools/encompass_client.py::get_encompass_client`,
using EncompassConnect with batched reads of 50 field IDs at a time; see
tools/document_review.py:2119-2138). Duplicating that auth flow here would
just create a second thing to keep in sync.

Instead, both agents' tools depend on this tiny `LOSClient` protocol. In
production, wire in `CallableLOSClient` around whatever function your
deployment already uses to batch-read Encompass fields (e.g. a thin wrapper
around LG-discOrch's `get_encompass_client(...).get_field(loan_id, field_ids)`).
For local dev/tests/demos, use `MockLOSClient`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LOSClient(Protocol):
    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        """Return {field_id: value} for the requested Encompass field IDs."""
        ...


class MockLOSClient:
    """In-memory LOS client for tests/demos/local dev.

    fixtures: {loan_guid: {field_id: value}}
    """

    def __init__(self, fixtures: Dict[str, Dict[str, Any]]):
        self._fixtures = fixtures

    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        loan_fields = self._fixtures.get(loan_guid, {})
        return {fid: loan_fields.get(fid) for fid in field_ids}

    def set_fields(self, loan_guid: str, values: Dict[str, Any]) -> None:
        self._fixtures.setdefault(loan_guid, {}).update(values)


class CallableLOSClient:
    """Adapter around an existing batched field-read function.

    Example (production wiring inside LG-discOrch or a service that already
    has an authenticated EncompassConnect client):

        from tools.encompass_client import get_encompass_client

        def _read(loan_guid, field_ids):
            client = get_encompass_client()
            out = {}
            for i in range(0, len(field_ids), 50):
                batch = field_ids[i:i + 50]
                out.update(client.get_field(loan_guid, batch))
            return out

        los = CallableLOSClient(_read)
    """

    def __init__(self, read_fields_fn: Callable[[str, List[str]], Dict[str, Any]]):
        self._read_fields_fn = read_fields_fn

    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        return self._read_fields_fn(loan_guid, field_ids)
