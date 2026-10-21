"""Document field extraction abstraction.

`extract_appraisal_data(document)` and the document-reading half of
`validate_title_commitment` / `validate_insurance_coverage` are schema-driven
LLM extractions in production (see LG-discOrch's extraction pipeline, which
runs a JSON-Schema draft-07 file against a PDF and returns a field->value
map — tools/tool_schema/*.json).

This module keeps that same contract (schema in, field dict out) but behind
a small `DocumentExtractor` protocol so:
  - tests/demos can use `StaticExtractor` (no API key, deterministic), and
  - production can use `OpenAIDocumentExtractor` (or swap in whatever
    extraction service a given deployment already has).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class DocumentExtractor(Protocol):
    def extract(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from `document_text` per `schema["properties"]`.

        Returns a flat {field_name: value} dict. Fields not found should be
        omitted or set to None, never fabricated.
        """
        ...


class StaticExtractor:
    """Returns a pre-canned field dict. For tests/demos — no network calls."""

    def __init__(self, canned_result: Dict[str, Any]):
        self._result = canned_result

    def extract(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._result)


def _schema_to_openai_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a draft-07 tool_schema file into an OpenAI structured-output schema.

    OpenAI's strict JSON-schema mode requires every property to be listed in
    "required" and additionalProperties=False, so we relax the draft-07
    "required" semantics (which only lists the verification_field) and instead
    allow every field to be nullable, letting the model return null for
    anything it can't find in the document.
    """
    props = schema.get("properties", {})
    out_props: Dict[str, Any] = {}
    for name, spec in props.items():
        json_type = spec.get("type", "string")
        if spec.get("type") == "array":
            out_props[name] = {**spec, "description": spec.get("description", "")}
        else:
            out_props[name] = {
                "type": [json_type, "null"],
                "description": spec.get("description", ""),
            }
    return {
        "type": "object",
        "properties": out_props,
        "required": list(out_props.keys()),
        "additionalProperties": False,
    }


class OpenAIDocumentExtractor:
    """Schema-guided extraction using the OpenAI API's structured outputs.

    Model defaults to the `APPRAISAL_TITLE_TAX_EXTRACTION_MODEL` env var, or
    "gpt-4o-mini" if unset. The architecture spec calls for "GPT-5 mini";
    swap the model string here once that identifier is available in your
    OpenAI account/region — nothing else about this class needs to change.
    """

    def __init__(self, model: str | None = None, client: Any = None):
        self.model = model or os.environ.get(
            "APPRAISAL_TITLE_TAX_EXTRACTION_MODEL", "gpt-4o-mini"
        )
        self._client = client  # lazily created on first use if not injected

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # local import: keep `openai` optional for tests

            self._client = OpenAI()
        return self._client

    def extract(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_client()
        doc_type = schema.get("document_type", "document")
        json_schema = _schema_to_openai_json_schema(schema)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You extract structured data from a {doc_type}. "
                        "Use ONLY the document text provided. If a field is not "
                        "present in the document, return null for it rather than "
                        "guessing. Follow each field's description precisely — "
                        "they describe exactly where in the document to look."
                    ),
                },
                {"role": "user", "content": document_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"{doc_type.replace(' ', '_').lower()}_fields",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        return json.loads(content)
