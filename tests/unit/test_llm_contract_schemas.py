from __future__ import annotations

from typing import Any

from ai_trading_coach.domain.llm_output_contracts import (
    JudgeVerdictContract,
    ParserOutputContract,
    ReporterOutputContract,
    ResearchSynthesisOutputContract,
)
from ai_trading_coach.domain.schema_validation import validate_strict_llm_schema


def _walk_no_anyof_or_extensions(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_no_anyof_or_extensions(item)
        return
    if not isinstance(node, dict):
        return
    assert "anyOf" not in node
    assert "extensions" not in node
    for key, value in node.items():
        assert key != "extensions"
        _walk_no_anyof_or_extensions(value)


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if not ref:
        return node
    key = ref.split("/")[-1]
    return defs[key]


def _walk_objects(schema: dict[str, Any], node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_objects(schema, item)
        return
    if not isinstance(node, dict):
        return

    resolved = _resolve(node, schema.get("$defs", {}))
    properties = resolved.get("properties")
    if resolved.get("type") == "object" or isinstance(properties, dict):
        assert resolved.get("additionalProperties") is False
        assert isinstance(properties, dict)
        required = resolved.get("required")
        assert isinstance(required, list)
        assert set(required) == set(properties.keys())

    for value in resolved.values():
        _walk_objects(schema, value)


def test_contract_schemas_are_strict_and_validator_accepts_them() -> None:
    for model_cls in (
        ParserOutputContract,
        ResearchSynthesisOutputContract,
        ReporterOutputContract,
        JudgeVerdictContract,
    ):
        schema = model_cls.model_json_schema()
        assert schema.get("type") == "object"
        _walk_objects(schema, schema)
        _walk_no_anyof_or_extensions(schema)
        validate_strict_llm_schema(model_cls)


def test_window_fields_are_enums_not_plain_strings() -> None:
    parser_schema = ParserOutputContract.model_json_schema()
    judgement_item = parser_schema["$defs"]["JudgementItemContract"]
    proposed = judgement_item["properties"]["proposed_evaluation_window"]
    assert "enum" in proposed
    assert proposed["type"] == "string"

    reporter_schema = ReporterOutputContract.model_json_schema()
    daily_feedback = reporter_schema["$defs"]["DailyJudgementFeedbackContract"]
    evaluation_window = daily_feedback["properties"]["evaluation_window"]
    assert "enum" in evaluation_window
    assert evaluation_window["type"] == "string"
