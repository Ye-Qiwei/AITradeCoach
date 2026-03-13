from __future__ import annotations

import warnings

import pytest

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.llm_output_contracts import JudgeVerdictContract
from ai_trading_coach.llm.gateway import LangChainLLMGateway


class _StructuredModelOK:
    def invoke(self, _messages):
        return {
            "passed": True,
            "reasons": ["ok"],
            "rewrite_instruction": "",
        }


class _StructuredModelRaises:
    def invoke(self, _messages):
        raise ValueError("structured boom")


class _ModelOK:
    def with_structured_output(self, _schema):
        return _StructuredModelOK()


class _ModelRaises:
    def with_structured_output(self, _schema):
        return _StructuredModelRaises()


class _ModelWarns:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        warnings.warn(
            "Pydantic serializer warnings: Something field_name='parsed' ParserOutputContract",
            UserWarning,
        )
        warnings.warn("keep me", UserWarning)
        return {
            "passed": True,
            "reasons": ["ok"],
            "rewrite_instruction": "",
        }


class _ModelStructuredFailsTextSucceeds:
    def with_structured_output(self, _schema):
        return _StructuredModelRaises()

    def invoke(self, _messages):
        class _Resp:
            content = """```json
            {"passed": true, "reasons": ["ok"], "rewrite_instruction": ""}
            ```"""

        return _Resp()


def _settings() -> Settings:
    return Settings(llm_provider_name="openai", openai_api_key="test-key", llm_model="gpt-4o-mini")


def test_invoke_structured_builds_trace_with_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelOK())
    gateway = LangChainLLMGateway(settings=_settings())

    result, trace = gateway.invoke_structured(
        schema=JudgeVerdictContract,
        messages=[{"role": "user", "content": "x"}],
        purpose=ModelCallPurpose.COGNITION_EVALUATION,
        prompt_version="report_judging.v1",
        input_summary="sample",
    )

    assert result.passed is True
    assert trace.call_id
    assert trace.provider == "openai"
    assert trace.model_name == "gpt-4o-mini"
    assert trace.purpose == ModelCallPurpose.COGNITION_EVALUATION


def test_invoke_structured_failure_raises_contextual_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelRaises())
    gateway = LangChainLLMGateway(settings=_settings())

    with pytest.raises(RuntimeError) as exc:
        gateway.invoke_structured(
            schema=JudgeVerdictContract,
            messages=[{"role": "user", "content": "x"}],
            purpose=ModelCallPurpose.COGNITION_EVALUATION,
            prompt_version="report_judging.v1",
            input_summary="sample",
        )

    text = str(exc.value)
    assert "schema=JudgeVerdictContract" in text
    assert "purpose=cognition_evaluation" in text
    assert "prompt_version=report_judging.v1" in text


def test_structured_warning_filter_is_narrow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelWarns())
    gateway = LangChainLLMGateway(settings=_settings())

    shown: list[str] = []

    def _showwarning(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        shown.append(str(message))

    monkeypatch.setattr(warnings, "showwarning", _showwarning)

    gateway.invoke_structured(
        schema=JudgeVerdictContract,
        messages=[{"role": "user", "content": "x"}],
        purpose=ModelCallPurpose.COGNITION_EVALUATION,
        prompt_version="report_judging.v1",
        input_summary="sample",
    )

    assert "keep me" in shown
    assert all("field_name='parsed'" not in item for item in shown)


def test_invoke_structured_falls_back_to_text_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_trading_coach.llm.gateway.build_langchain_chat_model",
        lambda **_: _ModelStructuredFailsTextSucceeds(),
    )
    gateway = LangChainLLMGateway(settings=_settings())

    result, trace = gateway.invoke_structured(
        schema=JudgeVerdictContract,
        messages=[{"role": "user", "content": "x"}],
        purpose=ModelCallPurpose.COGNITION_EVALUATION,
        prompt_version="report_judging.v1",
        input_summary="sample",
    )

    assert result.passed is True
    assert result.reasons == ["ok"]
    assert "fallback=text_json" in trace.output_summary
