from __future__ import annotations

import pytest

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.llm.gateway import LangChainLLMGateway


class _ModelTextOK:
    def invoke(self, _messages):
        class _Resp:
            content = "hello"

        return _Resp()


class _ModelRaises:
    def invoke(self, _messages):
        raise ValueError("boom")


class _ModelJsonText:
    def invoke(self, _messages):
        class _Resp:
            content = '```json\n{"k":1}\n```'

        return _Resp()


def _settings() -> Settings:
    return Settings(llm_provider_name="openai", openai_api_key="test-key", llm_model="gpt-4o-mini")


def test_invoke_text_builds_trace_with_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelTextOK())
    gateway = LangChainLLMGateway(settings=_settings())

    result, trace = gateway.invoke_text(
        messages=[{"role": "user", "content": "x"}],
        purpose=ModelCallPurpose.COGNITION_EVALUATION,
        prompt_version="report_judging.v1",
        input_summary="sample",
    )

    assert result == "hello"
    assert trace.call_id
    assert trace.provider == "openai"
    assert trace.model_name == "gpt-4o-mini"
    assert trace.purpose == ModelCallPurpose.COGNITION_EVALUATION


def test_invoke_text_failure_raises_contextual_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelRaises())
    gateway = LangChainLLMGateway(settings=_settings())

    with pytest.raises(RuntimeError) as exc:
        gateway.invoke_text(
            messages=[{"role": "user", "content": "x"}],
            purpose=ModelCallPurpose.COGNITION_EVALUATION,
            prompt_version="report_judging.v1",
            input_summary="sample",
        )

    text = str(exc.value)
    assert "purpose=cognition_evaluation" in text
    assert "prompt_version=report_judging.v1" in text
    assert "ValueError: boom" in text


def test_extract_json_payload_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _ModelJsonText())
    gateway = LangChainLLMGateway(settings=_settings())
    content, _ = gateway.invoke_text(
        messages=[{"role": "user", "content": "x"}],
        purpose=ModelCallPurpose.REPORT_GENERATION,
        prompt_version="report_generation.v3",
        input_summary="sample",
    )
    parsed = gateway._extract_json_payload(content)
    assert parsed == {"k": 1}
