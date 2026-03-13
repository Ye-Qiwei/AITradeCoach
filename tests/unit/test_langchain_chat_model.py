from __future__ import annotations

import sys
from types import SimpleNamespace

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.langchain_chat_model import build_langchain_chat_model


def test_build_langchain_chat_model_omits_temperature_for_gpt5(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=_FakeChatOpenAI))

    model = build_langchain_chat_model(
        Settings(llm_provider_name="openai", openai_api_key="test-key", llm_model="gpt-5-mini-2025-08-07")
    )

    assert isinstance(model, _FakeChatOpenAI)
    assert captured["model"] == "gpt-5-mini-2025-08-07"
    assert "temperature" not in captured


def test_build_langchain_chat_model_keeps_temperature_for_gpt4o(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=_FakeChatOpenAI))

    model = build_langchain_chat_model(
        Settings(llm_provider_name="openai", openai_api_key="test-key", llm_model="gpt-4o-mini")
    )

    assert isinstance(model, _FakeChatOpenAI)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0
