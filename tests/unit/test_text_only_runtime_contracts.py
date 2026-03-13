from __future__ import annotations

from pathlib import Path

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gateway import LangChainLLMGateway


def test_gateway_is_text_only(monkeypatch):
    class _Model:
        def invoke(self, _messages):
            class _R:
                content = "ok"

            return _R()

    monkeypatch.setattr("ai_trading_coach.llm.gateway.build_langchain_chat_model", lambda **_: _Model())
    gateway = LangChainLLMGateway(settings=Settings(llm_provider_name="openai", openai_api_key="k", llm_model="gpt-4o-mini"))
    assert not hasattr(gateway, "invoke_structured")


def test_debug_execute_collection_path_no_parser_contract_reference() -> None:
    script = Path("scripts/debug_langgraph_nodes.py").read_text(encoding="utf-8")
    assert "ParserOutputContract" not in script
