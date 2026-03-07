from __future__ import annotations

from ai_trading_coach.app.factory import build_orchestrator_modules
from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gateway import LangChainLLMGateway


def test_agents_share_same_langchain_gateway_instance() -> None:
    settings = Settings(
        openai_api_key="test",
        llm_provider_name="openai",
        llm_model_openai="gpt-4o-mini",
    )
    modules = build_orchestrator_modules(settings=settings, mcp_invoker=lambda *_: {"items": []})
    assert isinstance(modules.llm_gateway, LangChainLLMGateway)
    assert modules.parser_agent.gateway is modules.llm_gateway
    assert modules.reporter_agent.gateway is modules.llm_gateway
    assert modules.report_judge.gateway is modules.llm_gateway
