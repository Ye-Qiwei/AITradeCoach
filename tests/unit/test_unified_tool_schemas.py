from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def _tools(monkeypatch):
    from ai_trading_coach.modules.agent import web_tools as mod

    monkeypatch.setattr(mod, "_probe_local_playwright_runtime", lambda: (True, None))
    settings = Settings(llm_provider_name="openai", openai_api_key="x", brave_api_key="b", firecrawl_api_key="f")
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    return {t.name: t for t in build_runtime_research_tools(settings=settings, mcp_manager=manager, runtime=MCPToolRuntime())}


def test_required_fields_exposed(monkeypatch) -> None:
    tools = _tools(monkeypatch)
    assert "query" in tools["brave_search"].args
    assert "url" in tools["firecrawl_extract"].args
    assert "url" in tools["playwright_fetch"].args
    assert "ticker" in tools["get_price_history"].args
    assert "ticker" in tools["search_news"].args


def test_empty_calls_fail_at_entry(monkeypatch) -> None:
    import pytest

    tools = _tools(monkeypatch)
    for name in ["brave_search", "firecrawl_extract", "playwright_fetch", "get_price_history", "search_news"]:
        with pytest.raises(Exception):
            tools[name].invoke({})
