from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_web_tools_are_in_runtime_surface_when_available(monkeypatch) -> None:
    from ai_trading_coach.modules.agent import web_tools as mod

    monkeypatch.setattr(mod, "_probe_local_playwright_runtime", lambda: (True, None))
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        brave_api_key="b",
        firecrawl_api_key="f",
        mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history,yfinance:yfinance_get_ticker_news",
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    tools = build_runtime_research_tools(settings=settings, mcp_manager=manager, runtime=MCPToolRuntime())
    names = {tool.name for tool in tools}

    assert {"brave_search", "firecrawl_extract", "playwright_fetch"}.issubset(names)
