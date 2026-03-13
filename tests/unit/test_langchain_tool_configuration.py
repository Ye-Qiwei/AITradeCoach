from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools, resolve_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_unified_tool_builder_exposes_mcp_and_web_tools(monkeypatch) -> None:
    from ai_trading_coach.modules.agent import web_tools as mod

    monkeypatch.setattr(mod, "_probe_local_playwright_runtime", lambda: (True, None))
    settings = Settings(llm_provider_name="openai", openai_api_key="x", brave_api_key="b", firecrawl_api_key="f")
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    tools = build_runtime_research_tools(settings=settings, mcp_manager=manager, runtime=MCPToolRuntime())
    names = {tool.name for tool in tools}
    assert {"get_price_history", "search_news", "brave_search", "firecrawl_extract", "playwright_fetch", "yahoo_japan_fund_history"}.issubset(names)


def test_runtime_and_diagnostics_are_consistent() -> None:
    settings = Settings(llm_provider_name="openai", openai_api_key="x")
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    runtime_names = {tool.name for tool in build_runtime_research_tools(settings=settings, mcp_manager=manager, runtime=MCPToolRuntime())}
    diagnostics = {item.spec.name for item in resolve_research_tools(settings=settings, mcp_manager=manager) if item.available}
    assert runtime_names == diagnostics
