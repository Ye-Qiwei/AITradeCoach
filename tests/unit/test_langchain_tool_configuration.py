from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools, resolve_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_runtime_and_diagnostics_share_tool_names() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history,yfinance:yfinance_get_ticker_news",
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    runtime_tools = build_runtime_research_tools(
        settings=settings,
        mcp_manager=manager,
        runtime=MCPToolRuntime(),
    )
    runtime_names = {tool.name for tool in runtime_tools}

    diagnostics = resolve_research_tools(settings=settings, mcp_manager=manager)
    diagnostic_names = {item.agent_name for item in diagnostics if item.available}

    assert runtime_names == diagnostic_names
    assert "get_price_history" in runtime_names
    assert "search_news" in runtime_names
    assert "yahoo_japan_fund_history" in runtime_names


def test_default_map_has_no_placeholder_tools() -> None:
    settings = Settings(llm_provider_name="openai", openai_api_key="x")
    mapped = settings.evidence_tool_map()
    assert "filing" not in mapped
    assert "macro" not in mapped


def test_evidence_map_controls_runtime_backend() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        evidence_tool_map_json='{"news":"rss_search:rss_search"}',
        mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history,rss_search:rss_search",
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    diagnostics = resolve_research_tools(settings=settings, mcp_manager=manager)
    news = next(item for item in diagnostics if item.agent_name == "search_news")
    assert news.backend_name == "rss_search:rss_search"
    assert news.available is True
