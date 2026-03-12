from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, build_agent_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_build_agent_tools_only_curated_surface() -> None:
    settings = Settings(llm_provider_name="openai", openai_api_key="x", mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history,yfinance:yfinance_get_ticker_news")
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    tools = build_agent_tools(mcp_manager=manager, runtime=MCPToolRuntime())
    names = {tool.name for tool in tools}
    assert "yahoo_finance_price_history" in names
    assert "yahoo_finance_ticker_news" in names
    assert "yahoo_japan_fund_history" in names
    assert "yfinance_get_price_history" not in names
