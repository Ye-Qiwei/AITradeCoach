from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_mcp_diagnostics_exposes_raw_and_curated_mapping() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        mcp_servers_json='[{"server_id":"yfinance","transport":"stdio","command":"uvx","args":["yfmcp@latest"]}]',
        mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history,yfinance:yfinance_get_ticker_news,yfinance:yfinance_search",
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    info = manager.diagnostics()
    assert "yahoo_finance_price_history" in info["curated_tools"]
    assert "yfinance:yfinance_search" in info["raw_not_exposed_to_agent"]
