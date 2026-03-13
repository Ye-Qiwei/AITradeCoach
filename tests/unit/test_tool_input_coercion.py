from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_prepare_tool_arguments_for_new_yfinance_names() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        mcp_servers=[{"server_id":"yfinance","transport":"stdio","command":"uvx","args":["yfmcp@latest"]}],
        mcp_tool_allowlist_csv="yfinance:yfinance_get_price_history",
        evidence_tool_map_json='{"price_path":"yfinance:yfinance_get_price_history"}',
    )
    manager = MCPClientManager(settings=settings)
    args = manager.prepare_tool_arguments(server_id="yfinance", tool_name="yfinance_get_price_history", arguments={"tickers": ["AAPL"], "query": {"interval": "1d"}})
    assert args["ticker"] == "AAPL"
