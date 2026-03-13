from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_get_tool_ref_for_price_history() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        mcp_servers=[{"server_id":"yfinance","transport":"stdio","command":"uvx","args":["yfmcp@latest"]}],
    )
    manager = MCPClientManager(settings=settings)
    ref, reason = manager.get_tool_ref("get_price_history")
    assert reason is None
    assert ref is not None
    assert ref.tool_name == "yfinance_get_price_history"
