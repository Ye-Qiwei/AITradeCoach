from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_mcp_diagnostics_exposes_curated_mapping() -> None:
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        mcp_servers=[{"server_id":"yfinance","transport":"stdio","command":"uvx","args":["yfmcp@latest"]}],
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})
    info = manager.diagnostics()
    assert "get_price_history" in info["curated_tools"]
    assert "configured_servers" in info
