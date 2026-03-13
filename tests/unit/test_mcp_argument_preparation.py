from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_get_tool_ref_defaults() -> None:
    manager = MCPClientManager(settings=Settings(llm_provider_name="openai", openai_api_key="x"), invoker=lambda *_: {})
    price_ref, _ = manager.get_tool_ref("get_price_history")
    news_ref, _ = manager.get_tool_ref("search_news")
    assert price_ref is not None and price_ref.key == "yfinance:yfinance_get_price_history"
    assert news_ref is not None and news_ref.key == "yfinance:yfinance_get_ticker_news"


def test_get_tool_ref_unknown() -> None:
    manager = MCPClientManager(settings=Settings(llm_provider_name="openai", openai_api_key="x"), invoker=lambda *_: {})
    ref, reason = manager.get_tool_ref("unknown")
    assert ref is None
    assert "unknown" in (reason or "")
