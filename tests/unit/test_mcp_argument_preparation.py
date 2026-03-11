from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def _manager() -> MCPClientManager:
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="test-key",
        mcp_servers_json=(
            '[{"server_id":"yfinance","transport":"stdio","command":"uv","args":["run","server.py"]},'
            '{"server_id":"rss_search","transport":"stdio","command":"python3","args":["-m","rss"]}]'
        ),
        mcp_tool_allowlist_csv="yfinance:get_historical_stock_prices,yfinance:get_yahoo_finance_news,rss_search:rss_search",
        evidence_tool_map_json=(
            '{"price_path":"yfinance:get_historical_stock_prices","news":"rss_search:rss_search"}'
        ),
    )
    return MCPClientManager(settings=settings)


def test_prepare_tool_arguments_for_yfinance_history() -> None:
    manager = _manager()

    prepared = manager.prepare_tool_arguments(
        server_id="yfinance",
        tool_name="get_historical_stock_prices",
        arguments={
            "objective": "Check recent price path",
            "tickers": ["TSLA"],
            "time_window": "1 week",
            "query": {"interval": "1d"},
        },
    )

    assert prepared == {"ticker": "TSLA", "period": "5d", "interval": "1d"}


def test_prepare_tool_arguments_for_yfinance_news() -> None:
    manager = _manager()

    prepared = manager.prepare_tool_arguments(
        server_id="yfinance",
        tool_name="get_yahoo_finance_news",
        arguments={"tickers": ["NVDA"]},
    )

    assert prepared == {"ticker": "NVDA"}


def test_prepare_tool_arguments_for_rss_search() -> None:
    manager = _manager()

    prepared = manager.prepare_tool_arguments(
        server_id="rss_search",
        tool_name="rss_search",
        arguments={
            "objective": "Find recent earnings coverage",
            "tickers": ["AAPL"],
            "query": {"limit": 5},
        },
    )

    assert prepared == {
        "query": "Find recent earnings coverage AAPL",
        "limit": 5,
    }
