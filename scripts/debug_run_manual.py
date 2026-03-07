from __future__ import annotations

from datetime import date
from pathlib import Path
import json

from ai_trading_coach.app.factory import build_pipeline_orchestrator
from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest


def fake_mcp_invoker(server_id: str, tool_name: str, arguments: dict):
    """本地调试专用。绕过真实 MCP server，让每个 research tool 都返回可被 normalize 的结果。"""
    tickers = arguments.get("tickers") or ["SPY"]
    ticker = tickers[0]
    objective = arguments.get("objective", "") or "debug objective"

    if tool_name == "price_history":
        return {
            "items": [
                {
                    "title": f"{ticker} recent price path",
                    "uri": "local://debug/price_history",
                    "published_at": "2026-03-06T00:00:00+00:00",
                    "summary": f"Mock price evidence for {ticker}. Objective: {objective}",
                    "ticker": ticker,
                    "close": 100.5,
                    "change_pct": 1.8,
                    "date": "2026-03-06",
                }
            ]
        }

    if tool_name == "rss_search":
        return {
            "items": [
                {
                    "title": f"News about {ticker}",
                    "uri": "local://debug/news",
                    "published_at": "2026-03-06T01:00:00+00:00",
                    "summary": f"Mock news evidence related to {ticker}. Objective: {objective}",
                    "ticker": ticker,
                }
            ]
        }

    if tool_name == "list_filings":
        return {
            "items": [
                {
                    "title": f"{ticker} latest filing",
                    "uri": "local://debug/filing",
                    "published_at": "2026-03-05T22:00:00+00:00",
                    "summary": f"Mock filing evidence for {ticker}. Objective: {objective}",
                    "ticker": ticker,
                    "filing_type": "10-K",
                    "company": ticker,
                }
            ]
        }

    if tool_name == "series_observations":
        return {
            "items": [
                {
                    "title": "Macro series observation",
                    "uri": "local://debug/macro",
                    "published_at": "2026-03-06T02:00:00+00:00",
                    "summary": f"Mock macro evidence. Objective: {objective}",
                    "series_id": "DGS10",
                    "value": 4.2,
                    "unit": "percent",
                    "date": "2026-03-06",
                }
            ]
        }

    return {"items": []}


def main() -> None:
    settings = get_settings()
    settings.validate_llm_or_raise()

    orchestrator = build_pipeline_orchestrator(
        settings=settings,
        mcp_invoker=fake_mcp_invoker,
    )

    log_file = Path("examples/logs/daily_log_sample.md")
    text = log_file.read_text(encoding="utf-8")

    request = ReviewRunRequest(
        run_id="manual_demo_user_2026-03-07",
        user_id="demo_user",
        run_date=date(2026, 3, 7),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=text,
        options={"dry_run": True, "debug_mode": True},
    )

    result = orchestrator.run(request)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()