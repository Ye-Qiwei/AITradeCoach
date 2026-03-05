from __future__ import annotations

from datetime import date
from pathlib import Path

from ai_trading_coach.domain.contracts import LogIntakeInput
from ai_trading_coach.domain.enums import AssetType
from ai_trading_coach.modules.intake.service import MarkdownLogIntakeCanonicalizer


def test_markdown_log_intake_parses_core_fields() -> None:
    text = Path("examples/logs/daily_log_sample.md").read_text(encoding="utf-8")
    svc = MarkdownLogIntakeCanonicalizer()

    output = svc.ingest(
        LogIntakeInput(
            user_id="u1",
            run_date=date(2026, 3, 5),
            raw_log_text=text,
        )
    )

    normalized = output.normalized
    assert normalized.log_date == date(2026, 3, 4)
    assert normalized.traded_tickers == ["9660.HK"]
    assert "4901.T" in normalized.mentioned_tickers
    assert normalized.trade_events
    assert normalized.trade_events[0].ticker == "9660.HK"
    assert normalized.ai_directives


def test_log_intake_parses_multi_market_trade_records_and_asset_types() -> None:
    text = """---
date: 2026-03-05
---

# 交易日报

## 交易记录
- 0700.HK BUY 100 380 HKD | reason=港股仓位回补
- 4063.T SELL 50 2750 JPY | reason=日股短线兑现
- AAPL.US BUY 10 180 USD | reason=美股核心仓
- 0331418A.JP BUY 200 1250 JPY | asset_type=fund | reason=日本基金定投
"""
    svc = MarkdownLogIntakeCanonicalizer()
    output = svc.ingest(LogIntakeInput(user_id="u1", run_date=date(2026, 3, 5), raw_log_text=text))

    events = output.normalized.trade_events
    assert len(events) == 4
    assert events[0].ticker == "0700.HK"
    assert events[1].ticker == "4063.T"
    assert events[2].ticker == "AAPL.US"
    assert events[3].asset_type == AssetType.FUND
