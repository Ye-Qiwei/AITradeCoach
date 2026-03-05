from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ai_trading_coach.domain.enums import AnalysisWindowType, TradeSide
from ai_trading_coach.domain.models import DailyLogRaw, TradeEvent, WindowChoice


def test_daily_log_raw_requires_content() -> None:
    model = DailyLogRaw(log_id="log1", user_id="u1", content="hello")
    assert model.content == "hello"


def test_trade_event_buy_requires_price() -> None:
    with pytest.raises(ValidationError):
        TradeEvent(
            event_id="e1",
            user_id="u1",
            trade_date=date(2026, 3, 4),
            ticker="9660.HK",
            side=TradeSide.BUY,
            quantity=100,
        )


def test_window_choice_rejects_reversed_dates() -> None:
    with pytest.raises(ValidationError):
        WindowChoice(
            window_type=AnalysisWindowType.D5,
            start_date=date(2026, 3, 5),
            end_date=date(2026, 3, 4),
            reason="invalid",
        )
