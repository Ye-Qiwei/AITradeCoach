from __future__ import annotations

from datetime import date

import pytest

from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, JudgementItem, compute_due_date


def test_judgement_item_window_must_be_legal() -> None:
    with pytest.raises(ValueError):
        JudgementItem(category="market_view", target="SPX", thesis="Market rallies", evaluation_window="2 weeks")


def test_daily_feedback_window_must_be_legal() -> None:
    with pytest.raises(ValueError):
        DailyJudgementFeedback(initial_feedback="likely_correct", evaluation_window="tomorrow")


def test_compute_due_date() -> None:
    assert compute_due_date(date(2026, 1, 1), "1 week") == date(2026, 1, 8)
