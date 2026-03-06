from __future__ import annotations

from datetime import date

import pytest

from ai_trading_coach.domain.judgement_models import (
    DailyJudgementFeedback,
    JudgementItem,
    compute_due_date,
)


def test_judgement_item_window_must_be_legal() -> None:
    with pytest.raises(ValueError):
        JudgementItem(
            judgement_id="j1",
            category="market_view",
            target_asset_or_topic="SPX",
            thesis="Market rallies",
            evidence_from_user_log=["I think SPX goes up"],
            proposed_evaluation_window="2 weeks",
        )


def test_daily_feedback_window_must_be_legal() -> None:
    with pytest.raises(ValueError):
        DailyJudgementFeedback(
            judgement_id="j1",
            initial_feedback="likely_correct",
            evidence_summary="x",
            evaluation_window="tomorrow",
            window_rationale="x",
            followup_indicators=[],
            source_ids=["s1"],
        )


def test_compute_due_date() -> None:
    assert compute_due_date(date(2026, 1, 1), "1 week") == date(2026, 1, 8)
