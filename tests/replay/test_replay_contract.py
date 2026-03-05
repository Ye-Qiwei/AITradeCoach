from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import EvaluationCategory
from ai_trading_coach.domain.models import ReplayCase, ReplayPrediction
from ai_trading_coach.modules.promptops.replay import ReplayEvaluator

def test_replay_can_detect_ahead_of_market_cases() -> None:
    evaluator = ReplayEvaluator()
    result = evaluator.evaluate(
        cases=[
            ReplayCase(
                case_id="case_ahead",
                user_id="u1",
                run_date=date(2026, 3, 1),
                raw_log_text="demo",
                expected_categories=[EvaluationCategory.AHEAD_OF_MARKET],
                expected_follow_up_needed=True,
            )
        ],
        predictions=[
            ReplayPrediction(
                case_id="case_ahead",
                predicted_categories=[EvaluationCategory.AHEAD_OF_MARKET],
                predicted_follow_up_needed=True,
            )
        ],
    )
    assert result.case_count == 1
    assert result.average_score == 1.0
    assert result.ahead_of_market_recall == 1.0
