from __future__ import annotations

import pytest

from ai_trading_coach.domain.judgement_models import JudgementItem, ResearchOutput, ResearchedJudgementItem


def test_research_output_validation_passes_with_same_length() -> None:
    output = ResearchOutput(
        judgements=[
            ResearchedJudgementItem(category="asset_view", target="AAPL", thesis="up"),
            ResearchedJudgementItem(category="market_view", target="SPX", thesis="range"),
        ]
    )
    output.validate_against([
        JudgementItem(category="asset_view", target="AAPL", thesis="up"),
        JudgementItem(category="market_view", target="SPX", thesis="range"),
    ])


def test_research_output_validation_rejects_length_mismatch() -> None:
    output = ResearchOutput(judgements=[ResearchedJudgementItem(category="asset_view", target="AAPL", thesis="up")])
    with pytest.raises(ValueError, match="count mismatch"):
        output.validate_against([
            JudgementItem(category="asset_view", target="AAPL", thesis="up"),
            JudgementItem(category="market_view", target="SPX", thesis="range"),
        ])
