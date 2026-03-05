from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import EvidencePlanningInput
from ai_trading_coach.domain.enums import HypothesisType
from ai_trading_coach.domain.models import CognitionState, Hypothesis, RelevantMemorySet
from ai_trading_coach.modules.evidence.service import ClaimDrivenEvidencePlanner


def test_evidence_planner_builds_claim_driven_needs() -> None:
    planner = ClaimDrivenEvidencePlanner()
    cognition = CognitionState(
        cognition_id="c1",
        log_id="l1",
        user_id="u1",
        as_of_date=date(2026, 3, 5),
        hypotheses=[
            Hypothesis(
                hypothesis_id="h1",
                statement="财报催化后股价有修复空间",
                hypothesis_type=HypothesisType.SHORT_CATALYST,
                related_tickers=["9660.HK"],
            )
        ],
        risk_concerns=["地缘冲突可能导致风险偏好持续回落"],
    )

    out = planner.plan(
        EvidencePlanningInput(
            cognition_state=cognition,
            relevant_history=RelevantMemorySet(),
            task_goals=["daily_cognition_review"],
        )
    )

    assert out.plan.needs
    assert out.plan.requires_event_centered_analysis is True
    assert out.plan.priority_order
