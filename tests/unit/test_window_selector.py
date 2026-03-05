from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import WindowSelectorInput
from ai_trading_coach.domain.enums import HypothesisType
from ai_trading_coach.domain.models import (
    CognitionState,
    EvidencePlan,
    Hypothesis,
    PositionSnapshot,
    TradeLedger,
)
from ai_trading_coach.modules.window.rule_based_selector import RuleBasedWindowSelector


def test_window_selector_short_catalyst_enables_follow_up_when_low_completeness() -> None:
    selector = RuleBasedWindowSelector()
    input_data = WindowSelectorInput(
        plan=EvidencePlan(plan_id="p1", user_id="u1"),
        cognition_state=CognitionState(
            cognition_id="c1",
            log_id="l1",
            user_id="u1",
            as_of_date=date(2026, 3, 4),
            hypotheses=[
                Hypothesis(
                    hypothesis_id="h1",
                    statement="earnings surprise will rerate stock",
                    hypothesis_type=HypothesisType.SHORT_CATALYST,
                )
            ],
        ),
        trade_ledger=TradeLedger(ledger_id="ld1", user_id="u1", as_of_date=date(2026, 3, 4)),
        position_snapshot=PositionSnapshot(snapshot_id="ps1", user_id="u1", as_of_date=date(2026, 3, 4)),
        evidence_completeness=0.4,
    )

    decision = selector.select(input_data).decision

    selected_types = {x.window_type.value for x in decision.selected_windows}
    assert "event_centered_window" in selected_types
    assert "1D" in selected_types
    assert decision.follow_up_needed is True
