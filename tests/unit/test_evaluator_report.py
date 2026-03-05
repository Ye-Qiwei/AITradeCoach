from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import EvaluatorInput, ReportGeneratorInput
from ai_trading_coach.domain.enums import AnalysisWindowType, HypothesisType
from ai_trading_coach.domain.models import (
    CognitionState,
    EvidenceItem,
    EvidencePacket,
    Hypothesis,
    PnLSnapshot,
    PositionSnapshot,
    RelevantMemorySet,
    WindowChoice,
    WindowDecision,
)
from ai_trading_coach.modules.evaluator.service import LayeredCognitionRealityEvaluator
from ai_trading_coach.modules.report.service import StructuredReviewReportGenerator


def test_evaluator_and_report_generator_produce_structured_outputs() -> None:
    cognition = CognitionState(
        cognition_id="c1",
        log_id="l1",
        user_id="u1",
        as_of_date=date(2026, 3, 5),
        hypotheses=[
            Hypothesis(
                hypothesis_id="h1",
                statement="长期 thesis 仍有效",
                hypothesis_type=HypothesisType.LONG_THESIS,
                related_tickers=["9660.HK"],
            )
        ],
        explicit_rules=["先看资金面再放大基本面权重"],
    )
    evidence = EvidencePacket(
        packet_id="ep1",
        user_id="u1",
        completeness_score=0.62,
        price_evidence=[
            EvidenceItem(item_id="ev1", evidence_type="price_path", summary="mock", data={}, related_tickers=["9660.HK"])
        ],
    )
    window_decision = WindowDecision(
        decision_id="wd1",
        selected_windows=[
            WindowChoice(
                window_type=AnalysisWindowType.D120,
                start_date=date(2025, 11, 5),
                end_date=date(2026, 3, 5),
                reason="long thesis check",
            )
        ],
        selection_reason=["long thesis"],
        follow_up_needed=False,
        confidence=0.7,
    )

    evaluator = LayeredCognitionRealityEvaluator()
    eval_out = evaluator.evaluate(
        EvaluatorInput(
            cognition_state=cognition,
            evidence_packet=evidence,
            window_decision=window_decision,
            relevant_memories=RelevantMemorySet(),
            position_snapshot=PositionSnapshot(snapshot_id="ps1", user_id="u1", as_of_date=date(2026, 3, 5)),
        )
    )

    reporter = StructuredReviewReportGenerator()
    report_out = reporter.generate(
        ReportGeneratorInput(
            evaluation=eval_out.evaluation,
            position_snapshot=PositionSnapshot(snapshot_id="ps1", user_id="u1", as_of_date=date(2026, 3, 5)),
            pnl_snapshot=PnLSnapshot(snapshot_id="pnl1", user_id="u1", as_of_date=date(2026, 3, 5)),
            evidence_packet=evidence,
            window_decision=window_decision,
        )
    )

    assert eval_out.evaluation.hypothesis_assessments
    assert "layers" in eval_out.evaluation.extensions
    assert len(report_out.report.sections) == 10
    assert "## 1. 今日结论" in report_out.report.markdown_body
