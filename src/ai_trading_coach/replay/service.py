"""Replay runner for historical-case offline evaluation."""

from __future__ import annotations

from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReplayCase, ReplayEvaluationResult, ReplayPrediction, ReviewRunRequest
from ai_trading_coach.interfaces.modules import SystemOrchestrator
from ai_trading_coach.modules.promptops.replay import ReplayEvaluator


class ReplayRunner:
    """Run replay cases through orchestrator and evaluate prediction quality."""

    def __init__(
        self,
        orchestrator: SystemOrchestrator,
        evaluator: ReplayEvaluator | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.evaluator = evaluator or ReplayEvaluator()

    def run(self, cases: list[ReplayCase]) -> ReplayEvaluationResult:
        predictions: list[ReplayPrediction] = []
        for case in cases:
            request = ReviewRunRequest(
                run_id=f"replay_{case.case_id}",
                user_id=case.user_id,
                run_date=case.run_date,
                trigger_type=TriggerType.REPLAY,
                raw_log_text=case.raw_log_text,
                options={"dry_run": True},
            )
            result = self.orchestrator.run(request)

            categories = []
            follow_up_needed = None
            if result.evaluation is not None:
                categories = sorted(
                    {assessment.category for assessment in result.evaluation.hypothesis_assessments},
                    key=lambda item: item.value,
                )
                follow_up_needed = bool(result.evaluation.follow_up_signals)

            predictions.append(
                ReplayPrediction(
                    case_id=case.case_id,
                    predicted_categories=categories,
                    predicted_follow_up_needed=follow_up_needed,
                    notes=[result.status.value],
                )
            )

        return self.evaluator.evaluate(cases=cases, predictions=predictions)
