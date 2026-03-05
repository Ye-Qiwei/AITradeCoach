"""Replay evaluation utilities for controlled PromptOps offline checks."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import EvaluationCategory
from ai_trading_coach.domain.models import (
    ReplayCase,
    ReplayCaseResult,
    ReplayEvaluationResult,
    ReplayPrediction,
)


class ReplayEvaluator:
    """Evaluate replay cases against predicted outcomes."""

    def evaluate(self, cases: list[ReplayCase], predictions: list[ReplayPrediction]) -> ReplayEvaluationResult:
        prediction_map = {prediction.case_id: prediction for prediction in predictions}
        case_results: list[ReplayCaseResult] = []

        expected_total = 0
        matched_total = 0
        unexpected_total = 0

        follow_up_checks = 0
        follow_up_hits = 0

        ahead_expected = 0
        ahead_hits = 0

        for case in cases:
            prediction = prediction_map.get(case.case_id, ReplayPrediction(case_id=case.case_id))
            expected = set(case.expected_categories)
            predicted = set(prediction.predicted_categories)

            matched = sorted(expected & predicted, key=lambda item: item.value)
            missed = sorted(expected - predicted, key=lambda item: item.value)
            unexpected = sorted(predicted - expected, key=lambda item: item.value)

            expected_total += len(expected)
            matched_total += len(matched)
            unexpected_total += len(unexpected)

            if EvaluationCategory.AHEAD_OF_MARKET in expected:
                ahead_expected += 1
                if EvaluationCategory.AHEAD_OF_MARKET in matched:
                    ahead_hits += 1

            follow_up_match: bool | None = None
            if case.expected_follow_up_needed is not None:
                follow_up_checks += 1
                follow_up_match = case.expected_follow_up_needed == prediction.predicted_follow_up_needed
                if follow_up_match:
                    follow_up_hits += 1

            base_score = (len(matched) / len(expected)) if expected else 1.0
            penalty = min(0.4, 0.2 * len(unexpected))
            if follow_up_match is False:
                penalty = min(0.6, penalty + 0.2)
            case_score = max(0.0, min(1.0, base_score - penalty))

            notes: list[str] = []
            if missed:
                notes.append("存在漏判类别。")
            if unexpected:
                notes.append("存在额外误判类别。")
            if follow_up_match is False:
                notes.append("follow-up 判定与预期不一致。")

            case_results.append(
                ReplayCaseResult(
                    case_id=case.case_id,
                    expected_categories=sorted(expected, key=lambda item: item.value),
                    predicted_categories=sorted(predicted, key=lambda item: item.value),
                    matched_categories=matched,
                    missed_categories=missed,
                    unexpected_categories=unexpected,
                    follow_up_match=follow_up_match,
                    score=round(case_score, 4),
                    notes=notes,
                )
            )

        case_count = len(cases)
        average_score = (sum(result.score for result in case_results) / case_count) if case_count else 0.0
        hit_rate = (matched_total / expected_total) if expected_total else 0.0

        prediction_total = sum(len(pred.predicted_categories) for pred in predictions)
        unexpected_rate = (unexpected_total / prediction_total) if prediction_total else 0.0

        follow_up_accuracy: float | None = None
        if follow_up_checks > 0:
            follow_up_accuracy = follow_up_hits / follow_up_checks

        ahead_recall: float | None = None
        if ahead_expected > 0:
            ahead_recall = ahead_hits / ahead_expected

        recommendation = self._recommendation(
            average_score=average_score,
            hit_rate=hit_rate,
            unexpected_rate=unexpected_rate,
            ahead_recall=ahead_recall,
        )

        replay_id = f"replay_eval_{date.today().isoformat()}_{case_count}"
        return ReplayEvaluationResult(
            replay_id=replay_id,
            case_results=case_results,
            case_count=case_count,
            average_score=round(average_score, 4),
            category_hit_rate=round(hit_rate, 4),
            unexpected_category_rate=round(unexpected_rate, 4),
            follow_up_accuracy=round(follow_up_accuracy, 4) if follow_up_accuracy is not None else None,
            ahead_of_market_recall=round(ahead_recall, 4) if ahead_recall is not None else None,
            recommendation=recommendation,
        )

    def _recommendation(
        self,
        average_score: float,
        hit_rate: float,
        unexpected_rate: float,
        ahead_recall: float | None,
    ) -> str:
        if average_score >= 0.82 and hit_rate >= 0.8 and unexpected_rate <= 0.2:
            if ahead_recall is not None and ahead_recall < 0.6:
                return "整体可进入 A/B，但需先提升 ahead_of_market 识别召回。"
            return "离线表现达标，可进入 A/B 验证阶段。"
        if average_score >= 0.7:
            return "离线表现中等，建议继续扩充回放样本后再进入 A/B。"
        return "离线表现不足，建议拒绝晋升并迭代提案。"

