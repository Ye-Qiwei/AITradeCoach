"""Cognition vs reality evaluator implementation."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import EvaluatorInput, EvaluatorOutput
from ai_trading_coach.domain.enums import BiasType, EvaluationCategory
from ai_trading_coach.domain.models import (
    BiasFinding,
    EvaluationResult,
    ExecutionAssessment,
    HypothesisAssessment,
)


class LayeredCognitionRealityEvaluator:
    """Heuristic evaluator with fact/interpretation/evaluation separation."""

    def evaluate(self, data: EvaluatorInput) -> EvaluatorOutput:
        completeness = data.evidence_packet.completeness_score
        follow_up = data.window_decision.follow_up_needed

        hypothesis_assessments = self._assess_hypotheses(data, completeness, follow_up)
        bias_findings = self._detect_biases(data, completeness)

        strengths = self._strengths(data)
        mistakes = self._mistakes(data)
        ahead = self._ahead_of_market(data, completeness)
        execution = self._execution_assessment(data)
        follow_up_signals = self._follow_up_signals(data)
        warning_flags = self._warning_flags(data, completeness)

        summary = self._summary(hypothesis_assessments, completeness, follow_up)

        result = EvaluationResult(
            evaluation_id=f"eval_{data.cognition_state.cognition_id}",
            user_id=data.cognition_state.user_id,
            as_of_date=data.cognition_state.as_of_date,
            summary=summary,
            hypothesis_assessments=hypothesis_assessments,
            bias_findings=bias_findings,
            strengths=strengths,
            mistakes=mistakes,
            ahead_of_market_observations=ahead,
            execution_assessment=execution,
            follow_up_signals=follow_up_signals,
            warning_flags=warning_flags,
            extensions={
                "layers": {
                    "facts": self._fact_layer(data),
                    "interpretations": [assessment.commentary for assessment in hypothesis_assessments],
                    "evaluations": [assessment.category.value for assessment in hypothesis_assessments],
                }
            },
        )
        return EvaluatorOutput(evaluation=result)

    def _assess_hypotheses(
        self,
        data: EvaluatorInput,
        completeness: float,
        follow_up: bool,
    ) -> list[HypothesisAssessment]:
        assessments: list[HypothesisAssessment] = []
        weaken_ids = [item.item_id for item in data.evidence_packet.news_evidence[:1]]
        support_ids = [item.item_id for item in data.evidence_packet.price_evidence[:2]]

        for hyp in data.cognition_state.hypotheses:
            if follow_up and completeness < 0.55:
                category = EvaluationCategory.AHEAD_OF_MARKET
                thesis_still_valid = True
                market_phase = False
                commentary = "当前证据不足以终判，方向可能正确但仍在等待验证窗口。"
            elif completeness >= 0.7:
                category = EvaluationCategory.CORRECT
                thesis_still_valid = True
                market_phase = True
                commentary = "现有证据整体支持该判断。"
            elif completeness >= 0.55:
                category = EvaluationCategory.PARTIAL
                thesis_still_valid = True
                market_phase = True
                commentary = "判断部分成立，但需要更多证据确认关键链条。"
            else:
                category = EvaluationCategory.DIRECTION_RIGHT_TIMING_WRONG
                thesis_still_valid = True
                market_phase = False
                commentary = "方向未被证伪，但时间窗口可能尚未进入有效验证阶段。"

            assessments.append(
                HypothesisAssessment(
                    hypothesis_id=hyp.hypothesis_id,
                    category=category,
                    thesis_still_valid=thesis_still_valid,
                    market_in_verification_phase=market_phase,
                    support_evidence_ids=support_ids,
                    weaken_evidence_ids=weaken_ids,
                    commentary=commentary,
                    confidence=max(0.35, min(0.9, completeness + 0.15)),
                )
            )

        return assessments

    def _detect_biases(self, data: EvaluatorInput, completeness: float) -> list[BiasFinding]:
        findings: list[BiasFinding] = []

        has_fomo = any(signal.signal_type == "fomo" for signal in data.cognition_state.behavioral_signals)
        if has_fomo:
            findings.append(
                BiasFinding(
                    bias_type=BiasType.EMOTION,
                    severity=4,
                    description="存在明显 FOMO 信号，可能放大追涨或频繁切换倾向。",
                    evidence=[signal.evidence for signal in data.cognition_state.behavioral_signals if signal.signal_type == "fomo"],
                    correction="将候选机会放入观察清单，延迟 1-2 个交易日再做决策。",
                )
            )

        if completeness < 0.5:
            findings.append(
                BiasFinding(
                    bias_type=BiasType.EVIDENCE_SELECTION,
                    severity=3,
                    description="当前证据覆盖不足，容易出现样本选择偏差。",
                    evidence=data.evidence_packet.missing_requirements[:3],
                    correction="补充缺失证据后再做终判，避免单一维度下结论。",
                )
            )

        if data.window_decision.follow_up_needed:
            findings.append(
                BiasFinding(
                    bias_type=BiasType.TIME_SCALE,
                    severity=3,
                    description="当前窗口尚处预判阶段，存在时间尺度偏差风险。",
                    evidence=data.window_decision.selection_reason,
                    correction="按推荐复盘日期进行二次验证，避免过早定性。",
                )
            )

        return findings

    def _strengths(self, data: EvaluatorInput) -> list[str]:
        strengths: list[str] = []
        if data.cognition_state.deliberate_no_trade_decisions:
            strengths.append("在不确定阶段选择不交易，体现了纪律性。")
        if data.cognition_state.explicit_rules:
            strengths.append("有主动规则沉淀，说明认知框架在进化。")
        if data.cognition_state.risk_concerns:
            strengths.append("风险识别意识较强，能主动暴露脆弱点。")
        return strengths

    def _mistakes(self, data: EvaluatorInput) -> list[str]:
        mistakes: list[str] = []
        if data.cognition_state.outside_opportunities:
            mistakes.append("场外机会关注较多，可能分散主线判断。")
        if any(signal.signal_type == "stress_affected_execution" for signal in data.cognition_state.behavioral_signals):
            mistakes.append("高压力状态下执行，容易偏离原计划。")
        if not data.cognition_state.fact_statements:
            mistakes.append("事实锚点记录较少，复盘时验证链条不够完整。")
        return mistakes

    def _ahead_of_market(self, data: EvaluatorInput, completeness: float) -> list[str]:
        if not data.window_decision.follow_up_needed:
            return []
        if completeness >= 0.45:
            return ["判断可能超前于市场定价，尚需等待验证窗口。"]
        return ["短期未兑现不等于判断错误，当前更适合延迟终判。"]

    def _execution_assessment(self, data: EvaluatorInput) -> ExecutionAssessment:
        discipline_score = 0.7 if data.cognition_state.deliberate_no_trade_decisions else 0.55
        position_sizing_score = 0.7 if len(data.position_snapshot.holdings) <= 3 else 0.5
        risk_control_score = 0.72 if data.cognition_state.risk_concerns else 0.55

        notes = [
            f"holdings={len(data.position_snapshot.holdings)}",
            f"bias_findings={len(self._detect_biases(data, data.evidence_packet.completeness_score))}",
        ]
        return ExecutionAssessment(
            discipline_score=discipline_score,
            position_sizing_score=position_sizing_score,
            risk_control_score=risk_control_score,
            notes=notes,
        )

    def _follow_up_signals(self, data: EvaluatorInput) -> list[str]:
        signals = [
            question
            for choice in data.window_decision.selected_windows
            for question in choice.target_questions
        ]
        signals.extend([f"补充证据: {item}" for item in data.evidence_packet.missing_requirements[:3]])
        if data.window_decision.recommended_next_review_date:
            signals.append(f"下次复盘日期: {data.window_decision.recommended_next_review_date.isoformat()}")
        return signals

    def _warning_flags(self, data: EvaluatorInput, completeness: float) -> list[str]:
        flags: list[str] = []
        if completeness < 0.45:
            flags.append("证据覆盖不足，禁止做终局结论。")
        if any(signal.signal_type == "fomo" for signal in data.cognition_state.behavioral_signals):
            flags.append("FOMO 信号偏强，警惕冲动执行。")
        if any(bias.bias_type == BiasType.TIME_SCALE for bias in self._detect_biases(data, completeness)):
            flags.append("时间尺度偏差风险存在，需按窗口复核。")
        return flags

    def _summary(
        self,
        assessments: list[HypothesisAssessment],
        completeness: float,
        follow_up: bool,
    ) -> str:
        if not assessments:
            return "未提取到可评估假设，建议增强日志中的可验证判断。"

        top = assessments[0].category.value
        if follow_up:
            return f"当前评估为暂判（{top}），证据完整度={completeness:.2f}，建议按计划窗口复核。"
        return f"当前评估倾向为 {top}，证据完整度={completeness:.2f}。"

    def _fact_layer(self, data: EvaluatorInput) -> list[str]:
        return [
            f"price_evidence={len(data.evidence_packet.price_evidence)}",
            f"news_evidence={len(data.evidence_packet.news_evidence)}",
            f"filing_evidence={len(data.evidence_packet.filing_evidence)}",
            f"completeness={data.evidence_packet.completeness_score:.2f}",
            f"selected_windows={len(data.window_decision.selected_windows)}",
        ]


# Backward-compatible alias
PlaceholderCognitionRealityEvaluator = LayeredCognitionRealityEvaluator
