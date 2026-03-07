"""Adapters from strict LLM output contracts to internal domain models."""

from __future__ import annotations

from ai_trading_coach.domain.agent_models import JudgeVerdict, ReporterOutput
from ai_trading_coach.domain.judgement_models import (
    DailyJudgementFeedback,
    JudgementEvidence,
    JudgementItem,
    ParserOutput,
    ResearchSynthesisOutput,
    TradeAction,
)
from ai_trading_coach.domain.llm_output_contracts import (
    DailyJudgementFeedbackContract,
    JudgeVerdictContract,
    JudgementEvidenceContract,
    JudgementItemContract,
    ParserOutputContract,
    ReporterOutputContract,
    ResearchSynthesisOutputContract,
    TradeActionContract,
)


def _empty_to_none(value: str) -> str | None:
    return value or None


def trade_action_contract_to_domain(contract: TradeActionContract) -> TradeAction:
    return TradeAction(
        action=contract.action,
        target_asset=contract.target_asset,
        position_change=_empty_to_none(contract.position_change),
        action_time=_empty_to_none(contract.action_time),
        reason=_empty_to_none(contract.reason),
    )


def judgement_item_contract_to_domain(contract: JudgementItemContract) -> JudgementItem:
    return JudgementItem(
        judgement_id=contract.judgement_id,
        category=contract.category,
        target_asset_or_topic=contract.target_asset_or_topic,
        thesis=contract.thesis,
        confidence=contract.confidence,
        evidence_from_user_log=contract.evidence_from_user_log,
        implicitness=contract.implicitness,
        related_actions=contract.related_actions,
        related_non_actions=contract.related_non_actions,
        estimated_horizon=_empty_to_none(contract.estimated_horizon),
        proposed_evaluation_window=contract.proposed_evaluation_window,
    )


def parser_contract_to_domain(contract: ParserOutputContract) -> ParserOutput:
    return ParserOutput(
        parse_id=contract.parse_id,
        user_id=contract.user_id,
        run_date=contract.run_date,
        trade_actions=[trade_action_contract_to_domain(i) for i in contract.trade_actions],
        explicit_judgements=[judgement_item_contract_to_domain(i) for i in contract.explicit_judgements],
        implicit_judgements=[judgement_item_contract_to_domain(i) for i in contract.implicit_judgements],
        opportunity_judgements=[judgement_item_contract_to_domain(i) for i in contract.opportunity_judgements],
        non_action_judgements=[judgement_item_contract_to_domain(i) for i in contract.non_action_judgements],
        reflection_summary=contract.reflection_summary,
    )


def judgement_evidence_contract_to_domain(contract: JudgementEvidenceContract) -> JudgementEvidence:
    return JudgementEvidence(
        judgement_id=contract.judgement_id,
        evidence_item_ids=contract.evidence_item_ids,
        support_signal=contract.support_signal,
        sufficiency_reason=contract.sufficiency_reason,
    )


def research_synthesis_contract_to_domain(contract: ResearchSynthesisOutputContract) -> ResearchSynthesisOutput:
    return ResearchSynthesisOutput(
        research_id=contract.research_id,
        judgement_evidence=[judgement_evidence_contract_to_domain(i) for i in contract.judgement_evidence],
        stop_reason=contract.stop_reason,
    )


def daily_feedback_contract_to_domain(contract: DailyJudgementFeedbackContract) -> DailyJudgementFeedback:
    return DailyJudgementFeedback(
        judgement_id=contract.judgement_id,
        initial_feedback=contract.initial_feedback,
        evidence_summary=contract.evidence_summary,
        evaluation_window=contract.evaluation_window,
        window_rationale=contract.window_rationale,
        followup_indicators=contract.followup_indicators,
        source_ids=contract.source_ids,
    )


def reporter_contract_to_domain(contract: ReporterOutputContract) -> ReporterOutput:
    return ReporterOutput(
        markdown=contract.markdown,
        judgement_feedback=[daily_feedback_contract_to_domain(i) for i in contract.judgement_feedback],
    )


def judge_verdict_contract_to_domain(contract: JudgeVerdictContract) -> JudgeVerdict:
    return JudgeVerdict(
        passed=contract.passed,
        reasons=contract.reasons,
        rewrite_instruction=_empty_to_none(contract.rewrite_instruction),
        contradiction_flags=contract.contradiction_flags,
        citation_coverage=contract.citation_coverage,
    )
