"""Adapters from strict LLM output contracts to internal domain models."""

from __future__ import annotations

from ai_trading_coach.domain.agent_models import JudgeVerdict, ReporterOutput
from ai_trading_coach.domain.id_generation import make_action_id, make_judgement_id, make_parse_id, make_research_id
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
    ResearchAgentFinalContract,
    ResearchSynthesisOutputContract,
    TradeActionContract,
)


def _empty_to_none(value: str) -> str | None:
    return value or None


def trade_action_contract_to_domain(contract: TradeActionContract, *, action_id: str = "") -> TradeAction:
    return TradeAction(
        action_id=action_id,
        action=contract.action,
        target_asset=contract.target_asset,
        position_change=_empty_to_none(contract.position_change),
        action_time=_empty_to_none(contract.action_time),
        reason=_empty_to_none(contract.reason),
    )


def _valid_related_action_ids(candidates: list[str], valid_ids: set[str]) -> list[str]:
    return [item for item in candidates if item in valid_ids]


def judgement_item_contract_to_domain(contract: JudgementItemContract, *, judgement_id: str, valid_action_ids: set[str]) -> JudgementItem:
    return JudgementItem(
        judgement_id=judgement_id,
        category=contract.category,
        target_asset_or_topic=contract.target_asset_or_topic,
        thesis=contract.thesis,
        confidence=contract.confidence,
        evidence_from_user_log=contract.evidence_from_user_log,
        implicitness=contract.implicitness,
        related_actions=_valid_related_action_ids(contract.related_actions, valid_action_ids),
        related_non_actions=contract.related_non_actions,
        estimated_horizon=_empty_to_none(contract.estimated_horizon),
        proposed_evaluation_window=contract.proposed_evaluation_window,
    )


def parser_contract_to_domain(contract: ParserOutputContract, *, run_id: str, raw_log_text: str) -> ParserOutput:
    trade_actions = [
        trade_action_contract_to_domain(item, action_id=make_action_id(run_id, idx, item))
        for idx, item in enumerate(contract.trade_actions, start=1)
    ]
    valid_action_ids = {action.action_id for action in trade_actions if action.action_id}

    def _map_judgements(items: list[JudgementItemContract], kind: str) -> list[JudgementItem]:
        return [
            judgement_item_contract_to_domain(
                item,
                judgement_id=make_judgement_id(
                    run_id,
                    kind,
                    idx,
                    item.category,
                    item.target_asset_or_topic,
                    item.thesis,
                ),
                valid_action_ids=valid_action_ids,
            )
            for idx, item in enumerate(items, start=1)
        ]

    return ParserOutput(
        parse_id=make_parse_id(run_id, raw_log_text),
        user_id=contract.user_id,
        run_date=contract.run_date,
        trade_actions=trade_actions,
        explicit_judgements=_map_judgements(contract.explicit_judgements, "explicit"),
        implicit_judgements=_map_judgements(contract.implicit_judgements, "implicit"),
        opportunity_judgements=_map_judgements(contract.opportunity_judgements, "opportunity"),
        non_action_judgements=_map_judgements(contract.non_action_judgements, "non_action"),
        reflection_summary=contract.reflection_summary,
    )


def judgement_evidence_contract_to_domain(contract: JudgementEvidenceContract) -> JudgementEvidence:
    return JudgementEvidence(
        judgement_id=contract.judgement_id,
        evidence_item_ids=contract.evidence_item_ids,
        support_signal=contract.support_signal,
        sufficiency_reason=contract.sufficiency_reason,
    )


def research_synthesis_contract_to_domain(contract: ResearchSynthesisOutputContract, *, run_id: str) -> ResearchSynthesisOutput:
    return ResearchSynthesisOutput(
        research_id=make_research_id(run_id),
        judgement_evidence=[judgement_evidence_contract_to_domain(i) for i in contract.judgement_evidence],
        stop_reason=contract.stop_reason,
    )


def research_agent_contract_to_domain(contract: ResearchAgentFinalContract, *, run_id: str) -> ResearchSynthesisOutput:
    return ResearchSynthesisOutput(
        research_id=make_research_id(run_id),
        judgement_evidence=[JudgementEvidence.model_validate(item.model_dump(mode="json")) for item in contract.judgement_evidence],
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
    )
