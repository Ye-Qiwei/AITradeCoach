"""Adapters from strict LLM output contracts to internal domain models."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.agent_models import JudgeVerdict, ReporterOutput
from ai_trading_coach.domain.id_generation import make_action_id, make_judgement_id, make_parse_id, make_research_id
from ai_trading_coach.domain.judgement_models import (
    DailyJudgementFeedback,
    JudgementEvidence,
    JudgementItem,
    ParserOutput,
    ResearchOutput,
    TradeAction,
)
from ai_trading_coach.domain.llm_output_contracts import (
    DailyJudgementFeedbackContract,
    JudgeVerdictContract,
    JudgementEvidenceContract,
    ParsedJudgementContract,
    ParserOutputContract,
    ReporterOutputContract,
    ResearchAgentFinalContract,
    TradeActionContract,
)


def trade_action_contract_to_domain(contract: TradeActionContract, *, action_id: str = "") -> TradeAction:
    return TradeAction(
        action_id=action_id,
        action=contract.action,
        target_asset=contract.target_asset,
    )


def judgement_item_contract_to_domain(contract: ParsedJudgementContract, *, judgement_id: str, dependency_map: dict[str, str]) -> JudgementItem:
    return JudgementItem(
        judgement_id=judgement_id,
        category=contract.category,
        target=contract.target,
        thesis=contract.thesis,
        evaluation_window=contract.evaluation_window,
        dependencies=[dependency_map[item] for item in contract.dependencies if item in dependency_map],
    )


def parser_contract_to_domain(contract: ParserOutputContract, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> ParserOutput:
    trade_actions = [
        trade_action_contract_to_domain(item, action_id=make_action_id(run_id, idx, item))
        for idx, item in enumerate(contract.trade_actions, start=1)
    ]
    local_to_global: dict[str, str] = {
        item.local_id: make_judgement_id(run_id, "judgement", idx, item.category, item.target, item.thesis)
        for idx, item in enumerate(contract.judgements, start=1)
    }
    judgements = [
        judgement_item_contract_to_domain(item, judgement_id=local_to_global[item.local_id], dependency_map=local_to_global)
        for item in contract.judgements
    ]

    return ParserOutput(
        parse_id=make_parse_id(run_id, raw_log_text),
        user_id=user_id,
        run_date=run_date,
        trade_actions=trade_actions,
        judgements=judgements,
    )


def judgement_evidence_contract_to_domain(contract: JudgementEvidenceContract) -> JudgementEvidence:
    return JudgementEvidence(
        judgement_id=contract.judgement_id,
        evidence_item_ids=contract.evidence_item_ids,
        support_signal=contract.support_signal,
        evidence_quality=contract.evidence_quality,
    )


def research_agent_contract_to_domain(contract: ResearchAgentFinalContract, *, run_id: str) -> ResearchOutput:
    return ResearchOutput(
        research_id=make_research_id(run_id),
        judgement_evidence=[JudgementEvidence.model_validate(item.model_dump(mode="json")) for item in contract.judgement_evidence],
    )


def daily_feedback_contract_to_domain(contract: DailyJudgementFeedbackContract) -> DailyJudgementFeedback:
    return DailyJudgementFeedback(
        judgement_id=contract.judgement_id,
        initial_feedback=contract.initial_feedback,
        evaluation_window=contract.evaluation_window,
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
        rewrite_instruction=contract.rewrite_instruction or None,
    )
