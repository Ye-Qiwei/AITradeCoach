"""Strict LLM-facing structured output contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationWindowLiteral = Literal["1 day", "1 week", "1 month", "3 months", "1 year"]


class StrictLLMContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TradeActionContract(StrictLLMContractModel):
    action: Literal["buy", "sell", "add", "reduce", "hold", "watch"]
    target_asset: str
    position_change: str
    action_time: str
    reason: str


class AtomicJudgementContract(StrictLLMContractModel):
    id: str
    core_thesis: str
    evaluation_timeframe: EvaluationWindowLiteral
    dependencies: list[str]


class JudgementItemContract(StrictLLMContractModel):
    category: Literal[
        "market_view",
        "asset_view",
        "macro_view",
        "risk_view",
        "opportunity_view",
        "non_action",
        "reflection",
    ]
    target_asset_or_topic: str
    thesis: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_from_user_log: list[str]
    implicitness: Literal["explicit", "implicit", "mixed"]
    related_actions: list[str]
    related_non_actions: list[str]
    estimated_horizon: str
    proposed_evaluation_window: EvaluationWindowLiteral
    atomic_judgements: list[AtomicJudgementContract]


class ParserOutputContract(StrictLLMContractModel):
    user_id: str
    run_date: date
    trade_actions: list[TradeActionContract]
    explicit_judgements: list[JudgementItemContract]
    implicit_judgements: list[JudgementItemContract]
    opportunity_judgements: list[JudgementItemContract]
    non_action_judgements: list[JudgementItemContract]
    reflection_summary: list[str]

    def all_judgements(self) -> list[JudgementItemContract]:
        return [
            *self.explicit_judgements,
            *self.implicit_judgements,
            *self.opportunity_judgements,
            *self.non_action_judgements,
        ]


class JudgementEvidenceContract(StrictLLMContractModel):
    judgement_id: str
    evidence_item_ids: list[str]
    support_signal: Literal["support", "oppose", "uncertain"]
    sufficiency_reason: str


class ResearchAgentJudgementEvidenceContract(StrictLLMContractModel):
    judgement_id: str
    evidence_item_ids: list[str]
    support_signal: Literal["support", "oppose", "uncertain"]
    sufficiency_reason: str


class ResearchAgentFinalContract(StrictLLMContractModel):
    judgement_evidence: list[ResearchAgentJudgementEvidenceContract]
    stop_reason: str


class ResearchSynthesisOutputContract(StrictLLMContractModel):
    judgement_evidence: list[JudgementEvidenceContract]
    stop_reason: str


class DailyJudgementFeedbackContract(StrictLLMContractModel):
    judgement_id: str
    initial_feedback: Literal["likely_correct", "likely_wrong", "insufficient_evidence", "high_uncertainty"]
    evidence_summary: str
    evaluation_window: EvaluationWindowLiteral
    window_rationale: str
    followup_indicators: list[str]
    source_ids: list[str]


class ReporterOutputContract(StrictLLMContractModel):
    markdown: str = Field(..., min_length=20)
    judgement_feedback: list[DailyJudgementFeedbackContract]


class JudgeVerdictContract(StrictLLMContractModel):
    passed: bool
    reasons: list[str]
    rewrite_instruction: str
    contradiction_flags: list[str]
