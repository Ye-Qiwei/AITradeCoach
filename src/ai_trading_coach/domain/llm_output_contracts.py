"""Strict LLM-facing structured output contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationWindowLiteral = Literal["1 day", "1 week", "1 month", "3 months", "1 year"]


class StrictLLMContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TradeActionContract(StrictLLMContractModel):
    action: Literal["buy", "sell", "add", "reduce", "hold", "watch"]
    target_asset: str


class ParsedJudgementContract(StrictLLMContractModel):
    local_id: str
    category: Literal[
        "market_view",
        "asset_view",
        "macro_view",
        "risk_view",
        "opportunity_view",
        "non_action",
        "reflection",
    ]
    target: str
    thesis: str
    evaluation_window: EvaluationWindowLiteral
    dependencies: list[str]


class ParserOutputContract(StrictLLMContractModel):
    trade_actions: list[TradeActionContract]
    judgements: list[ParsedJudgementContract]


class JudgementEvidenceContract(StrictLLMContractModel):
    judgement_id: str
    evidence_item_ids: list[str]
    support_signal: Literal["support", "oppose", "uncertain"]
    evidence_quality: Literal["sufficient", "insufficient", "conflicting", "stale", "indirect"]


class ResearchAgentFinalContract(StrictLLMContractModel):
    judgement_evidence: list[JudgementEvidenceContract]


class DailyJudgementFeedbackContract(StrictLLMContractModel):
    judgement_id: str
    initial_feedback: Literal["likely_correct", "likely_wrong", "insufficient_evidence", "high_uncertainty"]
    evaluation_window: EvaluationWindowLiteral


class ReporterOutputContract(StrictLLMContractModel):
    markdown: str = Field(..., min_length=20)
    judgement_feedback: list[DailyJudgementFeedbackContract]


class JudgeVerdictContract(StrictLLMContractModel):
    passed: bool
    reasons: list[str]
    rewrite_instruction: str
