"""Schemas for parse/research/report active path."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_EVALUATION_WINDOWS = ("1 day", "1 week", "1 month", "3 months", "1 year")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SlimModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradeAction(SlimModel):
    action: Literal["buy", "sell", "add", "reduce", "hold", "watch"]
    target_asset: str


class JudgementItem(SlimModel):
    category: Literal["market_view", "asset_view", "macro_view", "risk_view", "opportunity_view", "non_action", "reflection"]
    target: str
    thesis: str
    evaluation_window: str = "1 week"

    @field_validator("evaluation_window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        if value not in ALLOWED_EVALUATION_WINDOWS:
            raise ValueError(f"evaluation window must be one of {ALLOWED_EVALUATION_WINDOWS}")
        return value


class ParserOutput(SlimModel):
    trade_actions: list[TradeAction] = Field(default_factory=list)
    judgements: list[JudgementItem] = Field(default_factory=list)

    def all_judgements(self) -> list[JudgementItem]:
        return self.judgements


class EvidenceSource(SlimModel):
    provider: str
    title: str | None = None
    uri: str | None = None
    published_at: str | None = None


class CollectedEvidenceItem(SlimModel):
    evidence_type: str = "other"
    summary: str
    related_tickers: list[str] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)


class JudgementEvidence(SlimModel):
    support_signal: Literal["support", "oppose", "uncertain"] = "uncertain"
    evidence_quality: Literal["sufficient", "insufficient", "conflicting", "stale", "indirect"] = "insufficient"
    evidence_summary: str = ""
    collected_evidence_items: list[CollectedEvidenceItem] = Field(default_factory=list)


class ResearchedJudgementItem(JudgementItem):
    evidence: JudgementEvidence = Field(default_factory=JudgementEvidence)


class ResearchOutput(SlimModel):
    judgements: list[ResearchedJudgementItem] = Field(default_factory=list)

    def validate_against(self, original_judgements: list[JudgementItem]) -> None:
        if len(self.judgements) != len(original_judgements):
            raise ValueError(f"Research judgements count mismatch: expected {len(original_judgements)}, got {len(self.judgements)}")


class DailyJudgementFeedback(SlimModel):
    initial_feedback: Literal["likely_correct", "likely_wrong", "insufficient_evidence", "high_uncertainty"]
    evaluation_window: str

    @field_validator("evaluation_window")
    @classmethod
    def validate_eval_window(cls, value: str) -> str:
        if value not in ALLOWED_EVALUATION_WINDOWS:
            raise ValueError(f"evaluation window must be one of {ALLOWED_EVALUATION_WINDOWS}")
        return value


class LongTermJudgementRecord(SlimModel):
    judgement_id: str
    user_id: str
    run_id: str
    run_date: date
    due_date: date
    judgement: JudgementItem
    initial_feedback: DailyJudgementFeedback
    cycle_evidence: list[dict] = Field(default_factory=list)
    final_score: float | None = None
    final_commentary: str | None = None
    final_outcome: str | None = None
    prompt_improvement_refs: list[str] = Field(default_factory=list)
    status: Literal["tracking", "due", "closed"] = "tracking"
    updated_at: datetime = Field(default_factory=utc_now)


def compute_due_date(start: date, window: str) -> date:
    if window == "1 day":
        return start + timedelta(days=1)
    if window == "1 week":
        return start + timedelta(days=7)
    if window == "1 month":
        return start + timedelta(days=30)
    if window == "3 months":
        return start + timedelta(days=90)
    if window == "1 year":
        return start + timedelta(days=365)
    raise ValueError(f"Unsupported window: {window}")
