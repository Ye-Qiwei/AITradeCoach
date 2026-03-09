"""Schemas for daily judgement extraction, research, and long-term evaluation."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

from pydantic import Field, field_validator

from ai_trading_coach.domain.models import ExtensibleModel

ALLOWED_EVALUATION_WINDOWS = ("1 day", "1 week", "1 month", "3 months", "1 year")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TradeAction(ExtensibleModel):
    action_id: str = ""
    action: Literal["buy", "sell", "add", "reduce", "hold", "watch"]
    target_asset: str
    position_change: str | None = None
    action_time: str | None = None
    reason: str | None = None


class JudgementItem(ExtensibleModel):
    judgement_id: str
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
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_from_user_log: list[str] = Field(default_factory=list)
    implicitness: Literal["explicit", "implicit", "mixed"] = "explicit"
    related_actions: list[str] = Field(default_factory=list)
    related_non_actions: list[str] = Field(default_factory=list)
    estimated_horizon: str | None = None
    proposed_evaluation_window: str = "1 week"

    @field_validator("proposed_evaluation_window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        if value not in ALLOWED_EVALUATION_WINDOWS:
            raise ValueError(f"evaluation window must be one of {ALLOWED_EVALUATION_WINDOWS}")
        return value


class ParserOutput(ExtensibleModel):
    parse_id: str
    user_id: str
    run_date: date
    trade_actions: list[TradeAction] = Field(default_factory=list)
    explicit_judgements: list[JudgementItem] = Field(default_factory=list)
    implicit_judgements: list[JudgementItem] = Field(default_factory=list)
    opportunity_judgements: list[JudgementItem] = Field(default_factory=list)
    non_action_judgements: list[JudgementItem] = Field(default_factory=list)
    reflection_summary: list[str] = Field(default_factory=list)

    def all_judgements(self) -> list[JudgementItem]:
        return [
            *self.explicit_judgements,
            *self.implicit_judgements,
            *self.opportunity_judgements,
            *self.non_action_judgements,
        ]


class JudgementEvidence(ExtensibleModel):
    judgement_id: str
    evidence_item_ids: list[str] = Field(default_factory=list)
    support_signal: Literal["support", "oppose", "uncertain"] = "uncertain"
    sufficiency_reason: str = ""


class ResearchOutput(ExtensibleModel):
    research_id: str
    judgement_evidence: list[JudgementEvidence] = Field(default_factory=list)
    stop_reason: str = ""

    def validate_against(self, judgement_ids: set[str], evidence_ids: set[str]) -> None:
        seen: set[str] = set()
        for item in self.judgement_evidence:
            if item.judgement_id not in judgement_ids:
                raise ValueError(f"Unknown judgement_id in research_output: {item.judgement_id}")
            if item.judgement_id in seen:
                raise ValueError(f"Duplicate judgement_id in research_output: {item.judgement_id}")
            seen.add(item.judgement_id)
            unknown = [eid for eid in item.evidence_item_ids if eid not in evidence_ids]
            if unknown:
                raise ValueError(f"Unknown evidence_item_ids for {item.judgement_id}: {unknown}")
            if not item.sufficiency_reason.strip():
                raise ValueError(f"Missing sufficiency_reason for {item.judgement_id}")
        missing = sorted(judgement_ids - seen)
        if missing:
            raise ValueError(f"Missing judgements in research_output: {missing}")




class ResearchSynthesisOutput(ExtensibleModel):
    research_id: str
    judgement_evidence: list[JudgementEvidence] = Field(default_factory=list)
    stop_reason: str = ""


class DailyJudgementFeedback(ExtensibleModel):
    judgement_id: str
    initial_feedback: Literal["likely_correct", "likely_wrong", "insufficient_evidence", "high_uncertainty"]
    evidence_summary: str
    evaluation_window: str
    window_rationale: str
    followup_indicators: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("evaluation_window")
    @classmethod
    def validate_eval_window(cls, value: str) -> str:
        if value not in ALLOWED_EVALUATION_WINDOWS:
            raise ValueError(f"evaluation window must be one of {ALLOWED_EVALUATION_WINDOWS}")
        return value


class LongTermJudgementRecord(ExtensibleModel):
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
