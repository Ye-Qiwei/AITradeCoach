"""Agent-first orchestration models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import (
    CognitionState,
    DailyLogNormalized,
    ExtensibleModel,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CombinedParseResult(ExtensibleModel):
    """Single-pass LLM parse result containing log normalization + cognition state."""

    normalized_log: DailyLogNormalized
    cognition_state: CognitionState


class StopCondition(ExtensibleModel):
    condition: str = Field(..., min_length=1)
    should_stop_when: str = Field(..., min_length=1)


class PlanSubTask(ExtensibleModel):
    subtask_id: str
    objective: str = Field(..., min_length=1)
    tool_category: Literal["market_data", "news_search", "filings_financials", "macro_data"]
    evidence_type: EvidenceType
    query: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    stop_conditions: list[StopCondition] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    time_window: str | None = None


class Plan(ExtensibleModel):
    plan_id: str
    created_at: datetime = Field(default_factory=utc_now)
    subtasks: list[PlanSubTask] = Field(default_factory=list)
    risk_uncertainties: list[str] = Field(default_factory=list)
    follow_up_triggers: list[str] = Field(default_factory=list)


class ReporterDraft(ExtensibleModel):
    markdown: str = Field(..., min_length=20)




class ReporterOutput(ExtensibleModel):
    markdown: str = Field(..., min_length=20)
    judgement_feedback: list["DailyJudgementFeedback"] = Field(default_factory=list)


from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback

class JudgeVerdict(ExtensibleModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    rewrite_instruction: str | None = None
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class SubTaskExecutionTrace(ExtensibleModel):
    subtask_id: str
    tool_ref: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    latency_ms: int = Field(default=0, ge=0)
    success: bool = True
    error_message: str | None = None
    evidence_item_count: int = Field(default=0, ge=0)
