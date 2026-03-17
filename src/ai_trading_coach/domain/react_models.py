"""Structured models for ReAct-style research stage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_trading_coach.domain.models import EvidenceItem


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReActStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_index: int = Field(ge=1)
    thought: str = Field(default="")
    action: str = Field(default="")
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = Field(default="")
    success: bool = Field(default=True)
    error_message: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None


class ResearchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research_id: str
    investigation_summary: str = Field(default="")
    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    collected_evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_steps: list[ReActStep] = Field(default_factory=list)
