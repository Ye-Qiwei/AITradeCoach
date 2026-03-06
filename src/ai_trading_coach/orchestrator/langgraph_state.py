"""LangGraph state schema for the single-path ReAct workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from ai_trading_coach.domain.agent_models import CombinedParseResult, JudgeVerdict, ReporterDraft
from ai_trading_coach.domain.models import EvidencePacket, ReviewRunRequest, TaskResult
from ai_trading_coach.domain.react_models import ResearchSummary


class OrchestratorGraphState(TypedDict, total=False):
    messages: list[BaseMessage]
    request: ReviewRunRequest
    parse_result: CombinedParseResult
    evidence_packet: EvidencePacket
    research_summary: ResearchSummary
    report_context: dict[str, Any]
    report_draft: ReporterDraft
    judge_verdict: JudgeVerdict
    rewrite_instruction: str | None
    rewrite_count: int
    model_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    errors: list[str]
    final_result: TaskResult


__all__ = ["OrchestratorGraphState"]
