"""State schema for daily LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, ParserOutput, ResearchOutput
from ai_trading_coach.domain.models import EvidencePacket, ReviewRunRequest, TaskResult


class OrchestratorGraphState(TypedDict, total=False):
    agent_messages: list[str]
    request: ReviewRunRequest
    parse_result: ParserOutput
    evidence_packet: EvidencePacket
    research_output: ResearchOutput
    report_context: dict[str, Any]
    report_draft: str
    judgement_feedback: list[DailyJudgementFeedback]
    judge_verdict: JudgeVerdict
    rewrite_instruction: str | None
    rewrite_count: int
    model_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    react_steps: list[dict[str, Any]]
    final_result: TaskResult
