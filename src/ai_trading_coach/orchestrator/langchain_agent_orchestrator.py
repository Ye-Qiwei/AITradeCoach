"""Compatibility wrapper for pipeline orchestrator with ReAct research stage."""

from __future__ import annotations

from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.orchestrator.system_orchestrator import PipelineOrchestrator


class LangChainAgentOrchestrator:
    """Backward-compatible facade preserving run(request)->TaskResult contract."""

    def __init__(self, *, legacy_orchestrator: PipelineOrchestrator, chat_model: object | None = None) -> None:
        self.legacy_orchestrator = legacy_orchestrator
        self.chat_model = chat_model

    def run(self, request: ReviewRunRequest) -> TaskResult:
        return self.legacy_orchestrator.run(request)


__all__ = ["LangChainAgentOrchestrator"]
