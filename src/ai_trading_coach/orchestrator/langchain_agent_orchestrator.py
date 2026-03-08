"""Graph-backed orchestrator using LangGraph as runtime."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


class LangChainAgentOrchestrator:
    """Orchestrator that invokes a compiled LangGraph state machine."""

    def __init__(self, *, compiled_graph: Any, chat_model: object | None = None) -> None:
        self.compiled_graph = compiled_graph
        self.chat_model = chat_model

    def run(self, request: ReviewRunRequest) -> TaskResult:
        initial_state: OrchestratorGraphState = {
            "request": request,
            "agent_messages": [],
            "rewrite_count": 0,
            "model_calls": [],
            "tool_calls": [],
            "errors": [],
        }
        output = self.compiled_graph.invoke(initial_state)
        return output["final_result"]

    def stream(self, request: ReviewRunRequest) -> Iterator[dict[str, Any]]:
        initial_state: OrchestratorGraphState = {
            "request": request,
            "agent_messages": [],
            "rewrite_count": 0,
            "model_calls": [],
            "tool_calls": [],
            "errors": [],
        }
        yield from self.compiled_graph.stream(initial_state)


__all__ = ["LangChainAgentOrchestrator"]
