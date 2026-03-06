"""LangChain built-in agent workflow orchestrator."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.orchestrator.system_orchestrator import PipelineOrchestrator


class _WorkflowInput(BaseModel):
    request_json: str = Field(description="Serialized ReviewRunRequest in JSON")


class LangChainAgentOrchestrator:
    """Recompose the pipeline with LangChain built-in agent chain."""

    def __init__(self, *, legacy_orchestrator: PipelineOrchestrator, chat_model: Any) -> None:
        self.legacy_orchestrator = legacy_orchestrator
        self.chat_model = chat_model
        self._agent = self._build_agent()

    def _build_agent(self):
        tool = StructuredTool.from_function(
            name="run_trading_review_pipeline",
            description=(
                "Run the full trading review workflow: parse -> plan -> execute -> report -> judge. "
                "Input must be a JSON string of ReviewRunRequest fields."
            ),
            args_schema=_WorkflowInput,
            func=self._run_pipeline_tool,
        )
        return create_agent(
            model=self.chat_model,
            tools=[tool],
            system_prompt=(
                "You are the workflow controller. Call run_trading_review_pipeline exactly once, "
                "then return only the JSON payload returned by that tool."
            ),
        )

    def _run_pipeline_tool(self, request_json: str) -> str:
        payload = json.loads(request_json)
        request = ReviewRunRequest.model_validate(payload)
        result = self.legacy_orchestrator.run(request)
        return result.model_dump_json()

    def run(self, request: ReviewRunRequest) -> TaskResult:
        response = self._agent.invoke({"messages": [{"role": "user", "content": request.model_dump_json()}]})
        messages = response.get("messages", [])
        if not messages:
            raise ValueError("LangChain agent returned no messages")
        output = getattr(messages[-1], "content", "")
        return TaskResult.model_validate_json(str(output))


__all__ = ["LangChainAgentOrchestrator"]
