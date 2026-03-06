"""Workflow definitions for the LangGraph ReAct daily review pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_coach.domain.enums import ModuleName


@dataclass(frozen=True)
class WorkflowStep:
    index: int
    name: ModuleName
    description: str


DAILY_REVIEW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(1, ModuleName.LOG_INTAKE, "Parser builds normalized log and cognition state"),
    WorkflowStep(2, ModuleName.MCP_GATEWAY, "ReAct research agent gathers MCP evidence with tool calling"),
    WorkflowStep(3, ModuleName.REPORT_GENERATOR, "Reporter drafts markdown report with citations"),
    WorkflowStep(4, ModuleName.EVALUATOR, "Judge validates draft and conditionally triggers rewrite"),
    WorkflowStep(5, ModuleName.ORCHESTRATOR, "Finalize TaskResult from LangGraph terminal node"),
)
