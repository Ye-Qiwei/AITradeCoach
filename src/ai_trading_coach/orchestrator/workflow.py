"""Workflow definitions for the daily review pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_coach.domain.enums import ModuleName


@dataclass(frozen=True)
class WorkflowStep:
    index: int
    name: ModuleName
    description: str


DAILY_REVIEW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(1, ModuleName.LOG_INTAKE, "LLM combined parse: log normalization + cognition state"),
    WorkflowStep(2, ModuleName.EVIDENCE_PLANNER, "Planner agent creates MCP subtask plan"),
    WorkflowStep(3, ModuleName.MCP_GATEWAY, "Executor runs MCP subtasks in parallel"),
    WorkflowStep(4, ModuleName.REPORT_GENERATOR, "Reporter drafts markdown report with citations"),
    WorkflowStep(5, ModuleName.EVALUATOR, "Judge validates and triggers rewrite loop when needed"),
)
