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
    WorkflowStep(1, ModuleName.LOG_INTAKE, "Read and normalize today's log"),
    WorkflowStep(2, ModuleName.LEDGER_ENGINE, "Update ledger and infer positions/PnL"),
    WorkflowStep(3, ModuleName.COGNITION_ENGINE, "Extract cognition objects and hypotheses"),
    WorkflowStep(4, ModuleName.MEMORY_SERVICE, "Recall relevant long-term memories"),
    WorkflowStep(5, ModuleName.EVIDENCE_PLANNER, "Plan evidence requirements"),
    WorkflowStep(6, ModuleName.MCP_GATEWAY, "Collect external evidence via MCP"),
    WorkflowStep(7, ModuleName.WINDOW_SELECTOR, "Select dynamic analysis windows"),
    WorkflowStep(8, ModuleName.EVALUATOR, "Evaluate cognition vs market reality"),
    WorkflowStep(9, ModuleName.REPORT_GENERATOR, "Generate daily review report"),
    WorkflowStep(10, ModuleName.MEMORY_SERVICE, "Write high-value memories"),
    WorkflowStep(11, ModuleName.PROMPTOPS, "Generate controlled improvement proposals"),
)
