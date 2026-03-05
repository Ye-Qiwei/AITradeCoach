"""Module interface definitions (contracts-first)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_trading_coach.domain.contracts import (
    CognitionExtractionInput,
    CognitionExtractionOutput,
    ContextBuildInput,
    ContextBuildOutput,
    EvaluatorInput,
    EvaluatorOutput,
    EvidencePlanningInput,
    EvidencePlanningOutput,
    LedgerInput,
    LedgerOutput,
    LogIntakeInput,
    LogIntakeOutput,
    MCPGatewayInput,
    MCPGatewayOutput,
    MemoryRecallOutput,
    MemoryRecallQuery,
    MemoryWriteInput,
    MemoryWriteOutput,
    PromptOpsInput,
    PromptOpsOutput,
    ReportGeneratorInput,
    ReportGeneratorOutput,
    WindowSelectorInput,
    WindowSelectorOutput,
)
from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult


@runtime_checkable
class SystemOrchestrator(Protocol):
    def run(self, request: ReviewRunRequest) -> TaskResult:
        """Execute end-to-end daily review workflow."""


@runtime_checkable
class DailyLogIntakeCanonicalizer(Protocol):
    def ingest(self, data: LogIntakeInput) -> LogIntakeOutput:
        """Parse and normalize raw daily logs."""


@runtime_checkable
class TradeLedgerPositionEngine(Protocol):
    def rebuild(self, data: LedgerInput) -> LedgerOutput:
        """Reconstruct ledger, position snapshot and pnl."""


@runtime_checkable
class CognitionExtractionEngine(Protocol):
    def extract(self, data: CognitionExtractionInput) -> CognitionExtractionOutput:
        """Extract cognition state and hypotheses from normalized logs."""


@runtime_checkable
class LongTermMemoryService(Protocol):
    def recall(self, query: MemoryRecallQuery) -> MemoryRecallOutput:
        """Retrieve relevant long-term memories."""

    def write(self, data: MemoryWriteInput) -> MemoryWriteOutput:
        """Write deduplicated memory records."""


@runtime_checkable
class ShortTermContextBuilder(Protocol):
    def build(self, data: ContextBuildInput) -> ContextBuildOutput:
        """Assemble execution context for current run."""


@runtime_checkable
class EvidencePlanner(Protocol):
    def plan(self, data: EvidencePlanningInput) -> EvidencePlanningOutput:
        """Create evidence plan before tool calls."""


@runtime_checkable
class MCPToolGateway(Protocol):
    def collect(self, data: MCPGatewayInput) -> MCPGatewayOutput:
        """Collect and normalize external evidence via MCP servers."""


@runtime_checkable
class DynamicAnalysisWindowSelector(Protocol):
    def select(self, data: WindowSelectorInput) -> WindowSelectorOutput:
        """Select analysis windows with explicit rationale."""


@runtime_checkable
class CognitionRealityEvaluator(Protocol):
    def evaluate(self, data: EvaluatorInput) -> EvaluatorOutput:
        """Evaluate cognition against market reality."""


@runtime_checkable
class ReviewReportGenerator(Protocol):
    def generate(self, data: ReportGeneratorInput) -> ReportGeneratorOutput:
        """Generate final daily review report."""


@runtime_checkable
class PromptOpsSelfImprovementEngine(Protocol):
    def propose(self, data: PromptOpsInput) -> PromptOpsOutput:
        """Generate offline-evaluable improvement proposals."""
