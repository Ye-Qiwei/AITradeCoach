"""Module-level input/output contracts."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import Field

from .models import (
    CognitionState,
    DailyLogNormalized,
    DailyLogRaw,
    DailyReviewReport,
    EvaluationResult,
    EvidencePacket,
    EvidencePlan,
    ExecutionContext,
    ImprovementBundle,
    MemoryRecord,
    MemoryWriteBatch,
    PnLSnapshot,
    PositionSnapshot,
    ReplayCase,
    ReplayEvaluationResult,
    ReplayPrediction,
    ReportQualityScore,
    RelevantMemorySet,
    TradeEvent,
    TradeLedger,
    WindowDecision,
)
from .models import ExtensibleModel


class LogIntakeInput(ExtensibleModel):
    user_id: str
    run_date: date
    raw_log_text: str
    source_path: str | None = None


class LogIntakeOutput(ExtensibleModel):
    raw: DailyLogRaw
    normalized: DailyLogNormalized


class LedgerInput(ExtensibleModel):
    user_id: str
    run_date: date
    historical_events: list[TradeEvent] = Field(default_factory=list)
    todays_events: list[TradeEvent] = Field(default_factory=list)
    latest_prices: dict[str, float] = Field(default_factory=dict)


class LedgerOutput(ExtensibleModel):
    ledger: TradeLedger
    position_snapshot: PositionSnapshot
    pnl_snapshot: PnLSnapshot


class CognitionExtractionInput(ExtensibleModel):
    normalized_log: DailyLogNormalized


class CognitionExtractionOutput(ExtensibleModel):
    cognition_state: CognitionState


class MemoryRecallQuery(ExtensibleModel):
    user_id: str
    date_from: date | None = None
    date_to: date | None = None
    tickers: list[str] = Field(default_factory=list)
    regime: str | None = None
    emotion_tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    top_k: int = Field(default=12, ge=1, le=100)


class MemoryRecallOutput(ExtensibleModel):
    relevant_memories: RelevantMemorySet


class MemoryWriteInput(ExtensibleModel):
    user_id: str
    batch: MemoryWriteBatch


class MemoryWriteOutput(ExtensibleModel):
    written_memory_ids: list[str] = Field(default_factory=list)
    dedup_count: int = 0
    merged_count: int = 0


class ContextBuildInput(ExtensibleModel):
    normalized_log: DailyLogNormalized
    cognition_state: CognitionState
    relevant_memories: RelevantMemorySet
    evidence_requirements: list[str] = Field(default_factory=list)
    task_goals: list[str] = Field(default_factory=list)


class ContextBuildOutput(ExtensibleModel):
    execution_context: ExecutionContext


class EvidencePlanningInput(ExtensibleModel):
    cognition_state: CognitionState
    active_theses: list[MemoryRecord] = Field(default_factory=list)
    relevant_history: RelevantMemorySet = Field(default_factory=RelevantMemorySet)
    task_goals: list[str] = Field(default_factory=list)


class EvidencePlanningOutput(ExtensibleModel):
    plan: EvidencePlan


class MCPGatewayInput(ExtensibleModel):
    plan: EvidencePlan


class MCPGatewayOutput(ExtensibleModel):
    packet: EvidencePacket


class WindowSelectorInput(ExtensibleModel):
    plan: EvidencePlan
    cognition_state: CognitionState
    trade_ledger: TradeLedger
    position_snapshot: PositionSnapshot
    market_volatility_state: str | None = None
    event_timestamps: list[date] = Field(default_factory=list)
    holding_period_days: int | None = None
    thesis_type_hint: str | None = None
    evidence_completeness: float | None = None


class WindowSelectorOutput(ExtensibleModel):
    decision: WindowDecision


class EvaluatorInput(ExtensibleModel):
    cognition_state: CognitionState
    evidence_packet: EvidencePacket
    window_decision: WindowDecision
    relevant_memories: RelevantMemorySet
    position_snapshot: PositionSnapshot


class EvaluatorOutput(ExtensibleModel):
    evaluation: EvaluationResult


class ReportGeneratorInput(ExtensibleModel):
    evaluation: EvaluationResult
    position_snapshot: PositionSnapshot
    pnl_snapshot: PnLSnapshot
    evidence_packet: EvidencePacket
    window_decision: WindowDecision
    user_focus_points: list[str] = Field(default_factory=list)


class ReportGeneratorOutput(ExtensibleModel):
    report: DailyReviewReport


class PromptOpsInput(ExtensibleModel):
    evaluation: EvaluationResult
    report: DailyReviewReport
    run_metrics: dict[str, Any] = Field(default_factory=dict)
    failure_cases: list[str] = Field(default_factory=list)
    active_prompt_versions: dict[str, str] = Field(default_factory=dict)
    replay_cases: list[ReplayCase] = Field(default_factory=list)
    replay_predictions: list[ReplayPrediction] = Field(default_factory=list)


class PromptOpsOutput(ExtensibleModel):
    bundle: ImprovementBundle
    report_quality: ReportQualityScore | None = None
    replay_result: ReplayEvaluationResult | None = None
