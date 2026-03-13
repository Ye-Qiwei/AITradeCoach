"""Core domain schemas for AI Trading Cognitive Coach."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    AnalysisWindowType,
    AssetType,
    BiasType,
    EvaluationCategory,
    EvidenceType,
    HypothesisStatus,
    HypothesisType,
    ImprovementScope,
    JudgementType,
    MemoryStatus,
    MemoryType,
    ModelCallPurpose,
    ModuleName,
    ProposalStatus,
    RunStatus,
    SourceType,
    TradeSide,
    TriggerType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CoachBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, validate_assignment=True)


class ExtensibleModel(CoachBaseModel):
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Extension slot for non-breaking custom fields.",
    )


class RunOptions(ExtensibleModel):
    force_rerun: bool = Field(default=False, description="Whether to bypass idempotency checks.")
    partial_modules: list[ModuleName] = Field(
        default_factory=list,
        description="Run only selected modules when provided.",
    )
    dry_run: bool = Field(default=False, description="If true, do not write memory or files.")
    debug_mode: bool = Field(default=False, description="If true, capture verbose trace and context snapshots.")


class ReviewRunRequest(ExtensibleModel):
    run_id: str = Field(..., description="Unique run identifier.")
    user_id: str = Field(..., description="User identity for multi-tenant isolation.")
    run_date: date = Field(..., description="Business date of this review run.")
    trigger_type: TriggerType = Field(..., description="Run trigger source.")
    raw_log_text: str = Field(..., min_length=1, description="Original daily log text.")
    options: RunOptions = Field(default_factory=RunOptions)


class FieldError(ExtensibleModel):
    field: str = Field(..., description="Field path with parsing issue.")
    message: str = Field(..., description="Human-readable parsing error.")
    severity: str = Field(default="warning", description="warning|error")


class UserState(ExtensibleModel):
    emotion: str | None = Field(default=None, description="Primary emotion from log.")
    stress: int | None = Field(default=None, ge=0, le=10, description="Stress level 0-10.")
    focus: int | None = Field(default=None, ge=0, le=10, description="Focus level 0-10.")


class MarketContext(ExtensibleModel):
    regime: str | None = Field(default=None, description="User-perceived market regime.")
    key_variables: list[str] = Field(default_factory=list, description="Macro or regime drivers.")


class TradeNarrative(ExtensibleModel):
    raw_line: str = Field(..., description="Original line in user journal.")
    parsed: bool = Field(default=True, description="Whether parser transformed this line into TradeEvent.")


class ScanSignals(ExtensibleModel):
    anxiety: list[str] = Field(default_factory=list, description="Anxiety signals in log.")
    fomo: list[str] = Field(default_factory=list, description="FOMO or outside opportunities.")
    not_trade: list[str] = Field(default_factory=list, description="Deliberate no-trade decisions.")


class ReflectionBlock(ExtensibleModel):
    facts: list[str] = Field(default_factory=list, description="Fact statements from reflection section.")
    gaps: list[str] = Field(default_factory=list, description="Expectation vs outcome gaps.")
    lessons: list[str] = Field(default_factory=list, description="Rules or lessons stated by user.")


class DailyLogRaw(ExtensibleModel):
    log_id: str = Field(..., description="Raw log identifier.")
    user_id: str = Field(..., description="Owner of the log.")
    source_type: SourceType = Field(default=SourceType.MARKDOWN, description="Input source format.")
    source_path: str | None = Field(default=None, description="Optional local file path.")
    content: str = Field(..., min_length=1, description="Raw unmodified log content.")
    ingested_at: datetime = Field(default_factory=utc_now, description="Ingestion timestamp UTC.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source metadata.")


class TradeEvent(ExtensibleModel):
    event_id: str = Field(..., description="Unique trade event id.")
    user_id: str = Field(..., description="Trade owner.")
    trade_date: date = Field(..., description="Trade execution date.")
    ticker: str = Field(..., min_length=1, description="Ticker symbol.")
    asset_type: AssetType = Field(default=AssetType.STOCK, description="Asset class.")
    side: TradeSide = Field(..., description="BUY or SELL.")
    quantity: float = Field(..., gt=0, description="Executed quantity.")
    unit_price: float | None = Field(default=None, gt=0, description="Execution unit price.")
    currency: str = Field(default="USD", description="Price currency code.")
    fees: float = Field(default=0.0, ge=0, description="Fees and slippage cost.")
    fill_ratio: float = Field(default=1.0, ge=0, le=1, description="Partial fill ratio [0,1].")
    reason: str | None = Field(default=None, description="User rationale.")
    source_tags: list[str] = Field(default_factory=list, description="Evidence source tags.")
    trigger: str | None = Field(default=None, description="Execution trigger condition.")
    moment_emotion: str | None = Field(default=None, description="Emotion at execution moment.")
    risk_note: str | None = Field(default=None, description="Risk and stop notes.")

    @model_validator(mode="after")
    def validate_price(self) -> "TradeEvent":
        if self.side == TradeSide.BUY and self.unit_price is None:
            raise ValueError("BUY trade event must include unit_price")
        return self


class DailyLogNormalized(ExtensibleModel):
    log_id: str = Field(..., description="Reference to raw log id.")
    user_id: str = Field(..., description="User id.")
    log_date: date = Field(..., description="Date in normalized form.")
    traded_tickers: list[str] = Field(default_factory=list, description="Tickers with actual trades.")
    mentioned_tickers: list[str] = Field(default_factory=list, description="Tickers mentioned in narrative.")
    user_state: UserState = Field(default_factory=UserState)
    market_context: MarketContext = Field(default_factory=MarketContext)
    trade_events: list[TradeEvent] = Field(default_factory=list, description="Parsed structured trade events.")
    trade_narratives: list[TradeNarrative] = Field(
        default_factory=list,
        description="Original trade lines preserved for traceability.",
    )
    scan_signals: ScanSignals = Field(default_factory=ScanSignals)
    reflection: ReflectionBlock = Field(default_factory=ReflectionBlock)
    ai_directives: list[str] = Field(default_factory=list, description="@AI directives extracted from log.")
    raw_text: str = Field(..., min_length=1, description="Original text preserved.")
    field_errors: list[FieldError] = Field(default_factory=list, description="Field-level parsing issues.")


class PositionLot(ExtensibleModel):
    lot_id: str = Field(..., description="Lot id generated from buy event.")
    ticker: str = Field(..., description="Ticker symbol.")
    entry_date: date = Field(..., description="Lot entry date.")
    quantity_open: float = Field(..., ge=0, description="Remaining open quantity in this lot.")
    cost_basis_per_unit: float = Field(..., gt=0, description="Entry price per unit for this lot.")


class PositionHolding(ExtensibleModel):
    ticker: str = Field(..., description="Ticker symbol.")
    asset_type: AssetType = Field(default=AssetType.STOCK)
    quantity: float = Field(..., ge=0, description="Current holding quantity.")
    avg_cost: float = Field(..., ge=0, description="Weighted average cost.")
    market_price: float | None = Field(default=None, ge=0, description="Latest known market price.")
    market_value: float | None = Field(default=None, description="Quantity * market_price.")
    unrealized_pnl: float | None = Field(default=None, description="Current unrealized pnl.")
    holding_period_days: int | None = Field(default=None, ge=0, description="Holding period in days.")
    lots: list[PositionLot] = Field(default_factory=list, description="Underlying lots for FIFO accounting.")


class TradeOutcomeCandidate(ExtensibleModel):
    ticker: str = Field(..., description="Ticker symbol.")
    direction: TradeSide = Field(..., description="Trade side of candidate evaluation.")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for inferred outcome.")
    summary: str = Field(..., description="Short outcome statement.")


class TradeLedger(ExtensibleModel):
    ledger_id: str = Field(..., description="Ledger id for this user and date.")
    user_id: str = Field(..., description="Ledger owner.")
    as_of_date: date = Field(..., description="Ledger as-of date.")
    events: list[TradeEvent] = Field(default_factory=list, description="All known trade events.")
    open_positions: list[PositionHolding] = Field(default_factory=list, description="Open holdings.")
    closed_positions: list[PositionHolding] = Field(default_factory=list, description="Closed position summaries.")
    missing_price_tickers: list[str] = Field(
        default_factory=list,
        description="Tickers with no valid latest price and degraded metrics.",
    )
    outcome_candidates: list[TradeOutcomeCandidate] = Field(default_factory=list)


class PositionSnapshot(ExtensibleModel):
    snapshot_id: str = Field(..., description="Position snapshot id.")
    user_id: str = Field(..., description="User id.")
    as_of_date: date = Field(..., description="Snapshot date.")
    holdings: list[PositionHolding] = Field(default_factory=list)
    total_market_value: float | None = Field(default=None, description="Aggregate market value.")
    total_cost_basis: float | None = Field(default=None, description="Aggregate cost basis.")
    exposure_by_asset: dict[str, float] = Field(default_factory=dict, description="Asset type -> notional.")
    cash_proxy_balance: float | None = Field(default=None, description="Estimated cash proxy if available.")


class TickerPnL(ExtensibleModel):
    ticker: str = Field(...)
    realized_pnl: float = Field(default=0.0)
    unrealized_pnl: float | None = Field(default=None)


class PnLSnapshot(ExtensibleModel):
    snapshot_id: str = Field(...)
    user_id: str = Field(...)
    as_of_date: date = Field(...)
    currency: str = Field(default="USD")
    realized_pnl: float = Field(default=0.0)
    unrealized_pnl: float | None = Field(default=None)
    total_pnl: float | None = Field(default=None)
    by_ticker: list[TickerPnL] = Field(default_factory=list)
    missing_price_tickers: list[str] = Field(default_factory=list)


class EmotionSignal(ExtensibleModel):
    signal_id: str = Field(..., description="Emotion signal id.")
    emotion: str = Field(..., description="Emotion label.")
    intensity: float = Field(..., ge=0, le=1, description="Normalized intensity.")
    evidence: str = Field(..., description="Source excerpt or cue.")


class BehavioralSignal(ExtensibleModel):
    signal_id: str = Field(..., description="Behavioral signal id.")
    signal_type: str = Field(..., description="Execution discipline, hesitation, fomo, etc.")
    intensity: float = Field(..., ge=0, le=1)
    polarity: str = Field(default="neutral", description="positive|negative|neutral")
    evidence: str = Field(..., description="Grounding text.")


class UserIntentSignal(ExtensibleModel):
    intent_id: str = Field(...)
    question: str = Field(..., description="Explicit AI question from user.")
    priority: int = Field(default=3, ge=1, le=5)


class Hypothesis(ExtensibleModel):
    hypothesis_id: str = Field(..., description="Hypothesis identifier.")
    statement: str = Field(..., min_length=3, description="Falsifiable hypothesis statement.")
    hypothesis_type: HypothesisType = Field(default=HypothesisType.OTHER)
    related_tickers: list[str] = Field(default_factory=list)
    timeframe_hint: str | None = Field(default=None, description="Expected verification horizon.")
    evidence_for: list[str] = Field(default_factory=list, description="Supporting reasons from user log.")
    evidence_against: list[str] = Field(default_factory=list, description="Known contradictions.")
    falsifiable_signals: list[str] = Field(default_factory=list, description="Signals that could invalidate it.")
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING)
    confidence: float = Field(default=0.5, ge=0, le=1)


class CognitionState(ExtensibleModel):
    cognition_id: str = Field(..., description="Cognition extraction id.")
    log_id: str = Field(..., description="Source normalized log id.")
    user_id: str = Field(...)
    as_of_date: date = Field(...)
    core_judgements: list[str] = Field(default_factory=list, description="Core explicit judgements.")
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    risk_concerns: list[str] = Field(default_factory=list)
    outside_opportunities: list[str] = Field(default_factory=list)
    deliberate_no_trade_decisions: list[str] = Field(default_factory=list)
    explicit_rules: list[str] = Field(default_factory=list, description="Explicit rules from lessons.")
    fuzzy_tendencies: list[str] = Field(default_factory=list, description="Non-explicit behavior tendencies.")
    fact_statements: list[str] = Field(default_factory=list)
    subjective_statements: list[str] = Field(default_factory=list)
    behavioral_signals: list[BehavioralSignal] = Field(default_factory=list)
    emotion_signals: list[EmotionSignal] = Field(default_factory=list)
    user_intent_signals: list[UserIntentSignal] = Field(default_factory=list)


class MemoryRecord(ExtensibleModel):
    memory_id: str = Field(..., description="Global memory id.")
    user_id: str = Field(...)
    memory_type: MemoryType = Field(...)
    source_date: date | None = Field(default=None)
    tickers: list[str] = Field(default_factory=list)
    regime: str | None = Field(default=None)
    emotion_tags: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.5, ge=0, le=1)
    document_text: str = Field(..., description="Primary memory text.")
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    keywords: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class QueryTimeRange(ExtensibleModel):
    start_date: date | None = Field(default=None)
    end_date: date | None = Field(default=None)
    relative_window: str | None = Field(default=None, description="1D/5D/20D/etc.")


class EvidenceNeed(ExtensibleModel):
    need_id: str = Field(...)
    hypothesis_id: str | None = Field(default=None)
    claim: str = Field(..., description="Claim to validate.")
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    macro_variables: list[str] = Field(default_factory=list)
    query_range: QueryTimeRange = Field(default_factory=QueryTimeRange)
    priority: int = Field(default=3, ge=1, le=5)
    event_centered: bool = Field(default=False)
    analog_history: bool = Field(default=False)
    questions: list[str] = Field(default_factory=list, description="Questions evidence must answer.")


class EvidencePlan(ExtensibleModel):
    plan_id: str = Field(...)
    user_id: str = Field(...)
    generated_at: datetime = Field(default_factory=utc_now)
    needs: list[EvidenceNeed] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list, description="need_id sequence.")
    requires_event_centered_analysis: bool = Field(default=False)
    requires_analog_history: bool = Field(default=False)
    planner_notes: list[str] = Field(default_factory=list)


class SourceAttribution(ExtensibleModel):
    source_id: str | None = Field(default=None)
    source_type: str = Field(..., description="news_api/price_api/filing_api/etc.")
    provider: str = Field(..., description="Underlying MCP server or provider name.")
    uri: str | None = Field(default=None)
    title: str | None = Field(default=None)
    published_at: datetime | None = Field(default=None)
    fetched_at: datetime = Field(default_factory=utc_now)
    reliability_score: float = Field(default=0.5, ge=0, le=1)


class EvidenceItem(ExtensibleModel):
    item_id: str | None = Field(default=None)
    evidence_type: EvidenceType = Field(...)
    summary: str = Field(...)
    data: dict[str, Any] = Field(default_factory=dict)
    related_tickers: list[str] = Field(default_factory=list)
    event_time: datetime | None = Field(default=None)
    sources: list[SourceAttribution] = Field(default_factory=list)


class EvidencePacket(ExtensibleModel):
    packet_id: str = Field(...)
    user_id: str = Field(...)
    collected_at: datetime = Field(default_factory=utc_now)
    price_evidence: list[EvidenceItem] = Field(default_factory=list)
    news_evidence: list[EvidenceItem] = Field(default_factory=list)
    filing_evidence: list[EvidenceItem] = Field(default_factory=list)
    sentiment_evidence: list[EvidenceItem] = Field(default_factory=list)
    market_regime_evidence: list[EvidenceItem] = Field(default_factory=list)
    discussion_evidence: list[EvidenceItem] = Field(default_factory=list)
    macro_evidence: list[EvidenceItem] = Field(default_factory=list)
    analog_evidence: list[EvidenceItem] = Field(default_factory=list)
    source_registry: list[SourceAttribution] = Field(default_factory=list)
    completeness_score: float = Field(default=0.0, ge=0, le=1)
    missing_requirements: list[str] = Field(default_factory=list)


class WindowChoice(ExtensibleModel):
    window_type: AnalysisWindowType = Field(...)
    start_date: date | None = Field(default=None)
    end_date: date | None = Field(default=None)
    reason: str = Field(..., description="Why this window is selected.")
    target_questions: list[str] = Field(default_factory=list, description="Questions this window should answer.")
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_dates(self) -> "WindowChoice":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("window start_date cannot be after end_date")
        return self


class WindowRejected(ExtensibleModel):
    window_type: AnalysisWindowType = Field(...)
    reason: str = Field(...)


class WindowDecision(ExtensibleModel):
    decision_id: str = Field(...)
    selected_windows: list[WindowChoice] = Field(default_factory=list)
    rejected_windows: list[WindowRejected] = Field(default_factory=list)
    selection_reason: list[str] = Field(default_factory=list)
    judgement_type: JudgementType = Field(default=JudgementType.PRELIMINARY)
    follow_up_needed: bool = Field(default=False)
    recommended_next_review_date: date | None = Field(default=None)
    confidence: float = Field(default=0.5, ge=0, le=1)


class HypothesisAssessment(ExtensibleModel):
    hypothesis_id: str = Field(...)
    category: EvaluationCategory = Field(...)
    thesis_still_valid: bool = Field(default=True)
    market_in_verification_phase: bool = Field(default=True)
    support_evidence_ids: list[str] = Field(default_factory=list)
    weaken_evidence_ids: list[str] = Field(default_factory=list)
    commentary: str = Field(...)
    confidence: float = Field(default=0.5, ge=0, le=1)


class BiasFinding(ExtensibleModel):
    bias_type: BiasType = Field(...)
    severity: int = Field(default=3, ge=1, le=5)
    description: str = Field(...)
    evidence: list[str] = Field(default_factory=list)
    correction: str = Field(...)


class ExecutionAssessment(ExtensibleModel):
    discipline_score: float = Field(default=0.5, ge=0, le=1)
    position_sizing_score: float = Field(default=0.5, ge=0, le=1)
    risk_control_score: float = Field(default=0.5, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class EvaluationResult(ExtensibleModel):
    evaluation_id: str = Field(...)
    user_id: str = Field(...)
    as_of_date: date = Field(...)
    summary: str = Field(...)
    hypothesis_assessments: list[HypothesisAssessment] = Field(default_factory=list)
    bias_findings: list[BiasFinding] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    ahead_of_market_observations: list[str] = Field(default_factory=list)
    execution_assessment: ExecutionAssessment = Field(default_factory=ExecutionAssessment)
    follow_up_signals: list[str] = Field(default_factory=list)
    warning_flags: list[str] = Field(default_factory=list)


class ReportSection(ExtensibleModel):
    title: str = Field(...)
    content: str = Field(...)


class DailyReviewReport(ExtensibleModel):
    report_id: str = Field(...)
    user_id: str = Field(...)
    report_date: date = Field(...)
    title: str = Field(default="Daily Trading Cognition Review")
    sections: list[ReportSection] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    next_watchlist: list[str] = Field(default_factory=list)
    strategy_adjustments: list[str] = Field(default_factory=list)
    risk_alerts: list[str] = Field(default_factory=list)
    generated_prompt_version: str = Field(...)
    markdown_body: str = Field(...)


class ImprovementProposal(ExtensibleModel):
    proposal_id: str = Field(...)
    generated_at: datetime = Field(default_factory=utc_now)
    scope: ImprovementScope = Field(...)
    problem_statement: str = Field(...)
    candidate_change: str = Field(...)
    expected_benefit: str = Field(...)
    risk_level: int = Field(default=3, ge=1, le=5)
    offline_eval_plan: str = Field(...)
    success_metrics: list[str] = Field(default_factory=list)
    status: ProposalStatus = Field(default=ProposalStatus.PROPOSED)


class ReportQualityScore(ExtensibleModel):
    score_id: str = Field(...)
    overall_score: float = Field(default=0.0, ge=0, le=1)
    structure_score: float = Field(default=0.0, ge=0, le=1)
    evidence_traceability_score: float = Field(default=0.0, ge=0, le=1)
    actionability_score: float = Field(default=0.0, ge=0, le=1)
    symmetry_score: float = Field(default=0.0, ge=0, le=1)
    risk_tone_score: float = Field(default=0.0, ge=0, le=1)
    missing_sections: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReplayCase(ExtensibleModel):
    case_id: str = Field(...)
    user_id: str = Field(...)
    run_date: date = Field(...)
    raw_log_text: str = Field(..., min_length=1)
    expected_categories: list[EvaluationCategory] = Field(default_factory=list)
    expected_follow_up_needed: bool | None = Field(default=None)
    description: str | None = Field(default=None)


class ReplayPrediction(ExtensibleModel):
    case_id: str = Field(...)
    predicted_categories: list[EvaluationCategory] = Field(default_factory=list)
    predicted_follow_up_needed: bool | None = Field(default=None)
    notes: list[str] = Field(default_factory=list)


class ReplayCaseResult(ExtensibleModel):
    case_id: str = Field(...)
    expected_categories: list[EvaluationCategory] = Field(default_factory=list)
    predicted_categories: list[EvaluationCategory] = Field(default_factory=list)
    matched_categories: list[EvaluationCategory] = Field(default_factory=list)
    missed_categories: list[EvaluationCategory] = Field(default_factory=list)
    unexpected_categories: list[EvaluationCategory] = Field(default_factory=list)
    follow_up_match: bool | None = Field(default=None)
    score: float = Field(default=0.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class ReplayEvaluationResult(ExtensibleModel):
    replay_id: str = Field(...)
    evaluated_at: datetime = Field(default_factory=utc_now)
    case_results: list[ReplayCaseResult] = Field(default_factory=list)
    case_count: int = Field(default=0, ge=0)
    average_score: float = Field(default=0.0, ge=0, le=1)
    category_hit_rate: float = Field(default=0.0, ge=0, le=1)
    unexpected_category_rate: float = Field(default=0.0, ge=0, le=1)
    follow_up_accuracy: float | None = Field(default=None, ge=0, le=1)
    ahead_of_market_recall: float | None = Field(default=None, ge=0, le=1)
    recommendation: str = Field(default="")


class ModelCallTrace(ExtensibleModel):
    call_id: str = Field(...)
    purpose: ModelCallPurpose = Field(...)
    model_name: str = Field(...)
    provider: str | None = Field(default=None)
    prompt_version: str | None = Field(default=None)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = Field(default=None)
    input_summary: str = Field(...)
    output_summary: str = Field(...)
    token_in: int | None = Field(default=None, ge=0)
    token_out: int | None = Field(default=None, ge=0)
    response_size: int | None = Field(default=None, ge=0)
    error_message: str | None = Field(default=None)
    latency_ms: int | None = Field(default=None, ge=0)


class ToolCallTrace(ExtensibleModel):
    call_id: str = Field(...)
    tool_name: str = Field(...)
    server_id: str = Field(...)
    request_summary: str = Field(...)
    response_summary: str = Field(...)
    payload_hash: str | None = Field(default=None)
    latency_ms: int = Field(..., ge=0)
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)


class ModuleRunSpan(ExtensibleModel):
    module_name: ModuleName = Field(...)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = Field(default=None)
    duration_ms: int | None = Field(default=None, ge=0)
    status: RunStatus = Field(default=RunStatus.SUCCESS)
    notes: list[str] = Field(default_factory=list)


class MemoryWriteResult(ExtensibleModel):
    collection: str = Field(...)
    memory_ids: list[str] = Field(default_factory=list)
    dedup_applied: bool = Field(default=False)
    merge_applied: bool = Field(default=False)


class RunTrace(ExtensibleModel):
    run_id: str = Field(...)
    user_id: str = Field(...)
    run_date: date = Field(...)
    trigger_type: TriggerType = Field(...)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = Field(default=None)
    module_spans: list[ModuleRunSpan] = Field(default_factory=list)
    model_calls: list[ModelCallTrace] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    window_decisions: list[WindowDecision] = Field(default_factory=list)
    evidence_sources: list[SourceAttribution] = Field(default_factory=list)
    report_version: str | None = Field(default=None)
    rewrite_rounds: int = Field(default=0, ge=0)
    debug_context: dict[str, Any] = Field(default_factory=dict)
    react_steps: list[dict[str, Any]] = Field(default_factory=list)


class StepResult(ExtensibleModel):
    module_name: ModuleName = Field(...)
    status: RunStatus = Field(...)
    details: str | None = Field(default=None)


class ErrorRecord(ExtensibleModel):
    module_name: ModuleName = Field(...)
    error_code: str = Field(...)
    message: str = Field(...)
    recoverable: bool = Field(default=False)


class TaskResult(ExtensibleModel):
    run_id: str = Field(...)
    status: RunStatus = Field(...)
    step_results: list[StepResult] = Field(default_factory=list)
    report: DailyReviewReport | None = Field(default=None)
    evaluation: EvaluationResult | None = Field(default=None)
    position_snapshot: PositionSnapshot | None = Field(default=None)
    pnl_snapshot: PnLSnapshot | None = Field(default=None)
    memory_write_results: list[MemoryWriteResult] = Field(default_factory=list)
    improvement_proposals: list[ImprovementProposal] = Field(default_factory=list)
    trace: RunTrace | None = Field(default=None)
    errors: list[ErrorRecord] = Field(default_factory=list)


class RelevantMemorySet(ExtensibleModel):
    records: list[MemoryRecord] = Field(default_factory=list)
    retrieval_notes: list[str] = Field(default_factory=list)


class ExecutionContext(ExtensibleModel):
    today_input: DailyLogNormalized = Field(...)
    related_history: RelevantMemorySet = Field(default_factory=RelevantMemorySet)
    market_evidence: EvidencePacket | None = Field(default=None)
    task_goals: list[str] = Field(default_factory=list)


class MemoryWriteBatch(ExtensibleModel):
    records: list[MemoryRecord] = Field(default_factory=list)
    dedup_keys: list[str] = Field(default_factory=list)


class PromptVersionCandidate(ExtensibleModel):
    version_id: str = Field(...)
    prompt_name: str = Field(...)
    content: str = Field(...)
    rationale: str = Field(...)


class ContextPolicyCandidate(ExtensibleModel):
    policy_id: str = Field(...)
    description: str = Field(...)
    retrieval_rules: list[str] = Field(default_factory=list)


class EvaluationRubricCandidate(ExtensibleModel):
    rubric_id: str = Field(...)
    title: str = Field(...)
    scoring_rules: list[str] = Field(default_factory=list)


class ImprovementBundle(ExtensibleModel):
    proposal: ImprovementProposal
    prompt_candidate: PromptVersionCandidate | None = None
    context_policy_candidate: ContextPolicyCandidate | None = None
    rubric_candidate: EvaluationRubricCandidate | None = None
