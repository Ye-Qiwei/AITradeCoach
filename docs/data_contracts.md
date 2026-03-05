# 领域数据模型契约

> 完整字段定义见 `src/ai_trading_coach/domain/models.py`。以下为实现级契约摘要（字段、类型、必填、约束、扩展位）。

## 通用约定
- 所有模型都包含 `extensions: dict[str, Any]` 作为可扩展字段。
- 所有模型启用 `extra=allow`，保证向后兼容。
- 时间字段优先使用 `date`（业务日）与 `datetime`（事件时间）。

## 1) DailyLogRaw
- `log_id: str` 必填，唯一原始日志 ID
- `user_id: str` 必填
- `source_type: SourceType` 选填，默认 `markdown`
- `source_path: str | None` 选填
- `content: str` 必填，`min_length=1`
- `ingested_at: datetime` 选填，默认当前 UTC
- `metadata: dict[str, Any]` 选填

## 2) DailyLogNormalized
- `log_id: str` 必填
- `user_id: str` 必填
- `log_date: date` 必填
- `traded_tickers: list[str]` 选填
- `mentioned_tickers: list[str]` 选填
- `user_state: UserState` 选填
- `market_context: MarketContext` 选填
- `trade_events: list[TradeEvent]` 选填
- `trade_narratives: list[TradeNarrative]` 选填（原文保留）
- `scan_signals: ScanSignals` 选填
- `reflection: ReflectionBlock` 选填
- `ai_directives: list[str]` 选填
- `raw_text: str` 必填
- `field_errors: list[FieldError]` 选填（字段级错误）

## 3) TradeEvent
- `event_id: str` 必填
- `user_id: str` 必填
- `trade_date: date` 必填
- `ticker: str` 必填
- `asset_type: AssetType` 选填
- `side: TradeSide` 必填（BUY/SELL）
- `quantity: float` 必填，`>0`
- `unit_price: float | None` 条件必填（BUY 必填）
- `fees: float` 选填，`>=0`
- `fill_ratio: float` 选填，`0~1`
- `reason/source_tags/trigger/moment_emotion/risk_note` 选填

## 4) TradeLedger
- `ledger_id: str` 必填
- `user_id: str` 必填
- `as_of_date: date` 必填
- `events: list[TradeEvent]` 必填（可空列表）
- `open_positions: list[PositionHolding]` 选填
- `closed_positions: list[PositionHolding]` 选填
- `missing_price_tickers: list[str]` 选填
- `outcome_candidates: list[TradeOutcomeCandidate]` 选填

## 5) PositionSnapshot
- `snapshot_id: str` 必填
- `user_id: str` 必填
- `as_of_date: date` 必填
- `holdings: list[PositionHolding]` 必填（可空列表）
- `total_market_value/total_cost_basis/cash_proxy_balance: float | None` 选填
- `exposure_by_asset: dict[str,float]` 选填

## 6) PnLSnapshot
- `snapshot_id: str` 必填
- `user_id: str` 必填
- `as_of_date: date` 必填
- `currency: str` 选填
- `realized_pnl: float` 选填
- `unrealized_pnl/total_pnl: float | None` 选填
- `by_ticker: list[TickerPnL]` 选填
- `missing_price_tickers: list[str]` 选填

## 7) CognitionState
- `cognition_id: str` 必填
- `log_id: str` 必填
- `user_id: str` 必填
- `as_of_date: date` 必填
- `core_judgements: list[str]` 选填
- `hypotheses: list[Hypothesis]` 选填
- `risk_concerns/outside_opportunities/deliberate_no_trade_decisions` 选填
- `explicit_rules/fuzzy_tendencies` 选填
- `fact_statements/subjective_statements` 选填
- `behavioral_signals/emotion_signals/user_intent_signals` 选填

## 8) Hypothesis
- `hypothesis_id: str` 必填
- `statement: str` 必填，`min_length=3`
- `hypothesis_type: HypothesisType` 选填
- `related_tickers: list[str]` 选填
- `timeframe_hint: str | None` 选填
- `evidence_for/evidence_against/falsifiable_signals: list[str]` 选填
- `status: HypothesisStatus` 选填
- `confidence: float` 选填，`0~1`

## 9) BehavioralSignal
- `signal_id: str` 必填
- `signal_type: str` 必填
- `intensity: float` 必填，`0~1`
- `polarity: str` 选填
- `evidence: str` 必填

## 10) MemoryRecord
- `memory_id: str` 必填
- `user_id: str` 必填
- `memory_type: MemoryType` 必填
- `source_date: date | None` 选填
- `tickers/emotion_tags/keywords: list[str]` 选填
- `regime: str | None` 选填
- `quality_score/importance/confidence: float` 选填，`0~1`
- `document_text: str` 必填
- `structured_payload: dict[str,Any]` 选填
- `status: MemoryStatus` 选填（active/archived/invalidated）
- `version: int` 选填，`>=1`
- `created_at/updated_at: datetime` 选填

## 11) EvidencePlan
- `plan_id: str` 必填
- `user_id: str` 必填
- `generated_at: datetime` 选填
- `needs: list[EvidenceNeed]` 必填（可空列表）
- `priority_order: list[str]` 选填（need_id 顺序）
- `requires_event_centered_analysis/requires_analog_history: bool` 选填
- `planner_notes: list[str]` 选填

## 12) EvidencePacket
- `packet_id: str` 必填
- `user_id: str` 必填
- `collected_at: datetime` 选填
- `price_evidence/news_evidence/filing_evidence/sentiment_evidence/market_regime_evidence/discussion_evidence/macro_evidence/analog_evidence: list[EvidenceItem]`
- `source_registry: list[SourceAttribution]` 选填
- `completeness_score: float` 选填，`0~1`
- `missing_requirements: list[str]` 选填

## 13) WindowDecision
- `decision_id: str` 必填
- `selected_windows: list[WindowChoice]` 必填
- `rejected_windows: list[WindowRejected]` 选填
- `selection_reason: list[str]` 选填
- `judgement_type: JudgementType` 选填
- `follow_up_needed: bool` 选填
- `recommended_next_review_date: date | None` 选填
- `confidence: float` 选填，`0~1`

## 14) EvaluationResult
- `evaluation_id: str` 必填
- `user_id: str` 必填
- `as_of_date: date` 必填
- `summary: str` 必填
- `hypothesis_assessments: list[HypothesisAssessment]` 选填
- `bias_findings: list[BiasFinding]` 选填
- `strengths/mistakes/ahead_of_market_observations/follow_up_signals/warning_flags: list[str]` 选填
- `execution_assessment: ExecutionAssessment` 选填

## 15) DailyReviewReport
- `report_id: str` 必填
- `user_id: str` 必填
- `report_date: date` 必填
- `title: str` 选填
- `sections: list[ReportSection]` 选填
- `key_takeaways/next_watchlist/strategy_adjustments/risk_alerts: list[str]` 选填
- `generated_prompt_version: str` 必填
- `markdown_body: str` 必填

## 16) ImprovementProposal
- `proposal_id: str` 必填
- `generated_at: datetime` 选填
- `scope: ImprovementScope` 必填
- `problem_statement/candidate_change/expected_benefit/offline_eval_plan: str` 必填
- `risk_level: int` 选填，`1~5`
- `success_metrics: list[str]` 选填
- `status: ProposalStatus` 选填

## 17) RunTrace
- `run_id/user_id: str` 必填
- `run_date: date` 必填
- `trigger_type: TriggerType` 必填
- `started_at/ended_at: datetime` 选填
- `module_spans: list[ModuleRunSpan]` 选填
- `model_calls: list[ModelCallTrace]` 选填
- `tool_calls: list[ToolCallTrace]` 选填
- `window_decisions: list[WindowDecision]` 选填
- `evidence_sources: list[SourceAttribution]` 选填
- `report_version: str | None` 选填
- `debug_context: dict[str, Any]` 选填

## 18) TaskResult
- `run_id: str` 必填
- `status: RunStatus` 必填
- `step_results: list[StepResult]` 选填
- `report/evaluation/position_snapshot/pnl_snapshot` 选填
- `memory_write_results: list[MemoryWriteResult]` 选填
- `improvement_proposals: list[ImprovementProposal]` 选填
- `trace: RunTrace | None` 选填
- `errors: list[ErrorRecord]` 选填
