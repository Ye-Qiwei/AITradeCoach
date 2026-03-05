"""System orchestrator for daily cognition review tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, TypeVar

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.contracts import (
    CognitionExtractionInput,
    ContextBuildInput,
    EvaluatorInput,
    EvidencePlanningInput,
    LedgerInput,
    LogIntakeInput,
    MCPGatewayInput,
    MemoryRecallQuery,
    MemoryWriteInput,
    PromptOpsInput,
    ReportGeneratorInput,
    WindowSelectorInput,
)
from ai_trading_coach.domain.enums import MemoryStatus, MemoryType, ModuleName, RunStatus
from ai_trading_coach.domain.models import (
    ErrorRecord,
    MemoryRecord,
    ModelCallTrace,
    MemoryWriteBatch,
    MemoryWriteResult,
    ModuleRunSpan,
    ReviewRunRequest,
    RunTrace,
    StepResult,
    TaskResult,
    ToolCallTrace,
)
from ai_trading_coach.interfaces.modules import (
    CognitionExtractionEngine,
    CognitionRealityEvaluator,
    DailyLogIntakeCanonicalizer,
    DynamicAnalysisWindowSelector,
    EvidencePlanner,
    LongTermMemoryService,
    MCPToolGateway,
    PromptOpsSelfImprovementEngine,
    ReviewReportGenerator,
    ShortTermContextBuilder,
    TradeLedgerPositionEngine,
)

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class OrchestratorModules:
    log_intake: DailyLogIntakeCanonicalizer
    ledger_engine: TradeLedgerPositionEngine
    cognition_engine: CognitionExtractionEngine
    memory_service: LongTermMemoryService
    context_builder: ShortTermContextBuilder
    evidence_planner: EvidencePlanner
    mcp_gateway: MCPToolGateway
    window_selector: DynamicAnalysisWindowSelector
    evaluator: CognitionRealityEvaluator
    report_generator: ReviewReportGenerator
    promptops_engine: PromptOpsSelfImprovementEngine


class PipelineOrchestrator:
    """Executes module chain with step trace, partial recovery, and controlled writeback."""

    def __init__(self, modules: OrchestratorModules) -> None:
        self.modules = modules

    def run(self, request: ReviewRunRequest) -> TaskResult:
        trace = RunTrace(
            run_id=request.run_id,
            user_id=request.user_id,
            run_date=request.run_date,
            trigger_type=request.trigger_type,
            started_at=utc_now(),
        )
        step_results: list[StepResult] = []
        module_spans: list[ModuleRunSpan] = []
        errors: list[ErrorRecord] = []

        intake_output = None
        ledger_output = None
        cognition_output = None
        recall_output = None
        context_output = None
        plan_output = None
        evidence_output = None
        window_output = None
        evaluator_output = None
        report_output = None
        promptops_output = None
        memory_write_results: list[MemoryWriteResult] = []
        model_call_traces: list[ModelCallTrace] = []

        try:
            intake_output = self._execute_step(
                module_name=ModuleName.LOG_INTAKE,
                action=lambda: self.modules.log_intake.ingest(
                    LogIntakeInput(
                        user_id=request.user_id,
                        run_date=request.run_date,
                        raw_log_text=request.raw_log_text,
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            ledger_output = self._execute_step(
                module_name=ModuleName.LEDGER_ENGINE,
                action=lambda: self.modules.ledger_engine.rebuild(
                    LedgerInput(
                        user_id=request.user_id,
                        run_date=request.run_date,
                        historical_events=[],
                        todays_events=intake_output.normalized.trade_events,
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            cognition_output = self._execute_step(
                module_name=ModuleName.COGNITION_ENGINE,
                action=lambda: self.modules.cognition_engine.extract(
                    CognitionExtractionInput(normalized_log=intake_output.normalized)
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            model_call_traces.extend(
                self._parse_model_call_traces(cognition_output.extensions.get("model_call_traces"))
            )

            recall_output = self._execute_step(
                module_name=ModuleName.MEMORY_SERVICE,
                action=lambda: self.modules.memory_service.recall(
                    MemoryRecallQuery(
                        user_id=request.user_id,
                        tickers=list(
                            {
                                *intake_output.normalized.traded_tickers,
                                *intake_output.normalized.mentioned_tickers,
                            }
                        ),
                        regime=intake_output.normalized.market_context.regime,
                        emotion_tags=[intake_output.normalized.user_state.emotion]
                        if intake_output.normalized.user_state.emotion
                        else [],
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            context_output = self._execute_step(
                module_name=ModuleName.CONTEXT_BUILDER,
                action=lambda: self.modules.context_builder.build(
                    ContextBuildInput(
                        normalized_log=intake_output.normalized,
                        cognition_state=cognition_output.cognition_state,
                        relevant_memories=recall_output.relevant_memories,
                        task_goals=["daily_cognition_review"],
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            plan_output = self._execute_step(
                module_name=ModuleName.EVIDENCE_PLANNER,
                action=lambda: self.modules.evidence_planner.plan(
                    EvidencePlanningInput(
                        cognition_state=cognition_output.cognition_state,
                        active_theses=self._extract_active_theses(recall_output.relevant_memories.records),
                        relevant_history=recall_output.relevant_memories,
                        task_goals=context_output.execution_context.task_goals,
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            evidence_output = self._execute_step(
                module_name=ModuleName.MCP_GATEWAY,
                action=lambda: self.modules.mcp_gateway.collect(MCPGatewayInput(plan=plan_output.plan)),
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.evidence_sources = evidence_output.packet.source_registry
            trace.tool_calls = self._parse_tool_call_traces(evidence_output.packet.extensions.get("tool_call_traces"))
            tool_failure_rate = self._tool_failure_rate(trace.tool_calls)

            window_output = self._execute_step(
                module_name=ModuleName.WINDOW_SELECTOR,
                action=lambda: self.modules.window_selector.select(
                    WindowSelectorInput(
                        plan=plan_output.plan,
                        cognition_state=cognition_output.cognition_state,
                        trade_ledger=ledger_output.ledger,
                        position_snapshot=ledger_output.position_snapshot,
                        evidence_completeness=evidence_output.packet.completeness_score,
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.window_decisions.append(window_output.decision)

            evaluator_output = self._execute_step(
                module_name=ModuleName.EVALUATOR,
                action=lambda: self.modules.evaluator.evaluate(
                    EvaluatorInput(
                        cognition_state=cognition_output.cognition_state,
                        evidence_packet=evidence_output.packet,
                        window_decision=window_output.decision,
                        relevant_memories=recall_output.relevant_memories,
                        position_snapshot=ledger_output.position_snapshot,
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )

            report_output = self._execute_step(
                module_name=ModuleName.REPORT_GENERATOR,
                action=lambda: self.modules.report_generator.generate(
                    ReportGeneratorInput(
                        evaluation=evaluator_output.evaluation,
                        position_snapshot=ledger_output.position_snapshot,
                        pnl_snapshot=ledger_output.pnl_snapshot,
                        evidence_packet=evidence_output.packet,
                        window_decision=window_output.decision,
                        trade_ledger=ledger_output.ledger,
                        relevant_memories=recall_output.relevant_memories,
                        user_focus_points=[intent.question for intent in cognition_output.cognition_state.user_intent_signals],
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.report_version = report_output.report.generated_prompt_version
            model_call_traces.extend(
                self._parse_model_call_traces(report_output.extensions.get("model_call_traces"))
            )

            if request.options.dry_run:
                self._record_step(
                    module_name=ModuleName.MEMORY_SERVICE,
                    status=RunStatus.SKIPPED,
                    details="dry_run enabled; memory write skipped",
                    step_results=step_results,
                    module_spans=module_spans,
                )
            else:
                memory_records = self._build_memory_records(
                    request=request,
                    intake_output=intake_output,
                    cognition_output=cognition_output,
                    evaluator_output=evaluator_output,
                    report_output=report_output,
                )
                write_output = self._execute_step(
                    module_name=ModuleName.MEMORY_SERVICE,
                    action=lambda: self.modules.memory_service.write(
                        MemoryWriteInput(
                            user_id=request.user_id,
                            batch=MemoryWriteBatch(records=memory_records),
                        )
                    ),
                    step_results=step_results,
                    module_spans=module_spans,
                )
                memory_write_results.append(
                    MemoryWriteResult(
                        collection="mixed",
                        memory_ids=write_output.written_memory_ids,
                        dedup_applied=write_output.dedup_count > 0,
                        merge_applied=write_output.merged_count > 0,
                    )
                )

            promptops_output = self._execute_step(
                module_name=ModuleName.PROMPTOPS,
                action=lambda: self.modules.promptops_engine.propose(
                    PromptOpsInput(
                        evaluation=evaluator_output.evaluation,
                        report=report_output.report,
                        run_metrics={
                            "step_count": len(step_results),
                            "evidence_completeness": evidence_output.packet.completeness_score,
                            "tool_failure_rate": tool_failure_rate,
                        },
                        active_prompt_versions={
                            "report_generation": report_output.report.generated_prompt_version.split(".")[1]
                            if "." in report_output.report.generated_prompt_version
                            else report_output.report.generated_prompt_version,
                        },
                    )
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            model_call_traces.extend(
                self._parse_model_call_traces(promptops_output.extensions.get("model_call_traces"))
            )
            trace.model_calls = model_call_traces

            if request.options.dry_run:
                self._record_step(
                    module_name=ModuleName.MEMORY_SERVICE,
                    status=RunStatus.SKIPPED,
                    details="dry_run enabled; improvement memory write skipped",
                    step_results=step_results,
                    module_spans=module_spans,
                )
            else:
                improvement_records = self._build_improvement_memory_records(
                    request=request,
                    promptops_output=promptops_output,
                )
                write_output = self._execute_step(
                    module_name=ModuleName.MEMORY_SERVICE,
                    action=lambda: self.modules.memory_service.write(
                        MemoryWriteInput(
                            user_id=request.user_id,
                            batch=MemoryWriteBatch(records=improvement_records),
                        )
                    ),
                    step_results=step_results,
                    module_spans=module_spans,
                )
                memory_write_results.append(
                    MemoryWriteResult(
                        collection="agent_improvement_notes",
                        memory_ids=write_output.written_memory_ids,
                        dedup_applied=write_output.dedup_count > 0,
                        merge_applied=write_output.merged_count > 0,
                    )
                )

            trace.module_spans = module_spans
            trace.ended_at = utc_now()
            trace.debug_context = {
                "memory_recalled": len(recall_output.relevant_memories.records),
                "plan_needs": len(plan_output.plan.needs),
                "evidence_missing": evidence_output.packet.missing_requirements,
                "selected_windows": [choice.window_type.value for choice in window_output.decision.selected_windows],
                "dry_run": request.options.dry_run,
                "promptops_status": promptops_output.bundle.proposal.status.value,
                "promptops_scope": promptops_output.bundle.proposal.scope.value,
                "report_quality_score": (
                    promptops_output.report_quality.overall_score if promptops_output.report_quality else None
                ),
                "model_calls": len(trace.model_calls),
                "module_models": get_settings().module_model_map(),
            }
            if not request.options.debug_mode:
                trace.debug_context.pop("selected_windows")

            return TaskResult(
                run_id=request.run_id,
                status=RunStatus.SUCCESS,
                step_results=step_results,
                report=report_output.report,
                evaluation=evaluator_output.evaluation,
                position_snapshot=ledger_output.position_snapshot,
                pnl_snapshot=ledger_output.pnl_snapshot,
                memory_write_results=memory_write_results,
                improvement_proposals=[promptops_output.bundle.proposal],
                trace=trace,
                errors=errors,
            )

        except NotImplementedError as exc:
            errors.append(
                ErrorRecord(
                    module_name=ModuleName.ORCHESTRATOR,
                    error_code="MODULE_NOT_IMPLEMENTED",
                    message=str(exc),
                    recoverable=True,
                )
            )
            self._record_step(
                module_name=ModuleName.ORCHESTRATOR,
                status=RunStatus.PARTIAL,
                details="Pipeline reached an unimplemented module.",
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.module_spans = module_spans
            trace.ended_at = utc_now()
            return TaskResult(
                run_id=request.run_id,
                status=RunStatus.PARTIAL,
                step_results=step_results,
                report=report_output.report if report_output else None,
                evaluation=evaluator_output.evaluation if evaluator_output else None,
                position_snapshot=ledger_output.position_snapshot if ledger_output else None,
                pnl_snapshot=ledger_output.pnl_snapshot if ledger_output else None,
                memory_write_results=memory_write_results,
                improvement_proposals=[promptops_output.bundle.proposal] if promptops_output else [],
                trace=trace,
                errors=errors,
            )

        except Exception as exc:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    module_name=ModuleName.ORCHESTRATOR,
                    error_code="PIPELINE_FAILED",
                    message=str(exc),
                    recoverable=False,
                )
            )
            self._record_step(
                module_name=ModuleName.ORCHESTRATOR,
                status=RunStatus.FAILED,
                details="Pipeline failed unexpectedly.",
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.module_spans = module_spans
            trace.ended_at = utc_now()
            return TaskResult(
                run_id=request.run_id,
                status=RunStatus.FAILED,
                step_results=step_results,
                report=report_output.report if report_output else None,
                evaluation=evaluator_output.evaluation if evaluator_output else None,
                position_snapshot=ledger_output.position_snapshot if ledger_output else None,
                pnl_snapshot=ledger_output.pnl_snapshot if ledger_output else None,
                memory_write_results=memory_write_results,
                improvement_proposals=[promptops_output.bundle.proposal] if promptops_output else [],
                trace=trace,
                errors=errors,
            )

    def _execute_step(
        self,
        module_name: ModuleName,
        action: Callable[[], T],
        step_results: list[StepResult],
        module_spans: list[ModuleRunSpan],
    ) -> T:
        started_at = utc_now()
        t0 = perf_counter()
        status = RunStatus.SUCCESS
        details: str | None = None

        try:
            return action()
        except NotImplementedError as exc:
            status = RunStatus.PARTIAL
            details = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001
            status = RunStatus.FAILED
            details = str(exc)
            raise
        finally:
            ended_at = utc_now()
            duration_ms = int((perf_counter() - t0) * 1000)
            module_spans.append(
                ModuleRunSpan(
                    module_name=module_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                    status=status,
                    notes=[details] if details else [],
                )
            )
            step_results.append(StepResult(module_name=module_name, status=status, details=details))

    def _record_step(
        self,
        module_name: ModuleName,
        status: RunStatus,
        details: str,
        step_results: list[StepResult],
        module_spans: list[ModuleRunSpan],
    ) -> None:
        timestamp = utc_now()
        module_spans.append(
            ModuleRunSpan(
                module_name=module_name,
                started_at=timestamp,
                ended_at=timestamp,
                duration_ms=0,
                status=status,
                notes=[details],
            )
        )
        step_results.append(StepResult(module_name=module_name, status=status, details=details))

    def _extract_active_theses(self, memories: list[MemoryRecord]) -> list[MemoryRecord]:
        return [memory for memory in memories if memory.memory_type == MemoryType.ACTIVE_THESIS]

    def _build_memory_records(
        self,
        request: ReviewRunRequest,
        intake_output,
        cognition_output,
        evaluator_output,
        report_output,
    ) -> list[MemoryRecord]:
        log = intake_output.normalized
        cognition = cognition_output.cognition_state
        evaluation = evaluator_output.evaluation
        report = report_output.report

        records: list[MemoryRecord] = [
            MemoryRecord(
                memory_id=f"mem_raw_{log.log_id}",
                user_id=request.user_id,
                memory_type=MemoryType.RAW_LOG,
                source_date=log.log_date,
                tickers=log.traded_tickers + log.mentioned_tickers,
                regime=log.market_context.regime,
                emotion_tags=[log.user_state.emotion] if log.user_state.emotion else [],
                quality_score=0.45,
                document_text=log.raw_text,
                structured_payload={
                    "field_errors": [error.model_dump() for error in log.field_errors],
                    "trade_event_count": len(log.trade_events),
                },
                status=MemoryStatus.ACTIVE,
                importance=0.55,
                confidence=0.8,
                keywords=["daily_log"],
            ),
            MemoryRecord(
                memory_id=f"mem_case_{evaluation.evaluation_id}",
                user_id=request.user_id,
                memory_type=MemoryType.COGNITIVE_CASE,
                source_date=evaluation.as_of_date,
                tickers=log.traded_tickers + log.mentioned_tickers,
                regime=log.market_context.regime,
                emotion_tags=[signal.emotion for signal in cognition.emotion_signals],
                quality_score=0.75,
                document_text=evaluation.summary,
                structured_payload={
                    "strengths": evaluation.strengths,
                    "mistakes": evaluation.mistakes,
                    "bias_findings": [bias.model_dump() for bias in evaluation.bias_findings],
                    "report_title": report.title,
                },
                status=MemoryStatus.ACTIVE,
                importance=0.8,
                confidence=0.7,
                keywords=["cognition_case", "daily_review"],
            ),
            MemoryRecord(
                memory_id=f"mem_profile_{request.user_id}_{log.log_date.isoformat()}",
                user_id=request.user_id,
                memory_type=MemoryType.USER_PROFILE,
                source_date=log.log_date,
                tickers=[],
                regime=log.market_context.regime,
                emotion_tags=[log.user_state.emotion] if log.user_state.emotion else [],
                quality_score=0.65,
                document_text="User profile incremental update from daily cognition run",
                structured_payload={
                    "emotion": log.user_state.emotion,
                    "stress": log.user_state.stress,
                    "focus": log.user_state.focus,
                    "explicit_rules": cognition.explicit_rules,
                },
                status=MemoryStatus.ACTIVE,
                importance=0.65,
                confidence=0.65,
                keywords=["user_profile_update"],
            ),
        ]

        for idx, hypothesis in enumerate(cognition.hypotheses[:5]):
            if hypothesis.confidence < 0.45:
                continue
            records.append(
                MemoryRecord(
                    memory_id=f"mem_thesis_{cognition.cognition_id}_{idx}",
                    user_id=request.user_id,
                    memory_type=MemoryType.ACTIVE_THESIS,
                    source_date=log.log_date,
                    tickers=hypothesis.related_tickers,
                    regime=log.market_context.regime,
                    emotion_tags=[log.user_state.emotion] if log.user_state.emotion else [],
                    quality_score=0.7,
                    document_text=hypothesis.statement,
                    structured_payload={
                        "hypothesis": hypothesis.model_dump(),
                        "timeframe_hint": hypothesis.timeframe_hint,
                    },
                    status=MemoryStatus.ACTIVE,
                    importance=0.75,
                    confidence=hypothesis.confidence,
                    keywords=["active_thesis"],
                )
            )

        # Keep only high-value entries: raw log, case, profile and confident active theses.
        return records

    def _build_improvement_memory_records(
        self,
        request: ReviewRunRequest,
        promptops_output,
    ) -> list[MemoryRecord]:
        proposal = promptops_output.bundle.proposal
        report_quality = promptops_output.report_quality
        replay_result = promptops_output.replay_result

        quality_score = 0.75
        if report_quality is not None:
            quality_score = max(0.5, min(0.95, report_quality.overall_score))
        confidence = 0.65 if proposal.status.value == "offline_evaluating" else 0.8

        return [
            MemoryRecord(
                memory_id=f"mem_improv_{proposal.proposal_id}",
                user_id=request.user_id,
                memory_type=MemoryType.IMPROVEMENT_NOTE,
                source_date=request.run_date,
                tickers=[],
                regime=None,
                emotion_tags=[],
                quality_score=quality_score,
                document_text=f"{proposal.problem_statement} | {proposal.candidate_change}",
                structured_payload={
                    "proposal": proposal.model_dump(mode="json"),
                    "report_quality": report_quality.model_dump(mode="json") if report_quality else None,
                    "replay_result": replay_result.model_dump(mode="json") if replay_result else None,
                    "prompt_candidate_id": (
                        promptops_output.bundle.prompt_candidate.version_id
                        if promptops_output.bundle.prompt_candidate
                        else None
                    ),
                    "context_policy_candidate_id": (
                        promptops_output.bundle.context_policy_candidate.policy_id
                        if promptops_output.bundle.context_policy_candidate
                        else None
                    ),
                    "rubric_candidate_id": (
                        promptops_output.bundle.rubric_candidate.rubric_id
                        if promptops_output.bundle.rubric_candidate
                        else None
                    ),
                },
                status=MemoryStatus.ACTIVE,
                importance=0.85,
                confidence=confidence,
                keywords=["promptops", proposal.scope.value, proposal.status.value],
            )
        ]

    def _parse_tool_call_traces(self, payload: object) -> list[ToolCallTrace]:
        if not isinstance(payload, list):
            return []
        traces: list[ToolCallTrace] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                traces.append(ToolCallTrace(**item))
            except Exception:  # noqa: BLE001
                continue
        return traces

    def _parse_model_call_traces(self, payload: object) -> list[ModelCallTrace]:
        if not isinstance(payload, list):
            return []
        traces: list[ModelCallTrace] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                traces.append(ModelCallTrace(**item))
            except Exception:  # noqa: BLE001
                continue
        return traces

    def _tool_failure_rate(self, traces: list[ToolCallTrace]) -> float:
        if not traces:
            return 0.0
        failed = sum(1 for trace in traces if not trace.success)
        return failed / len(traces)
