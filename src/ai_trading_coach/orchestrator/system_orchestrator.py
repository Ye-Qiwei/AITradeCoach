"""Planner -> Executor(parallel) -> Reporter -> Judge orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, TypeVar

from ai_trading_coach.config import Settings, get_settings
from ai_trading_coach.domain.agent_models import Plan
from ai_trading_coach.domain.enums import EvaluationCategory, ModuleName, RunStatus
from ai_trading_coach.domain.models import (
    DailyReviewReport,
    ErrorRecord,
    EvaluationResult,
    HypothesisAssessment,
    MemoryWriteResult,
    ModelCallTrace,
    ModuleRunSpan,
    PnLSnapshot,
    PositionSnapshot,
    ReportSection,
    ReviewRunRequest,
    RunTrace,
    StepResult,
    TaskResult,
    ToolCallTrace,
)
from ai_trading_coach.errors import ReportValidationError
from ai_trading_coach.modules.agent import (
    CombinedParserAgent,
    ContextBuilderV2,
    ExecutorEngine,
    PlannerAgent,
    ReportJudge,
    ReporterAgent,
)

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class OrchestratorModules:
    parser_agent: CombinedParserAgent
    planner_agent: PlannerAgent
    executor_engine: ExecutorEngine
    reporter_agent: ReporterAgent
    report_judge: ReportJudge
    context_builder: ContextBuilderV2


class PipelineOrchestrator:
    """Execute the LLM-first controllable agent workflow."""

    def __init__(self, modules: OrchestratorModules, settings: Settings | None = None) -> None:
        self.modules = modules
        self.settings = settings or get_settings()

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
        model_calls: list[ModelCallTrace] = []
        tool_calls: list[ToolCallTrace] = []

        final_report: DailyReviewReport | None = None
        final_evaluation: EvaluationResult | None = None

        try:
            parse_result, parse_trace = self._execute_step(
                module_name=ModuleName.LOG_INTAKE,
                action=lambda: self.modules.parser_agent.parse(
                    run_id=request.run_id,
                    user_id=request.user_id,
                    run_date=request.run_date,
                    raw_log_text=request.raw_log_text,
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            if parse_trace is not None:
                model_calls.append(parse_trace)

            planner_context = self.modules.context_builder.for_planner(parse_result=parse_result)
            plan, planner_trace = self._execute_step(
                module_name=ModuleName.EVIDENCE_PLANNER,
                action=lambda: self.modules.planner_agent.plan(
                    parse_result=parse_result,
                    planner_context=planner_context,
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            if planner_trace is not None:
                model_calls.append(planner_trace)

            executor_result = self._execute_step(
                module_name=ModuleName.MCP_GATEWAY,
                action=lambda: self.modules.executor_engine.execute(
                    plan=plan,
                    user_id=request.user_id,
                ),
                step_results=step_results,
                module_spans=module_spans,
            )
            evidence_packet = executor_result.evidence_packet
            tool_calls.extend(executor_result.tool_traces)
            trace.evidence_sources = evidence_packet.source_registry
            trace.debug_context["subtask_traces"] = [
                item.model_dump(mode="json") for item in executor_result.subtask_traces
            ]

            rewrite_instruction: str | None = None
            verdict_passed = False
            draft_markdown = ""
            max_rounds = max(0, self.settings.agent_max_rewrite_rounds)
            rounds = 0
            while rounds <= max_rounds:
                rounds += 1
                report_context = self.modules.context_builder.for_reporter(
                    evidence_packet=evidence_packet,
                    plan=plan,
                    intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
                )
                draft, report_trace = self._execute_step(
                    module_name=ModuleName.REPORT_GENERATOR,
                    action=lambda: self.modules.reporter_agent.generate(
                        evidence_packet=evidence_packet,
                        report_context=report_context,
                        intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
                        plan=plan,
                        rewrite_instruction=rewrite_instruction,
                    ),
                    step_results=step_results,
                    module_spans=module_spans,
                )
                if report_trace is not None:
                    model_calls.append(report_trace)
                draft_markdown = draft.markdown

                judge_context = self.modules.context_builder.for_judge(
                    report_markdown=draft.markdown,
                    evidence_packet=evidence_packet,
                    intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
                    plan=plan,
                    rewrite_instruction=rewrite_instruction,
                )
                verdict, judge_trace = self._execute_step(
                    module_name=ModuleName.EVALUATOR,
                    action=lambda: self.modules.report_judge.evaluate(
                        report_markdown=draft.markdown,
                        judge_context=judge_context,
                        intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
                        evidence_packet=evidence_packet,
                        plan=plan,
                    ),
                    step_results=step_results,
                    module_spans=module_spans,
                )
                if judge_trace is not None:
                    model_calls.append(judge_trace)

                if verdict.passed:
                    verdict_passed = True
                    break
                rewrite_instruction = verdict.rewrite_instruction or "Improve citations and align with user intent."

            trace.rewrite_rounds = rounds
            if not verdict_passed:
                raise ReportValidationError(
                    f"Judge failed after {rounds} round(s). Max rounds={max_rounds}."
                )

            final_report = self._build_report(
                request=request,
                markdown=draft_markdown,
                plan=plan,
            )
            final_evaluation = self._build_evaluation(
                request=request,
                plan=plan,
                summary="Judge passed with citation-complete report.",
            )

            trace.model_calls = model_calls
            trace.tool_calls = tool_calls
            trace.module_spans = module_spans
            trace.report_version = final_report.generated_prompt_version
            trace.debug_context["plan_subtasks"] = [item.model_dump(mode="json") for item in plan.subtasks]
            trace.debug_context["module_models"] = self.settings.module_model_map()
            trace.ended_at = utc_now()

            return TaskResult(
                run_id=request.run_id,
                status=RunStatus.SUCCESS,
                step_results=step_results,
                report=final_report,
                evaluation=final_evaluation,
                position_snapshot=PositionSnapshot(
                    snapshot_id=f"ps_{request.run_id}",
                    user_id=request.user_id,
                    as_of_date=request.run_date,
                ),
                pnl_snapshot=PnLSnapshot(
                    snapshot_id=f"pnl_{request.run_id}",
                    user_id=request.user_id,
                    as_of_date=request.run_date,
                ),
                memory_write_results=[MemoryWriteResult(collection="disabled", memory_ids=[])],
                improvement_proposals=[],
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
                details=str(exc),
                step_results=step_results,
                module_spans=module_spans,
            )
            trace.model_calls = model_calls
            trace.tool_calls = tool_calls
            trace.module_spans = module_spans
            trace.ended_at = utc_now()
            return TaskResult(
                run_id=request.run_id,
                status=RunStatus.FAILED,
                step_results=step_results,
                report=final_report,
                evaluation=final_evaluation,
                trace=trace,
                errors=errors,
            )

    def _execute_step(
        self,
        *,
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
        except Exception as exc:  # noqa: BLE001
            status = RunStatus.FAILED
            details = str(exc)
            raise
        finally:
            ended_at = utc_now()
            module_spans.append(
                ModuleRunSpan(
                    module_name=module_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=int((perf_counter() - t0) * 1000),
                    status=status,
                    notes=[details] if details else [],
                )
            )
            step_results.append(StepResult(module_name=module_name, status=status, details=details))

    def _record_step(
        self,
        *,
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

    def _build_report(self, *, request: ReviewRunRequest, markdown: str, plan: Plan) -> DailyReviewReport:
        sections = self._parse_markdown_sections(markdown)
        first_lines = [line.strip("- ").strip() for line in markdown.splitlines() if line.strip().startswith("-")]
        return DailyReviewReport(
            report_id=f"report_{request.run_id}",
            user_id=request.user_id,
            report_date=request.run_date,
            title=f"Daily Trading Review - {request.run_date.isoformat()}",
            sections=sections,
            key_takeaways=first_lines[:3],
            next_watchlist=plan.follow_up_triggers[:8],
            strategy_adjustments=plan.risk_uncertainties[:4],
            risk_alerts=plan.risk_uncertainties[:4],
            generated_prompt_version=self.settings.prompt_version,
            markdown_body=markdown if markdown.endswith("\n") else f"{markdown}\n",
        )

    def _build_evaluation(self, *, request: ReviewRunRequest, plan: Plan, summary: str) -> EvaluationResult:
        assessments = [
            HypothesisAssessment(
                hypothesis_id=subtask.subtask_id,
                category=EvaluationCategory.PARTIAL,
                commentary=subtask.objective,
            )
            for subtask in plan.subtasks[:4]
        ]
        return EvaluationResult(
            evaluation_id=f"eval_{request.run_id}",
            user_id=request.user_id,
            as_of_date=request.run_date,
            summary=summary,
            hypothesis_assessments=assessments,
            follow_up_signals=plan.follow_up_triggers,
            warning_flags=plan.risk_uncertainties,
        )

    def _parse_markdown_sections(self, markdown: str) -> list[ReportSection]:
        lines = markdown.splitlines()
        headings: list[tuple[str, int]] = []
        for idx, line in enumerate(lines):
            if line.startswith("## "):
                headings.append((line[3:].strip(), idx))

        if not headings:
            return [ReportSection(title="Summary", content=markdown.strip())]

        sections: list[ReportSection] = []
        for pos, (title, idx) in enumerate(headings):
            end = headings[pos + 1][1] if pos + 1 < len(headings) else len(lines)
            content = "\n".join(lines[idx + 1 : end]).strip()
            sections.append(ReportSection(title=title, content=content))
        return sections

