"""LangGraph node implementations for the ReAct-only orchestration path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import JudgeVerdict, ReporterDraft
from ai_trading_coach.domain.enums import EvaluationCategory, ModelCallPurpose, ModuleName, RunStatus
from ai_trading_coach.domain.models import (
    DailyReviewReport,
    ErrorRecord,
    EvaluationResult,
    HypothesisAssessment,
    MemoryWriteResult,
    ModuleRunSpan,
    PnLSnapshot,
    PositionSnapshot,
    ReportSection,
    RunTrace,
    StepResult,
    TaskResult,
)
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, build_langchain_mcp_tools
from ai_trading_coach.modules.agent.react_tools import build_evidence_packet
from ai_trading_coach.domain.react_models import ResearchSummary
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LangGraphNodeRuntime:
    parser_agent: CombinedParserAgent
    reporter_agent: ReporterAgent
    report_judge: ReportJudge
    context_builder: ContextBuilderV2
    mcp_manager: MCPClientManager
    chat_model: Any
    settings: Settings

    def parse_log(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        request = state["request"]
        parse_result, parse_trace = self.parser_agent.parse(
            run_id=request.run_id,
            user_id=request.user_id,
            run_date=request.run_date,
            raw_log_text=request.raw_log_text,
        )
        model_calls = list(state.get("model_calls", []))
        if parse_trace is not None:
            model_calls.append(parse_trace.model_dump(mode="json"))
        return {
            "parse_result": parse_result,
            "model_calls": model_calls,
            "messages": list(state.get("messages", [])),
        }

    def react_research(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        request = state["request"]
        parse_result = state["parse_result"]
        runtime = MCPToolRuntime()
        tools = build_langchain_mcp_tools(mcp_manager=self.mcp_manager, runtime=runtime)
        react_agent = create_react_agent(self.chat_model, tools)
        prompt = (
            "You are a trading research ReAct agent. Use tools to collect verifiable evidence, then stop. "
            "Only use: get_price_history, search_news, list_filings, get_macro_series. "
            "Never use bootstrap_investigation_plan."
        )
        result = react_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"{prompt}\n"
                            f"intent={[item.question for item in parse_result.cognition_state.user_intent_signals]}\n"
                            f"tickers={parse_result.normalized_log.mentioned_tickers or parse_result.normalized_log.traded_tickers}\n"
                            f"min_sources={self.settings.react_require_min_sources}"
                        )
                    )
                ]
            }
        )

        summary = result["messages"][-1].content if result.get("messages") else ""
        evidence_packet = build_evidence_packet(
            packet_id=f"packet_{request.run_id}",
            user_id=request.user_id,
            evidence_items=runtime.evidence_items,
        )
        research_summary = ResearchSummary(
            research_id=f"research_{request.run_id}",
            investigation_summary=str(summary),
            key_findings=[],
            open_questions=[],
            collected_evidence=runtime.evidence_items,
            evidence_item_ids=[item.item_id for item in runtime.evidence_items],
            tool_steps=[],
        )
        tool_calls = list(state.get("tool_calls", []))
        tool_calls.extend([trace.model_dump(mode="json") for trace in runtime.tool_traces])
        return {
            "evidence_packet": evidence_packet,
            "research_summary": research_summary,
            "messages": result.get("messages", []),
            "tool_calls": tool_calls,
        }

    def build_report_context(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        parse_result = state["parse_result"]
        evidence_packet = state["evidence_packet"]
        intent = [item.question for item in parse_result.cognition_state.user_intent_signals]
        outline = {
            "investigation_outline": [
                *parse_result.cognition_state.core_judgements[:3],
                *parse_result.cognition_state.risk_concerns[:3],
            ]
        }
        report_context = self.context_builder.for_reporter(
            evidence_packet=evidence_packet,
            intent=intent,
            investigation_outline=outline,
        )
        return {"report_context": report_context}

    def generate_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        parse_result = state["parse_result"]
        evidence_packet = state["evidence_packet"]
        draft, report_trace = self.reporter_agent.generate(
            evidence_packet=evidence_packet,
            report_context=state["report_context"],
            intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
            rewrite_instruction=state.get("rewrite_instruction"),
        )
        model_calls = list(state.get("model_calls", []))
        if report_trace is not None:
            model_calls.append(report_trace.model_dump(mode="json"))
        return {"report_draft": draft, "model_calls": model_calls}

    def judge_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        parse_result = state["parse_result"]
        evidence_packet = state["evidence_packet"]
        report_draft = state["report_draft"]
        judge_context = self.context_builder.for_judge(
            report_markdown=report_draft.markdown,
            evidence_packet=evidence_packet,
            intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
            rewrite_instruction=state.get("rewrite_instruction"),
        )
        verdict, judge_trace = self.report_judge.evaluate(
            report_markdown=report_draft.markdown,
            judge_context=judge_context,
            intent=[item.question for item in parse_result.cognition_state.user_intent_signals],
            evidence_packet=evidence_packet,
        )
        model_calls = list(state.get("model_calls", []))
        if judge_trace is not None:
            model_calls.append(judge_trace.model_dump(mode="json"))
        return {
            "judge_verdict": verdict,
            "rewrite_instruction": verdict.rewrite_instruction,
            "rewrite_count": int(state.get("rewrite_count", 0)) + (0 if verdict.passed else 1),
            "model_calls": model_calls,
        }

    def finalize_result(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        request = state["request"]
        draft = state["report_draft"]
        evidence_packet = state["evidence_packet"]
        verdict = state["judge_verdict"]
        report = self._build_report(request=request, markdown=draft.markdown, evidence_packet=evidence_packet)
        trace = self._build_trace(state=state)
        final = TaskResult(
            run_id=request.run_id,
            status=RunStatus.SUCCESS,
            step_results=[
                StepResult(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS),
                StepResult(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS),
                StepResult(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS),
                StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS),
            ],
            report=report,
            evaluation=EvaluationResult(
                evaluation_id=f"eval_{request.run_id}",
                user_id=request.user_id,
                as_of_date=request.run_date,
                summary="Judge passed with citation-complete report.",
                hypothesis_assessments=[
                    HypothesisAssessment(
                        hypothesis_id="react_research",
                        category=EvaluationCategory.PARTIAL,
                        commentary=verdict.reasons[0] if verdict.reasons else "passed",
                    )
                ],
                follow_up_signals=[],
                warning_flags=[],
            ),
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
            errors=[],
        )
        return {"final_result": final}

    def finalize_failure(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        request = state["request"]
        trace = self._build_trace(state=state)
        final = TaskResult(
            run_id=request.run_id,
            status=RunStatus.FAILED,
            step_results=[StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.FAILED)],
            trace=trace,
            errors=[
                ErrorRecord(
                    module_name=ModuleName.EVALUATOR,
                    error_code="REPORT_VALIDATION_FAILED",
                    message="Judge failed and rewrite budget exhausted.",
                    recoverable=False,
                )
            ],
        )
        return {"final_result": final}

    def route_after_judge(self, state: OrchestratorGraphState) -> str:
        verdict = state["judge_verdict"]
        if verdict.passed:
            return "pass"
        if int(state.get("rewrite_count", 0)) < max(0, self.settings.agent_max_rewrite_rounds):
            return "rewrite"
        return "fail"

    def _build_report(self, *, request, markdown: str, evidence_packet) -> DailyReviewReport:
        sections = self._parse_markdown_sections(markdown)
        first_lines = [line.strip("- ").strip() for line in markdown.splitlines() if line.strip().startswith("-")]
        return DailyReviewReport(
            report_id=f"report_{request.run_id}",
            user_id=request.user_id,
            report_date=request.run_date,
            title=f"Daily Trading Review - {request.run_date.isoformat()}",
            sections=sections,
            key_takeaways=first_lines[:3],
            next_watchlist=[],
            strategy_adjustments=[],
            risk_alerts=[],
            generated_prompt_version=self.settings.prompt_version,
            markdown_body=markdown if markdown.endswith("\n") else f"{markdown}\n",
        )

    def _build_trace(self, *, state: OrchestratorGraphState) -> RunTrace:
        request = state["request"]
        return RunTrace(
            run_id=request.run_id,
            user_id=request.user_id,
            run_date=request.run_date,
            trigger_type=request.trigger_type,
            started_at=utc_now(),
            ended_at=utc_now(),
            module_spans=[
                ModuleRunSpan(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS),
                ModuleRunSpan(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS),
                ModuleRunSpan(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS),
                ModuleRunSpan(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS),
            ],
            model_calls=state.get("model_calls", []),
            tool_calls=state.get("tool_calls", []),
            evidence_sources=state["evidence_packet"].source_registry if state.get("evidence_packet") else [],
            rewrite_rounds=int(state.get("rewrite_count", 0)) + 1,
            debug_context={"research_summary": state.get("research_summary", {})},
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


__all__ = ["LangGraphNodeRuntime"]
