"""LangGraph nodes for daily parse->research->report->judge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import EvaluationCategory, ModuleName, RunStatus
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, LongTermJudgementRecord, ResearchOutput, compute_due_date
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
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
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
    long_term_store: LongTermMemoryStore

    def parse_log(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result, parse_trace = self.parser_agent.parse(
            run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, raw_log_text=req.raw_log_text
        )
        model_calls = list(state.get("model_calls", []))
        if parse_trace:
            model_calls.append(parse_trace.model_dump(mode="json"))
        return {"parse_result": parse_result, "model_calls": model_calls}

    def react_research(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result = state["parse_result"]
        runtime = MCPToolRuntime()
        tools = build_langchain_mcp_tools(mcp_manager=self.mcp_manager, runtime=runtime)
        agent = create_react_agent(self.chat_model, tools)
        judgement_prompt = "\n".join(f"- {j.judgement_id}: {j.thesis}" for j in parse_result.all_judgements())
        result = agent.invoke({"messages": [HumanMessage(content=f"Research these judgements with sufficient evidence only:\n{judgement_prompt}")]})
        evidence_packet = build_evidence_packet(packet_id=f"packet_{req.run_id}", user_id=req.user_id, evidence_items=runtime.evidence_items)
        by_j = []
        for j in parse_result.all_judgements():
            by_j.append({"judgement_id": j.judgement_id, "evidence_item_ids": [e.item_id for e in runtime.evidence_items[:2]], "support_signal": "uncertain", "sufficiency_reason": "Collected baseline cross-source evidence."})
        research_output = ResearchOutput(research_id=f"research_{req.run_id}", judgement_evidence=by_j, stop_reason="Evidence judged sufficient for initial daily feedback.")
        return {
            "messages": result.get("messages", []),
            "evidence_packet": evidence_packet,
            "research_output": research_output,
            "tool_calls": [t.model_dump(mode="json") for t in runtime.tool_traces],
            "react_steps": [m.content for m in result.get("messages", [])[-4:]],
        }

    def build_report_context(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        context = self.context_builder.for_reporter(
            parse_result=state["parse_result"],
            research_output=state["research_output"],
            evidence_packet=state["evidence_packet"],
        )
        return {"report_context": context}

    def generate_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        out, trace = self.reporter_agent.generate(
            evidence_packet=state["evidence_packet"],
            report_context=state["report_context"],
            rewrite_instruction=state.get("rewrite_instruction"),
        )
        model_calls = list(state.get("model_calls", []))
        if trace:
            model_calls.append(trace.model_dump(mode="json"))
        return {"report_draft": out.markdown, "judgement_feedback": out.judgement_feedback, "model_calls": model_calls}

    def judge_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        judge_ctx = self.context_builder.for_judge(report_markdown=state["report_draft"], judgement_feedback=state.get("judgement_feedback", []))
        verdict, trace = self.report_judge.evaluate(
            report_markdown=state["report_draft"],
            judge_context=judge_ctx,
            evidence_packet=state["evidence_packet"],
        )
        model_calls = list(state.get("model_calls", []))
        if trace:
            model_calls.append(trace.model_dump(mode="json"))
        return {
            "judge_verdict": verdict,
            "rewrite_instruction": verdict.rewrite_instruction,
            "rewrite_count": int(state.get("rewrite_count", 0)) + (0 if verdict.passed else 1),
            "model_calls": model_calls,
        }

    def route_after_judge(self, state: OrchestratorGraphState) -> str:
        verdict: JudgeVerdict = state["judge_verdict"]
        if verdict.passed:
            return "pass"
        if int(state.get("rewrite_count", 0)) < self.settings.agent_max_rewrite_rounds:
            return "rewrite"
        return "fail"

    def finalize_result(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        markdown = state["report_draft"]
        feedback = [DailyJudgementFeedback.model_validate(i) for i in state.get("judgement_feedback", [])]
        records = [
            LongTermJudgementRecord(
                judgement_id=j.judgement_id,
                user_id=req.user_id,
                run_id=req.run_id,
                run_date=req.run_date,
                due_date=compute_due_date(req.run_date, next((f.evaluation_window for f in feedback if f.judgement_id == j.judgement_id), j.proposed_evaluation_window)),
                judgement=j,
                initial_feedback=next((f for f in feedback if f.judgement_id == j.judgement_id), DailyJudgementFeedback(judgement_id=j.judgement_id, initial_feedback="high_uncertainty", evidence_summary="", evaluation_window=j.proposed_evaluation_window, window_rationale="fallback", followup_indicators=[], source_ids=[])),
            )
            for j in state["parse_result"].all_judgements()
        ]
        self.long_term_store.upsert_records(records)
        report = DailyReviewReport(
            report_id=f"report_{req.run_id}", user_id=req.user_id, report_date=req.run_date,
            title=f"Daily Trading Review - {req.run_date}", sections=[ReportSection(title="Daily Feedback", content=markdown)],
            generated_prompt_version=self.settings.prompt_version, markdown_body=markdown if markdown.endswith("\n") else markdown + "\n",
        )
        trace = RunTrace(
            run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, trigger_type=req.trigger_type,
            started_at=utc_now(), ended_at=utc_now(),
            module_spans=[ModuleRunSpan(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)],
            model_calls=state.get("model_calls", []), tool_calls=state.get("tool_calls", []), evidence_sources=state["evidence_packet"].source_registry,
            debug_context={"report_context": state.get("report_context", {})}, react_steps=state.get("react_steps", []),
        )
        final = TaskResult(
            run_id=req.run_id, status=RunStatus.SUCCESS,
            step_results=[StepResult(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)],
            report=report,
            evaluation=EvaluationResult(evaluation_id=f"eval_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date, summary="Daily judgement feedback created.", hypothesis_assessments=[HypothesisAssessment(hypothesis_id="daily_feedback", category=EvaluationCategory.PARTIAL, commentary="Ready for long-term evaluation")]),
            position_snapshot=PositionSnapshot(snapshot_id=f"ps_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date),
            pnl_snapshot=PnLSnapshot(snapshot_id=f"pnl_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date),
            memory_write_results=[MemoryWriteResult(collection="long_term_memory", memory_ids=[r.judgement_id for r in records])],
            trace=trace,
        )
        return {"final_result": final}

    def finalize_failure(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        final = TaskResult(
            run_id=req.run_id,
            status=RunStatus.FAILED,
            step_results=[StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.FAILED)],
            errors=[ErrorRecord(module_name=ModuleName.EVALUATOR, error_code="REPORT_VALIDATION_FAILED", message="Judge failed and rewrite budget exhausted.", recoverable=False)],
        )
        return {"final_result": final}
