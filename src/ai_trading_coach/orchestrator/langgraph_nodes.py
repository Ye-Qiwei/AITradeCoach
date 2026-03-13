"""LangGraph nodes for daily parse->research->report->judge pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import EvaluationCategory, ModelCallPurpose, ModuleName, RunStatus
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, LongTermJudgementRecord, ResearchOutput, compute_due_date
from ai_trading_coach.domain.models import DailyReviewReport, ErrorRecord, EvaluationResult, HypothesisAssessment, MemoryWriteResult, ModuleRunSpan, PnLSnapshot, PositionSnapshot, ReportSection, RunTrace, StepResult, TaskResult
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.agent.evidence_packet_builder import build_evidence_packet
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools
from ai_trading_coach.modules.agent.text_output_parsing import parse_research_output_text
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _extract_agent_messages(raw_messages: list[Any]) -> dict[str, list[str]]:
    groups = {"input_messages": [], "intermediate_messages": [], "tool_error_messages": [], "final_messages": []}
    for message in raw_messages:
        text = _extract_message_text(message).strip()
        if not text:
            continue
        lowered = text.lower()
        if "tool_error:" in lowered:
            groups["tool_error_messages"].append(text)
        elif lowered.startswith("{") and '"task"' in lowered:
            groups["input_messages"].append(text)
        elif lowered.startswith("{") or lowered.startswith("["):
            groups["final_messages"].append(text)
        else:
            groups["intermediate_messages"].append(text)
    return groups


def _latest_prompt_version(model_calls: list[dict[str, Any]], purpose: ModelCallPurpose) -> str:
    for item in reversed(model_calls):
        if item.get("purpose") == purpose.value and item.get("prompt_version"):
            return str(item["prompt_version"])
    return "unknown"


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
    prompt_manager: PromptManager

    def parse_log(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result, parse_trace = self.parser_agent.parse(run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, raw_log_text=req.raw_log_text)
        model_calls = list(state.get("model_calls", []))
        if parse_trace is not None:
            model_calls.append(parse_trace.model_dump(mode="json"))
        return {"parse_result": parse_result, "model_calls": model_calls}

    def plan_research_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        return {"research_retry_count": int(state.get("research_retry_count", 0)), "is_sufficient": False, "verify_suggestions": list(state.get("verify_suggestions", []))}

    def execute_collection_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result = state["parse_result"]
        runtime = MCPToolRuntime()
        tools = build_runtime_research_tools(settings=self.settings, mcp_manager=self.mcp_manager, runtime=runtime)
        prompt = self.prompt_manager.load_active("research_agent")
        agent = create_agent(model=self.chat_model, tools=tools, system_prompt=prompt.system_prompt)
        payload = {
            "task": "For each judgement in order, collect evidence and return the same-ordered judgement list with an evidence field.",
            "verify_suggestions": state.get("verify_suggestions", []),
            "judgements": [j.model_dump(mode="json") for j in parse_result.all_judgements()],
        }
        result = agent.invoke({"messages": [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]})
        message_groups = _extract_agent_messages(result.get("messages", []))
        final_messages = message_groups["final_messages"] or message_groups["intermediate_messages"]
        if not final_messages:
            raise ValueError("Research agent produced no output messages.")
        research_output = parse_research_output_text(final_messages[-1], judgements=parse_result.all_judgements())
        research_output.validate_against(parse_result.all_judgements())

        accumulated_items = list(state.get("accumulated_evidence_items", [])) + list(runtime.evidence_items)
        evidence_packet = build_evidence_packet(packet_id=f"packet_{req.run_id}", user_id=req.user_id, evidence_items=accumulated_items)
        source_count = len(evidence_packet.source_registry)
        if research_output.judgements and source_count == 0:
            message_groups["final_messages"].append("research summary returned without evidence sources")

        return {
            "agent_messages": list(state.get("agent_messages", [])) + [*message_groups["input_messages"], *message_groups["intermediate_messages"], *message_groups["tool_error_messages"], *message_groups["final_messages"]],
            "agent_message_groups": message_groups,
            "evidence_packet": evidence_packet,
            "tool_calls": list(state.get("tool_calls", [])) + [t.model_dump(mode="json") for t in runtime.tool_traces],
            "react_steps": list(state.get("react_steps", [])) + [s.model_dump(mode="json") for s in runtime.react_steps],
            "research_output": research_output,
            "accumulated_evidence_items": accumulated_items,
            "accumulated_tool_failures": int(state.get("accumulated_tool_failures", 0)) + sum(1 for t in runtime.tool_traces if not t.success),
            "research_anomalies": ["research_summary_without_evidence"] if (research_output.judgements and source_count == 0) else [],
        }

    def verify_information_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        retry_count = int(state.get("research_retry_count", 0)) + 1
        research_output: ResearchOutput = state["research_output"]
        source_count = len(state["evidence_packet"].source_registry)
        covered = sum(1 for item in research_output.judgements if item.evidence.collected_evidence_items)
        total = len(state["parse_result"].all_judgements())
        tool_failures = int(state.get("accumulated_tool_failures", 0))
        sufficient = covered >= total and source_count >= self.settings.react_require_min_sources
        should_continue = (not sufficient) and retry_count < self.settings.react_max_iterations and tool_failures < self.settings.react_max_tool_failures
        return {
            "is_sufficient": sufficient,
            "verify_suggestions": [] if sufficient else ["Increase evidence coverage and source diversity."],
            "research_retry_count": retry_count,
            "continue_collection": should_continue,
            "insufficiency_reason": "" if sufficient else f"coverage={covered}/{total}; sources={source_count}",
            "research_stop_reason": "sufficient" if sufficient else ("max_iterations_reached" if retry_count >= self.settings.react_max_iterations else "insufficient_coverage"),
        }

    def route_after_verify(self, state: OrchestratorGraphState) -> str:
        return "continue_collection" if state.get("continue_collection") else "research_done"

    def build_report_context(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        return {"report_context": self.context_builder.for_reporter(parse_result=state["parse_result"], research_output=state["research_output"], evidence_packet=state["evidence_packet"])}

    def generate_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        out, trace = self.reporter_agent.generate(evidence_packet=state["evidence_packet"], report_context=state["report_context"], rewrite_instruction=state.get("rewrite_instruction"))
        model_calls = list(state.get("model_calls", []))
        if trace is not None:
            model_calls.append(trace.model_dump(mode="json"))
        return {"report_draft": out.markdown, "judgement_feedback": out.judgement_feedback, "model_calls": model_calls}

    def judge_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        judge_ctx = self.context_builder.for_judge(report_markdown=state["report_draft"], judgement_feedback=state.get("judgement_feedback", []), parse_result=state["parse_result"], research_output=state["research_output"], report_context=state.get("report_context", {}))
        verdict, trace = self.report_judge.evaluate(report_markdown=state["report_draft"], judge_context=judge_ctx, evidence_packet=state["evidence_packet"])
        model_calls = list(state.get("model_calls", []))
        if trace is not None:
            model_calls.append(trace.model_dump(mode="json"))
        rewrites_used = int(state.get("rewrite_count", 0)) + (0 if verdict.passed else 1)
        return {"judge_verdict": verdict, "rewrite_instruction": verdict.rewrite_instruction, "rewrite_count": rewrites_used, "model_calls": model_calls}

    def route_after_judge(self, state: OrchestratorGraphState) -> str:
        verdict: JudgeVerdict = state["judge_verdict"]
        if verdict.passed:
            return "pass"
        if int(state.get("rewrite_count", 0)) <= self.settings.agent_max_rewrite_rounds:
            return "rewrite"
        return "fail"

    def finalize_result(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        feedback = [DailyJudgementFeedback.model_validate(i) for i in state.get("judgement_feedback", [])]
        judgements = state["parse_result"].all_judgements()
        records = []
        for idx, judgement in enumerate(judgements):
            fb = feedback[idx] if idx < len(feedback) else DailyJudgementFeedback(initial_feedback="insufficient_evidence", evaluation_window=judgement.evaluation_window)
            records.append(LongTermJudgementRecord(judgement_id=f"{req.run_id}_j{idx+1}", user_id=req.user_id, run_id=req.run_id, run_date=req.run_date, due_date=compute_due_date(req.run_date, fb.evaluation_window), judgement=judgement, initial_feedback=fb))
        memory_results: list[MemoryWriteResult] = []
        if not req.options.dry_run:
            self.long_term_store.upsert_records(records)
            memory_results = [MemoryWriteResult(collection="long_term_memory", memory_ids=[r.judgement_id for r in records])]
        markdown = state["report_draft"]
        report = DailyReviewReport(report_id=f"report_{req.run_id}", user_id=req.user_id, report_date=req.run_date, title=f"Daily Trading Review - {req.run_date}", sections=[ReportSection(title="Daily Feedback", content=markdown)], generated_prompt_version=_latest_prompt_version(state.get("model_calls", []), ModelCallPurpose.REPORT_GENERATION), markdown_body=markdown if markdown.endswith("\n") else markdown + "\n")
        trace = RunTrace(run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, trigger_type=req.trigger_type, started_at=state.get("run_started_at", utc_now()), ended_at=utc_now(), module_spans=[ModuleRunSpan(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)], model_calls=state.get("model_calls", []), tool_calls=state.get("tool_calls", []), evidence_sources=state["evidence_packet"].source_registry, rewrite_rounds=int(state.get("rewrite_count", 0)), debug_context={"report_context": state.get("report_context", {}), "research_output": state["research_output"].model_dump(mode="json")}, react_steps=state.get("react_steps", []))
        final = TaskResult(run_id=req.run_id, status=RunStatus.SUCCESS, step_results=[StepResult(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)], report=report, evaluation=EvaluationResult(evaluation_id=f"eval_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date, summary="Daily judgement feedback created.", hypothesis_assessments=[HypothesisAssessment(hypothesis_id="daily_feedback", category=EvaluationCategory.PARTIAL, commentary="Ready for long-term evaluation")]), position_snapshot=PositionSnapshot(snapshot_id=f"ps_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date), pnl_snapshot=PnLSnapshot(snapshot_id=f"pnl_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date), memory_write_results=memory_results, trace=trace)
        return {"final_result": final}

    def finalize_failure(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        final = TaskResult(run_id=req.run_id, status=RunStatus.FAILED, step_results=[StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.FAILED)], errors=[ErrorRecord(module_name=ModuleName.EVALUATOR, error_code="REPORT_VALIDATION_FAILED", message="Judge failed and rewrite budget exhausted.", recoverable=False)])
        return {"final_result": final}
