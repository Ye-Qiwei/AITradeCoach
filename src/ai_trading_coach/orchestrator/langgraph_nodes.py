"""LangGraph nodes for daily parse->plan->research->report->judge pipeline."""

from __future__ import annotations

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
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.text_output_parsing import (
    build_research_output_from_items,
    build_researched_judgement,
    parse_single_research_output_text,
)
from ai_trading_coach.modules.agent.tools import ToolRuntime, build_runtime_tools, get_tool_availability
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _latest_prompt_version(model_calls: list[dict[str, Any]], purpose: ModelCallPurpose) -> str:
    for item in reversed(model_calls):
        if item.get("purpose") == purpose.value and item.get("prompt_version"):
            return str(item["prompt_version"])
    return "unknown"


def _render_single_judgement_research_task_markdown(*, judgement: Any, plan_markdown: str, judgement_index: int, total_judgements: int, related_assets: list[str]) -> str:
    related_assets_lines = [f"- {item}" for item in related_assets] or ["- none"]
    return "\n".join([
        "# Research Task",
        "",
        "This task contains exactly one judgement.",
        f"Judgement position: {judgement_index}/{total_judgements}",
        "",
        "## Input Judgement",
        f"- category: {judgement.category}",
        f"- target: {judgement.target}",
        f"- thesis: {judgement.thesis}",
        f"- evaluation_window: {judgement.evaluation_window}",
        "",
        "## Related assets from parsed log",
        *related_assets_lines,
        "",
        "## Research Plan",
        plan_markdown,
    ])


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
        parse_result = state["parse_result"]
        available_tools = [t.name for t in get_tool_availability(self.settings, self.mcp_manager) if t.available]
        prompt = self.prompt_manager.load_active("research_plan")
        model_calls = list(state.get("model_calls", []))
        plans: list[str] = []
        for idx, judgement in enumerate(parse_result.all_judgements(), start=1):
            context = {
                "judgement_index": idx,
                "judgement_total": len(parse_result.all_judgements()),
                "judgement": judgement.model_dump(mode="json"),
                "available_tools": available_tools,
                "verify_feedback": list(state.get("verify_suggestions", [])),
            }
            messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, context=context)
            plan_text, trace = self.reporter_agent.gateway.invoke_text(
                messages=messages,
                purpose=ModelCallPurpose.EVIDENCE_PLANNING,
                prompt_version=prompt.prompt_name,
                input_summary=f"single_judgement={idx}/{len(parse_result.all_judgements())}",
            )
            plans.append(plan_text)
            if trace is not None:
                model_calls.append(trace.model_dump(mode="json"))
        return {
            "per_judgement_plans": plans,
            "research_plan_markdown": "\n\n".join(plans),
            "research_retry_count": int(state.get("research_retry_count", 0)),
            "is_sufficient": False,
            "verify_suggestions": list(state.get("verify_suggestions", [])),
            "model_calls": model_calls,
        }

    def execute_collection_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result = state["parse_result"]
        judgements = parse_result.all_judgements()
        plan_list = list(state.get("per_judgement_plans", []))
        prompt = self.prompt_manager.load_active("research_agent")

        all_messages = list(state.get("agent_messages", []))
        grouped_messages: dict[str, list[str]] = dict(state.get("agent_message_groups", {}))
        all_tool_calls = list(state.get("tool_calls", []))
        all_react_steps = list(state.get("react_steps", []))
        all_evidence_items = list(state.get("accumulated_evidence_items", []))
        total_failures = int(state.get("accumulated_tool_failures", 0))
        researched_items = []

        for idx, judgement in enumerate(judgements, start=1):
            runtime = ToolRuntime()
            tools = build_runtime_tools(self.settings, self.mcp_manager, runtime)
            agent = create_agent(model=self.chat_model, tools=tools, system_prompt=prompt.system_prompt)
            plan_markdown = plan_list[idx - 1] if idx - 1 < len(plan_list) else ""
            task_markdown = _render_single_judgement_research_task_markdown(
                judgement=judgement,
                plan_markdown=plan_markdown,
                judgement_index=idx,
                total_judgements=len(judgements),
                related_assets=[item.target_asset for item in parse_result.trade_actions],
            )
            result = agent.invoke({"messages": [HumanMessage(content=task_markdown)]})
            messages = [_extract_message_text(m).strip() for m in result.get("messages", []) if _extract_message_text(m).strip()]
            all_messages.extend(messages)
            grouped_messages[f"judgement_{idx}"] = messages
            final_markdown = next((m for m in reversed(messages) if "# judgement evidence" in m.lower()), messages[-1] if messages else "")
            parsed = parse_single_research_output_text(final_markdown, fallback_judgement=judgement)
            researched_items.append(build_researched_judgement(fallback=judgement, parsed=parsed, evidence_items=runtime.evidence_items))
            all_tool_calls.extend([t.model_dump(mode="json") for t in runtime.tool_traces])
            all_react_steps.extend([s.model_dump(mode="json") for s in runtime.react_steps])
            all_evidence_items.extend(runtime.evidence_items)
            total_failures += sum(1 for t in runtime.tool_traces if not t.success)

        research_output = build_research_output_from_items(researched_items)
        research_output.validate_against(judgements)
        evidence_packet = build_evidence_packet(packet_id=f"packet_{req.run_id}", user_id=req.user_id, evidence_items=all_evidence_items)
        return {
            "agent_messages": all_messages,
            "agent_message_groups": grouped_messages,
            "evidence_packet": evidence_packet,
            "tool_calls": all_tool_calls,
            "react_steps": all_react_steps,
            "research_output": research_output,
            "accumulated_evidence_items": all_evidence_items,
            "accumulated_tool_failures": total_failures,
        }

    def verify_information_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        retry_count = int(state.get("research_retry_count", 0)) + 1
        research_output: ResearchOutput = state["research_output"]
        tool_calls = list(state.get("tool_calls", []))
        issues: list[str] = []
        for idx, judgement in enumerate(research_output.judgements, start=1):
            evidence = judgement.evidence
            source_count = sum(len(item.sources) for item in evidence.collected_evidence_items)
            has_items = bool(evidence.collected_evidence_items)
            if not has_items and evidence.evidence_quality == "sufficient":
                issues.append(f"judgement {idx}: sufficient without evidence items")
            if evidence.support_signal in {"support", "oppose"} and evidence.evidence_quality == "sufficient" and source_count < 1:
                issues.append(f"judgement {idx}: sufficient {evidence.support_signal} without sources")
            if not has_items and evidence.support_signal in {"support", "oppose"}:
                issues.append(f"judgement {idx}: directional conclusion without evidence")

        failed_calls = sum(1 for call in tool_calls if not bool(call.get("success", True)))
        if failed_calls == len(tool_calls) and failed_calls > 0:
            issues.append("all tool calls failed")

        sufficient = not issues
        suggestions = [] if sufficient else ["For each judgement, add direct evidence (article body, market data row, or primary source) and avoid directional conclusions without it."]
        return {
            "is_sufficient": sufficient,
            "verify_suggestions": suggestions,
            "research_retry_count": retry_count,
            "continue_collection": (not sufficient) and retry_count < self.settings.react_max_iterations,
            "insufficiency_reason": "" if sufficient else "; ".join(issues),
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
        trace = RunTrace(run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, trigger_type=req.trigger_type, started_at=state.get("run_started_at", utc_now()), ended_at=utc_now(), module_spans=[ModuleRunSpan(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)], model_calls=state.get("model_calls", []), tool_calls=state.get("tool_calls", []), evidence_sources=state["evidence_packet"].source_registry, rewrite_rounds=int(state.get("rewrite_count", 0)), debug_context={"research_plan_markdown": state.get("research_plan_markdown", ""), "report_context": state.get("report_context", {})}, react_steps=state.get("react_steps", []))
        final = TaskResult(run_id=req.run_id, status=RunStatus.SUCCESS, step_results=[StepResult(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)], report=report, evaluation=EvaluationResult(evaluation_id=f"eval_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date, summary="Daily judgement feedback created.", hypothesis_assessments=[HypothesisAssessment(hypothesis_id="daily_feedback", category=EvaluationCategory.PARTIAL, commentary="Ready for long-term evaluation")]), position_snapshot=PositionSnapshot(snapshot_id=f"ps_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date), pnl_snapshot=PnLSnapshot(snapshot_id=f"pnl_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date), memory_write_results=memory_results, trace=trace)
        return {"final_result": final}

    def finalize_failure(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        final = TaskResult(run_id=state["request"].run_id, status=RunStatus.FAILED, step_results=[StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.FAILED)], errors=[ErrorRecord(module_name=ModuleName.EVALUATOR, error_code="REPORT_VALIDATION_FAILED", message="Judge failed and rewrite budget exhausted.", recoverable=False)])
        return {"final_result": final}
