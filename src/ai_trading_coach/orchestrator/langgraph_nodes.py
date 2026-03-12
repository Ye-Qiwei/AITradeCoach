"""LangGraph nodes for daily parse->research->report->judge pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import EvaluationCategory, ModuleName, RunStatus
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, LongTermJudgementRecord, ResearchOutput, compute_due_date
from ai_trading_coach.domain.llm_output_adapters import research_agent_contract_to_domain
from ai_trading_coach.domain.llm_output_contracts import ResearchAgentFinalContract
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
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime
from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.evidence_packet_builder import build_evidence_packet
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _extract_agent_messages(raw_messages: list[Any]) -> dict[str, list[str]]:
    groups = {
        "input_messages": [],
        "intermediate_messages": [],
        "tool_error_messages": [],
        "final_messages": [],
    }
    for message in raw_messages:
        text = _extract_message_text(message).strip()
        if not text:
            continue
        lowered = text.lower()
        if "tool_error:" in lowered or "error invoking tool" in lowered:
            groups["tool_error_messages"].append(text)
        elif lowered.startswith("{") and '"task"' in lowered:
            groups["input_messages"].append(text)
        elif "returning structured response" in lowered or lowered.startswith("{"):
            groups["final_messages"].append(text)
        else:
            groups["intermediate_messages"].append(text)
    return groups


def _parse_final_contract(result: dict[str, Any]) -> ResearchAgentFinalContract:
    expected_keys = sorted(ResearchAgentFinalContract.model_fields.keys())
    structured = result.get("structured_response")
    if structured is not None:
        return ResearchAgentFinalContract.model_validate(structured)

    groups = _extract_agent_messages(result.get("messages", []))
    messages = groups["final_messages"] or groups["intermediate_messages"]
    if not messages:
        raise ValueError("Research agent produced no output messages.")

    raw_tail = messages[-1]
    snippet = raw_tail if len(raw_tail) <= 400 else f"{raw_tail[:400]}...(truncated)"
    json_parse_ok = False
    actual_keys: list[str] = []
    try:
        parsed = json.loads(raw_tail)
        json_parse_ok = True
        if isinstance(parsed, dict):
            actual_keys = sorted(parsed.keys())
        else:
            actual_keys = [f"<non-object:{type(parsed).__name__}>"]
        return ResearchAgentFinalContract.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(
            "Failed to parse research final contract. "
            f"structured_response_present={structured is not None}; "
            f"json_parse_ok={json_parse_ok}; "
            f"expected_top_level_keys={expected_keys}; "
            f"actual_top_level_keys={actual_keys}; "
            f"last_message_snippet={snippet}"
        ) from exc


def _normalize_research_output_evidence_ids(
    research_output: ResearchOutput,
    *,
    all_items: list[Any],
) -> None:
    alias_to_item_id: dict[str, str] = {}
    for item in all_items:
        item_id = getattr(item, "item_id", "")
        if not item_id:
            continue
        alias_to_item_id[item_id] = item_id
        for source in getattr(item, "sources", []):
            uri = getattr(source, "uri", None)
            if isinstance(uri, str) and uri.strip():
                alias_to_item_id[uri.strip()] = item_id

    for judgement in research_output.judgement_evidence:
        normalized: list[str] = []
        for raw_id in judgement.evidence_item_ids:
            mapped = alias_to_item_id.get(raw_id, raw_id)
            if mapped not in normalized:
                normalized.append(mapped)
        judgement.evidence_item_ids = normalized


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
        model_calls.append(parse_trace.model_dump(mode="json"))
        return {"parse_result": parse_result, "model_calls": model_calls}


    def plan_research_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        return {
            "research_retry_count": int(state.get("research_retry_count", 0)),
            "is_sufficient": False,
            "verify_suggestions": list(state.get("verify_suggestions", [])),
        }

    def execute_collection_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result = state["parse_result"]
        runtime = MCPToolRuntime()
        tools = build_runtime_research_tools(
            settings=self.settings,
            mcp_manager=self.mcp_manager,
            runtime=runtime,
        )
        prompt = self.prompt_manager.load_active("research_agent")
        agent = create_agent(
            model=self.chat_model,
            tools=tools,
            system_prompt=prompt.system_prompt,
            response_format=ToolStrategy(ResearchAgentFinalContract),
        )
        payload = {
            "task": (
                "Collect evidence to evaluate each judgement_id exactly once. "
                "Return strict ResearchAgentFinalContract JSON only."
            ),
            "verify_suggestions": state.get("verify_suggestions", []),
            "judgements": [
                {
                    "judgement_id": j.judgement_id,
                    "category": j.category,
                    "target": j.target,
                    "thesis": j.thesis,
                    "evaluation_window": j.evaluation_window,
                    "dependencies": j.dependencies,
                }
                for j in parse_result.all_judgements()
            ],
        }
        result = agent.invoke({"messages": [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]})
        accumulated_items = list(state.get("accumulated_evidence_items", [])) + list(runtime.evidence_items)
        evidence_packet = build_evidence_packet(packet_id=f"packet_{req.run_id}", user_id=req.user_id, evidence_items=accumulated_items)
        final_contract = _parse_final_contract(result)
        synthesis_out = research_agent_contract_to_domain(final_contract, run_id=req.run_id)
        research_output = ResearchOutput.model_validate(synthesis_out.model_dump(mode="json"))
        all_items = [
            *evidence_packet.price_evidence,
            *evidence_packet.news_evidence,
            *evidence_packet.filing_evidence,
            *evidence_packet.sentiment_evidence,
            *evidence_packet.market_regime_evidence,
            *evidence_packet.discussion_evidence,
            *evidence_packet.analog_evidence,
            *evidence_packet.macro_evidence,
        ]
        _normalize_research_output_evidence_ids(research_output, all_items=all_items)
        research_output.validate_against({j.judgement_id for j in parse_result.all_judgements()}, {item.item_id for item in all_items})
        message_groups = _extract_agent_messages(result.get("messages", []))
        source_count = len(evidence_packet.source_registry)
        if research_output.judgement_evidence and source_count == 0:
            message_groups["final_messages"].append("structured response returned without evidence sources")
        return {
            "agent_messages": list(state.get("agent_messages", [])) + [
                *message_groups["input_messages"],
                *message_groups["intermediate_messages"],
                *message_groups["tool_error_messages"],
                *message_groups["final_messages"],
            ],
            "agent_message_groups": message_groups,
            "evidence_packet": evidence_packet,
            "tool_calls": list(state.get("tool_calls", [])) + [t.model_dump(mode="json") for t in runtime.tool_traces],
            "react_steps": list(state.get("react_steps", [])) + [s.model_dump(mode="json") for s in runtime.react_steps],
            "research_output": research_output,
            "accumulated_evidence_items": accumulated_items,
            "accumulated_tool_failures": int(state.get("accumulated_tool_failures", 0)) + sum(1 for t in runtime.tool_traces if not t.success),
            "research_anomalies": ["structured_response_without_evidence"] if (research_output.judgement_evidence and source_count == 0) else [],
        }

    def verify_information_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        retry_count = int(state.get("research_retry_count", 0)) + 1
        research_output = state["research_output"]
        source_count = len({src.source_id for src in state["evidence_packet"].source_registry})
        all_judgement_ids = {j.judgement_id for j in state["parse_result"].all_judgements()}
        covered = {item.judgement_id for item in research_output.judgement_evidence if item.evidence_item_ids}
        tool_failures = int(state.get("accumulated_tool_failures", 0))
        sufficient = all_judgement_ids.issubset(covered) and source_count >= self.settings.react_require_min_sources
        should_continue = (not sufficient) and retry_count < self.settings.react_max_iterations and tool_failures < self.settings.react_max_tool_failures
        insufficiency_reason = "" if sufficient else f"coverage={len(covered)}/{len(all_judgement_ids)}; sources={source_count}"
        return {
            "is_sufficient": sufficient,
            "verify_suggestions": [] if sufficient else ["Increase source diversity and judgement coverage."],
            "research_retry_count": retry_count,
            "insufficiency_reason": insufficiency_reason,
            "continue_collection": should_continue,
        }

    def route_after_verify(self, state: OrchestratorGraphState) -> str:
        if state.get("is_sufficient", False):
            return "research_done"
        if state.get("continue_collection", False):
            return "continue_collection"
        return "research_done"

    def build_report_context(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        context = self.context_builder.for_reporter(parse_result=state["parse_result"], research_output=state["research_output"], evidence_packet=state["evidence_packet"])
        return {"report_context": context}

    def generate_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        out, trace = self.reporter_agent.generate(evidence_packet=state["evidence_packet"], report_context=state["report_context"], rewrite_instruction=state.get("rewrite_instruction"))
        expected = {j.judgement_id for j in state["parse_result"].all_judgements()}
        got = [f.judgement_id for f in out.judgement_feedback]
        if set(got) != expected or len(got) != len(set(got)):
            raise ValueError("ReporterOutput judgement_feedback ids must match parser judgements exactly.")
        model_calls = list(state.get("model_calls", []))
        model_calls.append(trace.model_dump(mode="json"))
        return {"report_draft": out.markdown, "judgement_feedback": out.judgement_feedback, "model_calls": model_calls}

    def judge_report(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        judge_ctx = self.context_builder.for_judge(report_markdown=state["report_draft"], judgement_feedback=state.get("judgement_feedback", []), parse_result=state["parse_result"], research_output=state["research_output"], report_context=state.get("report_context", {}))
        verdict, trace = self.report_judge.evaluate(report_markdown=state["report_draft"], judge_context=judge_ctx, evidence_packet=state["evidence_packet"])
        model_calls = list(state.get("model_calls", []))
        model_calls.append(trace.model_dump(mode="json"))
        rewrites_used = int(state.get("rewrite_count", 0))
        if not verdict.passed:
            rewrites_used += 1
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
        markdown = state["report_draft"]
        feedback = [DailyJudgementFeedback.model_validate(i) for i in state.get("judgement_feedback", [])]
        feedback_by_id = {f.judgement_id: f for f in feedback}
        records = [
            LongTermJudgementRecord(
                judgement_id=j.judgement_id,
                user_id=req.user_id,
                run_id=req.run_id,
                run_date=req.run_date,
                due_date=compute_due_date(req.run_date, feedback_by_id[j.judgement_id].evaluation_window),
                judgement=j,
                initial_feedback=feedback_by_id[j.judgement_id],
            )
            for j in state["parse_result"].all_judgements()
        ]
        memory_results: list[MemoryWriteResult] = []
        if not req.options.dry_run:
            self.long_term_store.upsert_records(records)
            memory_results = [MemoryWriteResult(collection="long_term_memory", memory_ids=[r.judgement_id for r in records])]
        report = DailyReviewReport(report_id=f"report_{req.run_id}", user_id=req.user_id, report_date=req.run_date, title=f"Daily Trading Review - {req.run_date}", sections=[ReportSection(title="Daily Feedback", content=markdown)], generated_prompt_version=self.settings.prompt_version, markdown_body=markdown if markdown.endswith("\n") else markdown + "\n")
        trace = RunTrace(
            run_id=req.run_id,
            user_id=req.user_id,
            run_date=req.run_date,
            trigger_type=req.trigger_type,
            started_at=state.get("run_started_at", utc_now()),
            ended_at=utc_now(),
            module_spans=[ModuleRunSpan(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), ModuleRunSpan(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)],
            model_calls=state.get("model_calls", []),
            tool_calls=state.get("tool_calls", []),
            evidence_sources=state["evidence_packet"].source_registry,
            rewrite_rounds=int(state.get("rewrite_count", 0)),
            debug_context={"report_context": state.get("report_context", {}), "research_output": state["research_output"].model_dump(mode="json")},
            react_steps=state.get("react_steps", []),
        )
        final = TaskResult(
            run_id=req.run_id,
            status=RunStatus.SUCCESS,
            step_results=[StepResult(module_name=ModuleName.LOG_INTAKE, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.MCP_GATEWAY, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.REPORT_GENERATOR, status=RunStatus.SUCCESS), StepResult(module_name=ModuleName.EVALUATOR, status=RunStatus.SUCCESS)],
            report=report,
            evaluation=EvaluationResult(evaluation_id=f"eval_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date, summary="Daily judgement feedback created.", hypothesis_assessments=[HypothesisAssessment(hypothesis_id="daily_feedback", category=EvaluationCategory.PARTIAL, commentary="Ready for long-term evaluation")]),
            position_snapshot=PositionSnapshot(snapshot_id=f"ps_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date),
            pnl_snapshot=PnLSnapshot(snapshot_id=f"pnl_{req.run_id}", user_id=req.user_id, as_of_date=req.run_date),
            memory_write_results=memory_results,
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
