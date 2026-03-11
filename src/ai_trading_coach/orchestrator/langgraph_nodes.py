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
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, build_langchain_mcp_tools
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.react_tools import build_evidence_packet
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_agent_messages(raw_messages: list[Any]) -> list[str]:
    out: list[str] = []
    for message in raw_messages:
        content = getattr(message, "content", message)
        out.append(content if isinstance(content, str) else str(content))
    return out


def _parse_final_contract(result: dict[str, Any]) -> ResearchAgentFinalContract:
    structured = result.get("structured_response")
    if structured is not None:
        return ResearchAgentFinalContract.model_validate(structured)
    messages = _extract_agent_messages(result.get("messages", []))
    if not messages:
        raise ValueError("Research agent produced no output messages.")
    return ResearchAgentFinalContract.model_validate(json.loads(messages[-1]))


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
    llm_gateway: LangChainLLMGateway
    prompt_manager: PromptManager

    def parse_log(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result, parse_trace = self.parser_agent.parse(run_id=req.run_id, user_id=req.user_id, run_date=req.run_date, raw_log_text=req.raw_log_text)
        model_calls = list(state.get("model_calls", []))
        model_calls.append(parse_trace.model_dump(mode="json"))
        return {"parse_result": parse_result, "model_calls": model_calls}


    def react_research(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        rolling = dict(state)
        rolling.update(self.plan_research_node(rolling))
        while True:
            rolling.update(self.execute_collection_node(rolling))
            rolling.update(self.verify_information_node(rolling))
            if self.route_after_verify(rolling) == "research_done":
                break
        return {
            "agent_messages": rolling.get("agent_messages", []),
            "evidence_packet": rolling["evidence_packet"],
            "tool_calls": rolling.get("tool_calls", []),
            "react_steps": rolling.get("react_steps", []),
            "research_output": rolling["research_output"],
            "analysis_framework": rolling.get("analysis_framework", ""),
            "analysis_directions": rolling.get("analysis_directions", []),
            "info_requirements": rolling.get("info_requirements", []),
            "collected_info": rolling.get("collected_info", []),
            "is_sufficient": rolling.get("is_sufficient", True),
            "verify_suggestions": rolling.get("verify_suggestions", []),
            "research_retry_count": rolling.get("research_retry_count", 0),
        }

    def plan_research_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        parse_result = state["parse_result"]
        atomic_items: list[dict[str, Any]] = []
        for judgement in parse_result.all_judgements():
            for atomic in judgement.atomic_judgements:
                atomic_items.append(
                    {
                        "judgement_id": judgement.judgement_id,
                        "atomic_id": atomic.id,
                        "core_thesis": atomic.core_thesis,
                        "evaluation_timeframe": atomic.evaluation_timeframe,
                        "dependencies": atomic.dependencies,
                    }
                )
        analysis_directions = [
            "market_pricing_and_price_path",
            "fundamental_or_macro_driver_validation",
            "event_news_and_policy_signal_crosscheck",
        ]
        info_requirements = [
            {"requirement_id": f"req_{idx}", "judgement_id": item["judgement_id"], "atomic_id": item["atomic_id"], "need": item["core_thesis"]}
            for idx, item in enumerate(atomic_items, start=1)
        ]
        return {
            "analysis_framework": "Hypothesis decomposition -> evidence collection -> sufficiency verification",
            "analysis_directions": analysis_directions,
            "info_requirements": info_requirements,
            "collected_info": list(state.get("collected_info", [])),
            "research_retry_count": int(state.get("research_retry_count", 0)),
            "is_sufficient": False,
            "verify_suggestions": list(state.get("verify_suggestions", [])),
        }

    def execute_collection_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        req = state["request"]
        parse_result = state["parse_result"]
        runtime = MCPToolRuntime()
        tools = build_langchain_mcp_tools(mcp_manager=self.mcp_manager, runtime=runtime)
        prompt = self.prompt_manager.load_active("research_agent")
        agent = create_agent(model=self.chat_model, tools=tools, system_prompt=prompt.system_prompt)
        payload = {
            "task": "Collect evidence for requirements and output final strict JSON.",
            "analysis_framework": state.get("analysis_framework", ""),
            "analysis_directions": state.get("analysis_directions", []),
            "info_requirements": state.get("info_requirements", []),
            "collected_info": state.get("collected_info", []),
            "verify_suggestions": state.get("verify_suggestions", []),
            "judgements": [
                {
                    "judgement_id": j.judgement_id,
                    "category": j.category,
                    "target_asset_or_topic": j.target_asset_or_topic,
                    "thesis": j.thesis,
                    "atomic_judgements": [a.model_dump(mode="json") for a in j.atomic_judgements],
                    "evidence_from_user_log": j.evidence_from_user_log,
                    "implicitness": j.implicitness,
                    "proposed_evaluation_window": j.proposed_evaluation_window,
                }
                for j in parse_result.all_judgements()
            ],
        }
        result = agent.invoke({"messages": [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]})
        evidence_packet = build_evidence_packet(packet_id=f"packet_{req.run_id}", user_id=req.user_id, evidence_items=runtime.evidence_items)
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
        research_output.validate_against({j.judgement_id for j in parse_result.all_judgements()}, {item.item_id for item in all_items})
        collected_info = list(state.get("collected_info", []))
        collected_info.extend(
            [{"item_id": item.item_id, "summary": item.summary} for item in runtime.evidence_items]
        )
        return {
            "agent_messages": _extract_agent_messages(result.get("messages", [])),
            "evidence_packet": evidence_packet,
            "tool_calls": [t.model_dump(mode="json") for t in runtime.tool_traces],
            "react_steps": [s.model_dump(mode="json") for s in runtime.react_steps],
            "research_output": research_output,
            "collected_info": collected_info,
        }

    def verify_information_node(self, state: OrchestratorGraphState) -> OrchestratorGraphState:
        required = state.get("info_requirements", [])
        collected = state.get("collected_info", [])
        retry_count = int(state.get("research_retry_count", 0)) + 1
        sufficient = bool(collected) and len(collected) >= min(len(required), 3)
        suggestions: list[str] = []
        if not sufficient:
            suggestions = [
                "Broaden keyword set with synonyms and event-specific terms",
                "Use brave_search first, then firecrawl_extract for full text",
                "If static fetch is weak, switch to playwright_fetch for dynamic pages",
            ]
            if retry_count >= 3:
                sufficient = True
                suggestions.append("Max retry reached; proceed with uncertainty annotation.")
        return {
            "is_sufficient": sufficient,
            "verify_suggestions": suggestions,
            "research_retry_count": retry_count,
        }

    def route_after_verify(self, state: OrchestratorGraphState) -> str:
        return "research_done" if state.get("is_sufficient", False) else "continue_collection"

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
        judge_ctx = self.context_builder.for_judge(report_markdown=state["report_draft"], judgement_feedback=state.get("judgement_feedback", []), parse_result=state["parse_result"], research_output=state["research_output"])
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
            started_at=utc_now(),
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
