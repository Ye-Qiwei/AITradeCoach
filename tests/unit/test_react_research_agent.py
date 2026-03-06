from __future__ import annotations

import json
from datetime import date, datetime, timezone

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.llm.provider import LLMCallRecord
from ai_trading_coach.modules.agent import (
    CombinedParserAgent,
    ContextBuilderV2,
    ExecutorEngine,
    PlannerAgent,
    ReActResearchAgent,
    ReportJudge,
    ReporterAgent,
)
from ai_trading_coach.modules.agent.react_tools import ReactResearchTools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator import OrchestratorModules, PipelineOrchestrator


def _combined_parse_payload() -> dict[str, object]:
    return {
        "parse_id": "parse_rx",
        "normalized_log": {
            "log_id": "log_rx",
            "user_id": "u1",
            "log_date": "2026-03-05",
            "traded_tickers": ["AAPL.US"],
            "mentioned_tickers": ["AAPL.US"],
            "user_state": {"emotion": "calm", "stress": 2, "focus": 8},
            "market_context": {"regime": "risk_on", "key_variables": ["rates"]},
            "trade_events": [],
            "trade_narratives": [],
            "scan_signals": {"anxiety": [], "fomo": [], "not_trade": []},
            "reflection": {"facts": ["price stabilized"], "gaps": [], "lessons": ["wait"]},
            "ai_directives": ["@AI 看是否过度乐观"],
            "raw_text": "sample",
            "field_errors": [],
        },
        "cognition_state": {
            "cognition_id": "cog_rx",
            "log_id": "log_rx",
            "user_id": "u1",
            "as_of_date": "2026-03-05",
            "core_judgements": ["trend improving"],
            "hypotheses": [],
            "risk_concerns": ["macro surprise"],
            "outside_opportunities": [],
            "deliberate_no_trade_decisions": [],
            "explicit_rules": ["keep size moderate"],
            "fuzzy_tendencies": [],
            "fact_statements": ["price stabilized"],
            "subjective_statements": [],
            "behavioral_signals": [],
            "emotion_signals": [],
            "user_intent_signals": [{"intent_id": "i1", "question": "是否过度自信", "priority": 4}],
        },
    }


class ScenarioProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_call: LLMCallRecord | None = None

    def __init__(self, *, summary_text: str = "done") -> None:
        self.summary_text = summary_text

    def chat_json(
        self,
        schema_name: str,
        messages: list[dict[str, str]],
        timeout: float,
        prompt_version: str | None = None,
    ) -> dict[str, object]:
        del timeout
        now = datetime.now(timezone.utc)
        if schema_name == "react_research_decision.v1":
            payload = json.loads(messages[-1]["content"])
            evidence_count = int(payload["evidence_count"])
            recent = payload["recent_steps"]
            last_obs = str(recent[-1].get("observation_summary", "")) if recent else ""
            if "allowlist" in last_obs:
                out = {"thought": "价格工具失败，改走新闻", "action": "search_news", "action_input": {"tickers": ["AAPL.US"]}}
            elif evidence_count == 0:
                out = {"thought": "先看价格", "action": "get_price_history", "action_input": {"tickers": ["AAPL.US"], "query": {"window": "5d"}}}
            else:
                out = {"thought": "信息足够，结束", "action": "finish_research", "action_input": {}}
        elif schema_name == "react_research_summary.v1":
            out = {
                "investigation_summary": self.summary_text,
                "key_findings": ["finding-1"],
                "open_questions": ["question-1"],
            }
        elif schema_name == "plan.v1":
            out = {"plan_id": "plan_rx", "subtasks": [], "risk_uncertainties": [], "follow_up_triggers": ["watch"]}
        elif schema_name == "combined_parse_result.v1":
            out = _combined_parse_payload()
        elif schema_name == "reporter_draft.v1":
            payload = json.loads(messages[-1]["content"])
            source_index = payload.get("source_index", [])
            source_id = source_index[0].get("source_id") if source_index else "src_missing"
            out = {"markdown": f"# Daily Review Report\n## Summary\n- ok [source:{source_id}]\n## Evidence\n- e [source:{source_id}]\n## Key Risks\n- r [source:{source_id}]\n## Actions\n- a [source:{source_id}]"}
        elif schema_name == "judge_verdict.v1":
            out = {
                "passed": True,
                "reasons": [],
                "rewrite_instruction": None,
                "contradiction_flags": [],
                "citation_coverage": 1.0,
            }
        else:
            raise AssertionError(f"unexpected schema={schema_name}")

        self.last_call = LLMCallRecord(
            provider_name=self.provider_name,
            model_name=self.model_name,
            schema_name=schema_name,
            prompt_version=prompt_version,
            started_at=now,
            ended_at=now,
            latency_ms=3,
            response_size=len(str(out)),
            token_in=11,
            token_out=17,
            error=None,
        )
        return out

    def chat_text(self, messages: list[dict[str, str]], prompt_version: str | None = None) -> str:
        del messages, prompt_version
        return ""


def test_react_agent_runs_multi_turn_and_finish_research() -> None:
    provider = ScenarioProvider(summary_text="multi-turn complete")
    settings = Settings(
        _env_file=None,
        react_max_iterations=4,
        react_max_tool_failures=2,
        react_require_min_sources=1,
        mcp_servers_json='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        evidence_tool_map_json='{"price_path":"srv:price_tool","news":"srv:news_tool","filing":"srv:filing_tool","macro":"srv:macro_tool"}',
        mcp_tool_allowlist_csv="srv:price_tool,srv:news_tool,srv:filing_tool,srv:macro_tool",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del server_id, arguments
        return [{"title": tool_name, "ticker": "AAPL.US", "price": 100.0}]

    manager = MCPClientManager(settings=settings, invoker=invoker)
    tools = ReactResearchTools(mcp_manager=manager)
    planner = PlannerAgent(provider=provider, timeout_seconds=5)
    parser = CombinedParserAgent(provider=provider, timeout_seconds=5)
    parse_result, _ = parser.parse(
        run_id="run_1",
        user_id="u1",
        run_date=date(2026, 3, 5),
        raw_log_text="sample",
    )
    agent = ReActResearchAgent(provider=provider, tools=tools, settings=settings, planner_agent=planner)

    summary = agent.run(request_id="run_1", user_id="u1", parse_result=parse_result)

    assert len(summary.tool_steps) >= 2
    assert summary.tool_steps[-1].action == "finish_research"
    assert summary.investigation_summary == "multi-turn complete"


def test_observation_changes_next_tool_and_failure_budget_keeps_agent_alive() -> None:
    provider = ScenarioProvider(summary_text="fallback complete")
    settings = Settings(
        _env_file=None,
        react_max_iterations=4,
        react_max_tool_failures=2,
        react_require_min_sources=1,
        mcp_servers_json='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        evidence_tool_map_json='{"price_path":"srv:price_tool","news":"srv:news_tool"}',
        mcp_tool_allowlist_csv="srv:news_tool",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del server_id, arguments
        return [{"title": tool_name, "ticker": "AAPL.US"}]

    manager = MCPClientManager(settings=settings, invoker=invoker)
    tools = ReactResearchTools(mcp_manager=manager)
    planner = PlannerAgent(provider=provider, timeout_seconds=5)
    parse_result, _ = CombinedParserAgent(provider=provider, timeout_seconds=5).parse(
        run_id="run_2",
        user_id="u1",
        run_date=date(2026, 3, 5),
        raw_log_text="sample",
    )
    agent = ReActResearchAgent(provider=provider, tools=tools, settings=settings, planner_agent=planner)

    summary = agent.run(request_id="run_2", user_id="u1", parse_result=parse_result)

    assert len(summary.tool_steps) >= 2
    assert summary.tool_steps[0].action == "get_price_history"
    assert summary.tool_steps[0].success is False
    assert summary.tool_steps[1].action == "search_news"
    assert summary.tool_steps[1].success is True


def test_pipeline_still_returns_task_result_with_reporter_and_judge() -> None:
    provider = ScenarioProvider(summary_text="pipeline complete")
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="test-key",
        react_max_iterations=4,
        react_max_tool_failures=2,
        react_require_min_sources=1,
        mcp_servers_json='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        evidence_tool_map_json='{"price_path":"srv:price_tool","news":"srv:news_tool","filing":"srv:filing_tool","macro":"srv:macro_tool"}',
        mcp_tool_allowlist_csv="srv:price_tool,srv:news_tool,srv:filing_tool,srv:macro_tool",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del server_id, arguments
        return [{"title": tool_name, "ticker": "AAPL.US"}]

    manager = MCPClientManager(settings=settings, invoker=invoker)
    modules = OrchestratorModules(
        parser_agent=CombinedParserAgent(provider=provider, timeout_seconds=5),
        planner_agent=PlannerAgent(provider=provider, timeout_seconds=5),
        executor_engine=ExecutorEngine(mcp_manager=manager),
        reporter_agent=ReporterAgent(provider=provider, timeout_seconds=5),
        report_judge=ReportJudge(provider=provider, timeout_seconds=5),
        context_builder=ContextBuilderV2(settings=settings),
        react_research_agent=ReActResearchAgent(
            provider=provider,
            tools=ReactResearchTools(mcp_manager=manager),
            settings=settings,
            planner_agent=PlannerAgent(provider=provider, timeout_seconds=5),
        ),
    )

    result = PipelineOrchestrator(modules=modules, settings=settings).run(
        ReviewRunRequest(
            run_id="run_pipeline",
            user_id="u1",
            run_date=date(2026, 3, 5),
            trigger_type=TriggerType.MANUAL,
            raw_log_text="sample",
        )
    )

    assert result.status == RunStatus.SUCCESS
    assert result.report is not None
    assert result.evaluation is not None
    assert result.trace is not None
    assert len(result.trace.react_steps) >= 2
