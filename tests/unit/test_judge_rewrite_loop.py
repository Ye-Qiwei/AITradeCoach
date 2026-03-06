from __future__ import annotations

from datetime import date

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import CombinedParseResult, JudgeVerdict, ReporterDraft
from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.modules.agent.context_builder_v2 import ContextBuilderV2
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_graph import build_review_graph
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class StubParser:
    def parse(self, *, run_id: str, user_id: str, run_date: date, raw_log_text: str):
        del raw_log_text
        payload = {
            "parse_id": f"parse_{run_id}",
            "normalized_log": {
                "log_id": f"log_{run_id}",
                "user_id": user_id,
                "log_date": run_date.isoformat(),
                "traded_tickers": ["AAPL.US"],
                "mentioned_tickers": ["AAPL.US"],
                "user_state": {"emotion": "calm", "stress": 3, "focus": 7},
                "market_context": {"regime": "risk_on", "key_variables": ["rates"]},
                "trade_events": [],
                "trade_narratives": [],
                "scan_signals": {"anxiety": [], "fomo": [], "not_trade": []},
                "reflection": {"facts": ["sample"], "gaps": [], "lessons": []},
                "ai_directives": [],
                "raw_text": "sample",
                "field_errors": [],
            },
            "cognition_state": {
                "cognition_id": f"cog_{run_id}",
                "log_id": f"log_{run_id}",
                "user_id": user_id,
                "as_of_date": run_date.isoformat(),
                "core_judgements": ["trend"],
                "hypotheses": [],
                "risk_concerns": ["macro"],
                "outside_opportunities": [],
                "deliberate_no_trade_decisions": [],
                "explicit_rules": [],
                "fuzzy_tendencies": [],
                "fact_statements": [],
                "subjective_statements": [],
                "behavioral_signals": [],
                "emotion_signals": [],
                "user_intent_signals": [{"intent_id": "i1", "question": "风险在哪里", "priority": 4}],
            },
        }
        return CombinedParseResult.model_validate(payload), None


class StubReporter:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, evidence_packet, report_context, intent, rewrite_instruction=None):
        del evidence_packet, report_context, intent, rewrite_instruction
        self.calls += 1
        return ReporterDraft(markdown="# Daily Review Report\n## Summary\n- ok"), None


class SequenceJudge:
    def __init__(self, seq: list[bool]) -> None:
        self.seq = seq

    def evaluate(self, *, report_markdown, judge_context, intent, evidence_packet):
        del report_markdown, judge_context, intent, evidence_packet
        passed = self.seq.pop(0)
        return (
            JudgeVerdict(
                passed=passed,
                reasons=[] if passed else ["rewrite"],
                rewrite_instruction=None if passed else "rewrite",
                contradiction_flags=[],
                citation_coverage=1.0,
            ),
            None,
        )


class FakeReactAgent:
    def invoke(self, payload):
        return payload | {"messages": payload["messages"]}


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="k",
        mcp_servers_json='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        evidence_tool_map_json='{"price_path":"srv:price_tool","news":"srv:news_tool","filing":"srv:filing_tool","macro":"srv:macro_tool"}',
        mcp_tool_allowlist_csv="srv:price_tool,srv:news_tool,srv:filing_tool,srv:macro_tool",
        agent_max_rewrite_rounds=2,
    )


def test_graph_conditional_edge_rewrites_then_passes(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_react_agent", lambda *_: FakeReactAgent())
    settings = _settings()
    reporter = StubReporter()
    runtime = LangGraphNodeRuntime(
        parser_agent=StubParser(),
        reporter_agent=reporter,
        report_judge=SequenceJudge([False, True]),
        context_builder=ContextBuilderV2(settings=settings),
        mcp_manager=MCPClientManager(settings=settings, invoker=lambda *_: []),
        chat_model=object(),
        settings=settings,
    )
    graph = build_review_graph(runtime)
    request = ReviewRunRequest(
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 3, 5),
        trigger_type=TriggerType.MANUAL,
        raw_log_text="sample",
    )

    result = graph.invoke({"request": request, "messages": [], "rewrite_count": 0, "model_calls": [], "tool_calls": [], "errors": []})["final_result"]

    assert result.status == RunStatus.SUCCESS
    assert reporter.calls == 2


def test_graph_pass_path_produces_task_result(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_react_agent", lambda *_: FakeReactAgent())
    settings = _settings()
    runtime = LangGraphNodeRuntime(
        parser_agent=StubParser(),
        reporter_agent=StubReporter(),
        report_judge=SequenceJudge([True]),
        context_builder=ContextBuilderV2(settings=settings),
        mcp_manager=MCPClientManager(settings=settings, invoker=lambda *_: []),
        chat_model=object(),
        settings=settings,
    )
    result = build_review_graph(runtime).invoke(
        {"request": ReviewRunRequest(run_id="r2", user_id="u1", run_date=date(2026, 3, 5), trigger_type=TriggerType.MANUAL, raw_log_text="sample"), "messages": [], "rewrite_count": 0, "model_calls": [], "tool_calls": [], "errors": []}
    )["final_result"]

    assert result.status == RunStatus.SUCCESS
    assert result.report is not None
