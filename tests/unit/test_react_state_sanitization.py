from __future__ import annotations

from datetime import date

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class _FakeJudgement:
    judgement_id = "j1"
    thesis = "thesis"


class _FakeParseResult:
    def all_judgements(self):
        return [_FakeJudgement()]


class _FakeMessageText:
    content = "agent says hi"


class _FakeMessageObject:
    content = {"detail": 1}


class _FakeMessageBare:
    def __str__(self) -> str:
        return "bare-message"


class _FakeAgent:
    def invoke(self, _payload):
        return {"messages": [_FakeMessageText(), _FakeMessageObject(), _FakeMessageBare()]}


class _FakeToolTrace:
    def model_dump(self, mode="json"):
        return {"tool": "t"}


class _FakeStep:
    def model_dump(self, mode="json"):
        return {"step": "s"}


class _FakeRuntime:
    evidence_items = []
    tool_traces = [_FakeToolTrace()]
    react_steps = [_FakeStep()]


def test_react_research_sanitizes_agent_messages(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_react_agent", lambda *_: _FakeAgent())
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_langchain_mcp_tools", lambda **_: [])
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.MCPToolRuntime", _FakeRuntime)
    monkeypatch.setattr(
        "ai_trading_coach.orchestrator.langgraph_nodes.build_evidence_packet",
        lambda **_: {"packet": "ok"},
    )

    runtime = LangGraphNodeRuntime(
        parser_agent=None,
        reporter_agent=None,
        report_judge=None,
        context_builder=None,
        mcp_manager=None,
        chat_model=object(),
        settings=Settings(llm_provider_name="openai", openai_api_key="test"),
        long_term_store=None,
        llm_gateway=None,
        prompt_manager=None,
    )

    result = runtime.react_research(
        {
            "request": ReviewRunRequest(
                run_id="r1",
                user_id="u1",
                run_date=date(2026, 3, 5),
                trigger_type=TriggerType.MANUAL,
                raw_log_text="x",
            ),
            "parse_result": _FakeParseResult(),
        }
    )

    assert result["agent_messages"] == ["agent says hi", "{'detail': 1}", "bare-message"]
    assert "messages" not in result
