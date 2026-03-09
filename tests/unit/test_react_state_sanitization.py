from __future__ import annotations

from datetime import date

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class _FakeJudgement:
    judgement_id = "j1"
    category = "market_view"
    target_asset_or_topic = "SPY"
    thesis = "thesis"
    evidence_from_user_log = ["note"]
    implicitness = "explicit"
    proposed_evaluation_window = "1 week"


class _FakeParseResult:
    def all_judgements(self):
        return [_FakeJudgement()]


class _FakeMessageText:
    content = '{"judgement_evidence":[{"judgement_id":"j1","evidence_item_ids":[],"support_signal":"uncertain","sufficiency_reason":"none"}],"stop_reason":"done"}'


class _FakeMessageObject:
    content = {"detail": 1}


class _FakeMessageBare:
    def __str__(self) -> str:
        return "bare-message"


class _FakeAgent:
    def invoke(self, _payload):
        return {"messages": [_FakeMessageObject(), _FakeMessageBare(), _FakeMessageText()]}


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


class _FakePromptManager:
    def load_active(self, _name):
        return type("Prompt", (), {"system_prompt": "sys"})()


def test_react_research_sanitizes_agent_messages(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_agent", lambda **_: _FakeAgent())
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_langchain_mcp_tools", lambda **_: [])
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.MCPToolRuntime", _FakeRuntime)
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_evidence_packet", lambda **_: type("Packet", (), {
        "price_evidence": [], "news_evidence": [], "filing_evidence": [], "sentiment_evidence": [],
        "market_regime_evidence": [], "discussion_evidence": [], "analog_evidence": [], "macro_evidence": [],
        "source_registry": []
    })())

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
        prompt_manager=_FakePromptManager(),
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

    assert result["agent_messages"][-1].startswith('{"judgement_evidence"')
    assert "messages" not in result
    assert result["research_output"].judgement_evidence[0].support_signal == "uncertain"
