from __future__ import annotations

from datetime import date

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class _J:
    judgement_id = "j1"
    category = "market_view"
    target_asset_or_topic = "SPY"
    thesis = "up"
    evidence_from_user_log = ["up"]
    implicitness = "explicit"
    proposed_evaluation_window = "1 week"
    atomic_judgements = []


class _Parse:
    def all_judgements(self):
        return [_J()]


class _Agent:
    def invoke(self, _payload):
        return {
            "messages": [
                type("M", (), {"content": '{"judgement_evidence":[{"judgement_id":"j1","evidence_item_ids":["e1"],"support_signal":"support","sufficiency_reason":"matched"}],"stop_reason":"done"}'})()
            ]
        }


class _Prompt:
    def load_active(self, _name):
        return type("Prompt", (), {"system_prompt": "sys"})()


class _E:
    item_id = "e1"


class _Packet:
    price_evidence = [_E()]
    news_evidence = []
    filing_evidence = []
    sentiment_evidence = []
    market_regime_evidence = []
    discussion_evidence = []
    analog_evidence = []
    macro_evidence = []
    source_registry = []


class _Runtime:
    evidence_items = []
    tool_traces = []
    react_steps = []


def test_react_research_outputs_research_output(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_agent", lambda **_: _Agent())
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_langchain_mcp_tools", lambda **_: [])
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_evidence_packet", lambda **_: _Packet())
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.MCPToolRuntime", _Runtime)

    runtime = LangGraphNodeRuntime(None, None, None, None, None, object(), Settings(llm_provider_name="openai", openai_api_key="x"), None, None, _Prompt())
    out = runtime.react_research({"request": ReviewRunRequest(run_id="r", user_id="u", run_date=date(2026, 1, 1), trigger_type=TriggerType.MANUAL, raw_log_text="x"), "parse_result": _Parse()})
    assert out["research_output"].judgement_evidence[0].support_signal in {"support", "oppose", "uncertain"}
