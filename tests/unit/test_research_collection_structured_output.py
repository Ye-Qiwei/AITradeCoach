from __future__ import annotations

from datetime import date

import pytest
from langchain.agents.structured_output import ToolStrategy

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.llm_output_contracts import ResearchAgentFinalContract
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime, _parse_final_contract


class _J:
    judgement_id = "j1"
    category = "market_view"
    target_asset_or_topic = "SPY"
    thesis = "SPY rebounds"
    evidence_from_user_log = ["user thought oversold"]
    implicitness = "explicit"
    proposed_evaluation_window = "1 week"
    atomic_judgements = []


class _Parse:
    def all_judgements(self):
        return [_J()]


class _PromptManager:
    def load_active(self, _name):
        return type("Prompt", (), {"system_prompt": "sys"})()


class _E:
    item_id = "e1"
    summary = "evidence"


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
    evidence_items = [_E()]
    tool_traces = []
    react_steps = []


def _valid_contract_payload() -> dict:
    return {
        "judgement_evidence": [
            {
                "judgement_id": "j1",
                "evidence_item_ids": ["e1"],
                "support_signal": "support",
                "sufficiency_reason": "evidence aligns with thesis",
            }
        ],
        "stop_reason": "sufficient evidence collected",
    }


def test_parse_final_contract_prefers_structured_response() -> None:
    contract = _parse_final_contract(
        {
            "structured_response": _valid_contract_payload(),
            "messages": [type("M", (), {"content": "not-json"})()],
        }
    )
    assert contract.judgement_evidence[0].judgement_id == "j1"


def test_parse_final_contract_fallback_from_message_json() -> None:
    contract = _parse_final_contract(
        {
            "messages": [
                type("M", (), {"content": '{"judgement_evidence":[{"judgement_id":"j1","evidence_item_ids":["e1"],"support_signal":"oppose","sufficiency_reason":"counter evidence"}],"stop_reason":"done"}'})()
            ]
        }
    )
    assert contract.judgement_evidence[0].support_signal == "oppose"


def test_parse_final_contract_error_is_diagnostic_for_schema_drift() -> None:
    with pytest.raises(ValueError) as exc_info:
        _parse_final_contract({"messages": [type("M", (), {"content": '{"findings": [], "summary": {}}'})()]})

    message = str(exc_info.value)
    assert "structured_response_present=False" in message
    assert "expected_top_level_keys" in message
    assert "actual_top_level_keys" in message
    assert "findings" in message
    assert "summary" in message


def test_execute_collection_node_passes_structured_response_format(monkeypatch) -> None:
    captured: dict = {}

    class _Agent:
        def invoke(self, _payload):
            return {"structured_response": _valid_contract_payload(), "messages": []}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return _Agent()

    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_agent", _fake_create_agent)
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_langchain_mcp_tools", lambda **_: [])
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_evidence_packet", lambda **_: _Packet())
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.MCPToolRuntime", _Runtime)

    runtime = LangGraphNodeRuntime(
        parser_agent=None,
        reporter_agent=None,
        report_judge=None,
        context_builder=None,
        mcp_manager=None,
        chat_model=object(),
        settings=Settings(llm_provider_name="openai", openai_api_key="x"),
        long_term_store=None,
        llm_gateway=None,
        prompt_manager=_PromptManager(),
    )

    state = {
        "request": ReviewRunRequest(
            run_id="r1",
            user_id="u1",
            run_date=date(2026, 1, 1),
            trigger_type=TriggerType.MANUAL,
            raw_log_text="x",
        ),
        "parse_result": _Parse(),
        "analysis_framework": "f",
        "analysis_directions": [],
        "info_requirements": [],
        "collected_info": [],
        "verify_suggestions": [],
    }

    out = runtime.execute_collection_node(state)
    assert "research_output" in out
    assert isinstance(captured["response_format"], ToolStrategy)
    assert captured["response_format"].schema is ResearchAgentFinalContract


def test_research_agent_v2_prompt_contract_alignment() -> None:
    prompt_text = open("config/prompts/research_agent.v2.md", encoding="utf-8").read()
    assert "judgement_evidence" in prompt_text
    assert "stop_reason" in prompt_text
    assert "support_signal" in prompt_text
    assert "sufficiency_reason" in prompt_text
    assert "Do not output" in prompt_text
