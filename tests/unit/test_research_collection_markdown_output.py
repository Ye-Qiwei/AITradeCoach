from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.judgement_models import JudgementItem, ParserOutput
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class _DummyPromptManager:
    def load_active(self, _name: str):
        return type("Prompt", (), {"system_prompt": "x", "prompt_name": "research_agent"})()


class _DummyAgent:
    def __init__(self) -> None:
        self.last_content = ""

    def invoke(self, payload):
        self.last_content = payload["messages"][0].content
        return {
            "messages": [
                type("M", (), {"content": "# Judgement Evidence\n\n## Judgement 1\n- support_signal: support\n- evidence_quality: sufficient\n- cited_sources:\n  - src_1\n- rationale: grounded"})()
            ]
        }


def test_execute_collection_node_sends_markdown_brief(monkeypatch) -> None:
    dummy_agent = _DummyAgent()
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.build_runtime_research_tools", lambda **_: [])
    monkeypatch.setattr("ai_trading_coach.orchestrator.langgraph_nodes.create_agent", lambda **_: dummy_agent)

    runtime = LangGraphNodeRuntime(
        parser_agent=None,
        reporter_agent=None,
        report_judge=None,
        context_builder=None,
        mcp_manager=None,
        chat_model=None,
        settings=type("S", (), {"react_require_min_sources": 0})(),
        long_term_store=None,
        prompt_manager=_DummyPromptManager(),
    )
    state = {
        "request": type("Req", (), {"run_id": "r1", "user_id": "u1", "run_date": date(2026, 1, 1)})(),
        "parse_result": ParserOutput(
            user_id="u1",
            run_date=date(2026, 1, 1),
            judgements=[JudgementItem(category="asset_view", target="AAPL", thesis="Up", evaluation_window="1 week")],
        ),
    }

    out = runtime.execute_collection_node(state)
    assert "# Research Task" in dummy_agent.last_content
    assert '"task"' not in dummy_agent.last_content
    assert out["research_output"].judgements[0].target == "AAPL"
