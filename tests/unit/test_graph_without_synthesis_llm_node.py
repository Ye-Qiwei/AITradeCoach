from __future__ import annotations

from ai_trading_coach.orchestrator import langgraph_graph


def test_graph_builder_no_synthesis_node_reference() -> None:
    source = open(langgraph_graph.__file__, encoding="utf-8").read()
    assert "synthesize_research_output" not in source
