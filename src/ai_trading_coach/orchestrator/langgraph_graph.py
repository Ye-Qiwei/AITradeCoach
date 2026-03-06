"""LangGraph topology for the single-path ReAct review workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def build_review_graph(runtime: LangGraphNodeRuntime):
    graph = StateGraph(OrchestratorGraphState)
    graph.add_node("parse_log", runtime.parse_log)
    graph.add_node("react_research", runtime.react_research)
    graph.add_node("build_report_context", runtime.build_report_context)
    graph.add_node("generate_report", runtime.generate_report)
    graph.add_node("judge_report", runtime.judge_report)
    graph.add_node("finalize_result", runtime.finalize_result)
    graph.add_node("finalize_failure", runtime.finalize_failure)

    graph.add_edge(START, "parse_log")
    graph.add_edge("parse_log", "react_research")
    graph.add_edge("react_research", "build_report_context")
    graph.add_edge("build_report_context", "generate_report")
    graph.add_edge("generate_report", "judge_report")

    graph.add_conditional_edges(
        "judge_report",
        runtime.route_after_judge,
        {
            "pass": "finalize_result",
            "rewrite": "generate_report",
            "fail": "finalize_failure",
        },
    )
    graph.add_edge("finalize_result", END)
    graph.add_edge("finalize_failure", END)
    return graph.compile()


__all__ = ["build_review_graph"]
