"""LangGraph topology for the single-path ReAct review workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState


def build_review_graph(runtime: LangGraphNodeRuntime):
    graph = StateGraph(OrchestratorGraphState)
    graph.add_node("parse_log", runtime.parse_log)
    graph.add_node("plan_research", runtime.plan_research_node)
    graph.add_node("execute_collection", runtime.execute_collection_node)
    graph.add_node("verify_information", runtime.verify_information_node)
    graph.add_node("build_report_context", runtime.build_report_context)
    graph.add_node("generate_report", runtime.generate_report)
    graph.add_node("judge_report", runtime.judge_report)
    graph.add_node("finalize_result", runtime.finalize_result)
    graph.add_node("finalize_failure", runtime.finalize_failure)

    graph.add_edge(START, "parse_log")
    graph.add_edge("parse_log", "plan_research")
    graph.add_edge("plan_research", "execute_collection")
    graph.add_edge("execute_collection", "verify_information")

    graph.add_conditional_edges(
        "verify_information",
        runtime.route_after_verify,
        {
            "continue_collection": "execute_collection",
            "research_done": "build_report_context",
        },
    )

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
