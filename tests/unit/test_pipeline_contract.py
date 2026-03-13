from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, JudgementItem, ParserOutput, ResearchOutput, ResearchedJudgementItem
from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.orchestrator.langchain_agent_orchestrator import LangChainAgentOrchestrator
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class FakeGraph:
    def __init__(self) -> None:
        self.invoked = False

    def invoke(self, state):
        self.invoked = True
        request = state["request"]
        return {"final_result": TaskResult(run_id=request.run_id, status=RunStatus.SUCCESS, step_results=[])}


def test_orchestrator_run_invokes_graph() -> None:
    graph = FakeGraph()
    orchestrator = LangChainAgentOrchestrator(compiled_graph=graph)
    request = ReviewRunRequest(run_id="run_graph", user_id="u1", run_date=date(2026, 3, 5), trigger_type=TriggerType.MANUAL, raw_log_text="sample")
    result = orchestrator.run(request)
    assert graph.invoked is True
    assert result.run_id == "run_graph"


def test_judge_report_handles_none_trace() -> None:
    class DummyJudge:
        def evaluate(self, **kwargs):
            from ai_trading_coach.domain.agent_models import JudgeVerdict

            return JudgeVerdict(passed=True), None

    class DummyContext:
        def for_judge(self, **kwargs):
            return {"judgement_feedback": kwargs["judgement_feedback"], "judgement_bundles": kwargs["report_context"].get("judgement_bundles", [])}

    runtime = LangGraphNodeRuntime(None, None, DummyJudge(), DummyContext(), None, None, type("S", (), {"agent_max_rewrite_rounds": 0})(), None, None)
    state = {
        "report_draft": "## A\ninitial_feedback: likely_correct\nevaluation_window: 1 week",
        "judgement_feedback": [DailyJudgementFeedback(initial_feedback="likely_correct", evaluation_window="1 week")],
        "parse_result": ParserOutput(user_id="u", run_date=date(2026, 1, 1), judgements=[JudgementItem(category="asset_view", target="AAPL", thesis="up")]),
        "research_output": ResearchOutput(judgements=[ResearchedJudgementItem(category="asset_view", target="AAPL", thesis="up")]),
        "evidence_packet": type("EP", (), {"source_registry": []})(),
        "report_context": {"judgement_bundles": []},
        "model_calls": [],
        "rewrite_count": 0,
    }
    out = runtime.judge_report(state)
    assert out["judge_verdict"].passed is True
