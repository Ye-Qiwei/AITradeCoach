from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.orchestrator.langchain_agent_orchestrator import LangChainAgentOrchestrator


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
    request = ReviewRunRequest(
        run_id="run_graph",
        user_id="u1",
        run_date=date(2026, 3, 5),
        trigger_type=TriggerType.MANUAL,
        raw_log_text="sample",
    )
    result = orchestrator.run(request)
    assert graph.invoked is True
    assert result.run_id == "run_graph"
