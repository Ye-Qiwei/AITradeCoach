from __future__ import annotations

from datetime import date

from ai_trading_coach.app.factory import build_orchestrator_modules
from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest, TaskResult
from ai_trading_coach.modules.agent.react_research_agent import ReActResearchAgent
from ai_trading_coach.orchestrator.langchain_agent_orchestrator import LangChainAgentOrchestrator


class FakeGraph:
    def __init__(self) -> None:
        self.invoked = False

    def invoke(self, state):
        self.invoked = True
        request = state["request"]
        return {"final_result": TaskResult(run_id=request.run_id, status=RunStatus.SUCCESS, step_results=[])}


class DummyProvider:
    provider_name = "stub"
    model_name = "stub"

    def chat_json(self, schema_name, messages, timeout, prompt_version=None):
        del schema_name, messages, timeout, prompt_version
        return {}

    def chat_text(self, messages, prompt_version=None):
        del messages, prompt_version
        return ""


def test_langchain_orchestrator_run_invokes_compiled_graph() -> None:
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


def test_factory_modules_exclude_planner_and_executor_runtime_dependencies() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="k",
        mcp_servers_json='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        evidence_tool_map_json='{"price_path":"srv:price_tool"}',
        mcp_tool_allowlist_csv="srv:price_tool",
    )

    modules = build_orchestrator_modules(settings=settings, mcp_invoker=lambda *_: [])

    assert not hasattr(modules, "planner_agent")
    assert not hasattr(modules, "executor_engine")


def test_react_agent_actions_do_not_include_bootstrap_plan() -> None:
    content = ReActResearchAgent._decide.__code__.co_consts
    joined = " ".join(str(item) for item in content if isinstance(item, str))
    assert "bootstrap_investigation_plan" not in joined


def test_run_manual_cli_contract_kept() -> None:
    from ai_trading_coach.app import run_manual

    source = run_manual.run.__code__.co_names
    assert "build_pipeline_orchestrator" in source
    assert "run" in source
