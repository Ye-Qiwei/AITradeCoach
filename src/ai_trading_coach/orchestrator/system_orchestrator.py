"""Compatibility definitions for graph-backed orchestration modules."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langchain_agent_orchestrator import LangChainAgentOrchestrator
from ai_trading_coach.orchestrator.langgraph_graph import build_review_graph
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


@dataclass
class OrchestratorModules:
    parser_agent: CombinedParserAgent
    reporter_agent: ReporterAgent
    report_judge: ReportJudge
    context_builder: ContextBuilderV2
    mcp_manager: MCPClientManager
    long_term_store: LongTermMemoryStore
    llm_gateway: LangChainLLMGateway
    prompt_manager: PromptManager


class PipelineOrchestrator(LangChainAgentOrchestrator):
    """Backward-compatible name backed by compiled LangGraph runtime."""

    def __init__(self, *, modules: OrchestratorModules, settings: Settings, chat_model: object) -> None:
        runtime = LangGraphNodeRuntime(
            parser_agent=modules.parser_agent,
            reporter_agent=modules.reporter_agent,
            report_judge=modules.report_judge,
            context_builder=modules.context_builder,
            mcp_manager=modules.mcp_manager,
            chat_model=chat_model,
            settings=settings,
            long_term_store=modules.long_term_store,
            llm_gateway=modules.llm_gateway,
            prompt_manager=modules.prompt_manager,
        )
        super().__init__(compiled_graph=build_review_graph(runtime), chat_model=chat_model)


__all__ = ["OrchestratorModules", "PipelineOrchestrator"]
