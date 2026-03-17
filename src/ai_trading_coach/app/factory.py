"""
Note: daily agent payloads use id-less list alignment between nodes.
Factory functions for the LangGraph-backed agent pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator import LangChainAgentOrchestrator, OrchestratorModules, PipelineOrchestrator


def build_orchestrator_modules(
    settings: Settings,
    mcp_invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None,
) -> OrchestratorModules:
    settings.validate_llm_or_raise()
    gateway = LangChainLLMGateway(settings)
    prompt_manager = PromptManager(settings.prompt_root)
    return OrchestratorModules(
        parser_agent=CombinedParserAgent(gateway=gateway, prompt_manager=prompt_manager),
        reporter_agent=ReporterAgent(gateway=gateway, prompt_manager=prompt_manager),
        report_judge=ReportJudge(gateway=gateway, prompt_manager=prompt_manager),
        context_builder=ContextBuilderV2(),
        mcp_manager=MCPClientManager(settings=settings, invoker=mcp_invoker),
        long_term_store=LongTermMemoryStore(),
        llm_gateway=gateway,
        prompt_manager=prompt_manager,
    )


def build_pipeline_orchestrator(
    settings: Settings,
    mcp_invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None,
) -> LangChainAgentOrchestrator:
    modules = build_orchestrator_modules(settings=settings, mcp_invoker=mcp_invoker)
    return PipelineOrchestrator(modules=modules, settings=settings, chat_model=modules.llm_gateway.model)


def build_cognition_engine(settings: Settings) -> CombinedParserAgent:
    settings.validate_llm_or_raise()
    gateway = LangChainLLMGateway(settings)
    prompt_manager = PromptManager(settings.prompt_root)
    return CombinedParserAgent(gateway=gateway, prompt_manager=prompt_manager)


def build_report_generator(settings: Settings) -> ReporterAgent:
    settings.validate_llm_or_raise()
    gateway = LangChainLLMGateway(settings)
    prompt_manager = PromptManager(settings.prompt_root)
    return ReporterAgent(gateway=gateway, prompt_manager=prompt_manager)


__all__ = [
    "build_orchestrator_modules",
    "build_pipeline_orchestrator",
    "build_cognition_engine",
    "build_report_generator",
]
