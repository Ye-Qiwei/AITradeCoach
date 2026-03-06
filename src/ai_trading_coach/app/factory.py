"""Factory functions for the LangGraph-backed agent pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.langchain_chat_model import build_langchain_chat_model
from ai_trading_coach.llm.registry import build_required_llm_provider
from ai_trading_coach.modules.agent import CombinedParserAgent, ContextBuilderV2, ReportJudge, ReporterAgent
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator import LangChainAgentOrchestrator, OrchestratorModules, PipelineOrchestrator


def build_orchestrator_modules(
    settings: Settings,
    mcp_invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None,
) -> OrchestratorModules:
    settings.validate_llm_or_raise()
    provider = build_required_llm_provider(
        settings=settings,
        model_name=settings.selected_llm_model(),
        timeout_seconds=settings.llm_timeout_seconds,
    )
    return OrchestratorModules(
        parser_agent=CombinedParserAgent(provider=provider, timeout_seconds=settings.llm_timeout_seconds),
        reporter_agent=ReporterAgent(provider=provider, timeout_seconds=settings.llm_timeout_seconds),
        report_judge=ReportJudge(provider=provider, timeout_seconds=settings.llm_timeout_seconds),
        context_builder=ContextBuilderV2(settings=settings),
        mcp_manager=MCPClientManager(settings=settings, invoker=mcp_invoker),
    )


def build_pipeline_orchestrator(
    settings: Settings,
    mcp_invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None,
) -> LangChainAgentOrchestrator:
    modules = build_orchestrator_modules(settings=settings, mcp_invoker=mcp_invoker)
    chat_model = build_langchain_chat_model(settings=settings, timeout_seconds=settings.llm_timeout_seconds)
    return PipelineOrchestrator(modules=modules, settings=settings, chat_model=chat_model)


def build_cognition_engine(settings: Settings) -> CombinedParserAgent:
    """Backward-compatible alias for parser agent construction."""

    settings.validate_llm_or_raise()
    provider = build_required_llm_provider(settings=settings)
    return CombinedParserAgent(provider=provider, timeout_seconds=settings.llm_timeout_seconds)


def build_report_generator(settings: Settings) -> ReporterAgent:
    """Backward-compatible alias for reporter agent construction."""

    settings.validate_llm_or_raise()
    provider = build_required_llm_provider(settings=settings)
    return ReporterAgent(provider=provider, timeout_seconds=settings.llm_timeout_seconds)


__all__ = [
    "build_orchestrator_modules",
    "build_pipeline_orchestrator",
    "build_cognition_engine",
    "build_report_generator",
]
