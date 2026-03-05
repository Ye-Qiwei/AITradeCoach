"""Module factory for runtime engine selection."""

from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.registry import build_llm_provider
from ai_trading_coach.modules.cognition.llm_engine import LLMCognitionExtractionEngine
from ai_trading_coach.modules.cognition.service import HeuristicCognitionExtractionEngine
from ai_trading_coach.modules.context.service import BaselineShortTermContextBuilder
from ai_trading_coach.modules.evaluator.service import LayeredCognitionRealityEvaluator
from ai_trading_coach.modules.evidence.service import ClaimDrivenEvidencePlanner
from ai_trading_coach.modules.intake.service import MarkdownLogIntakeCanonicalizer
from ai_trading_coach.modules.ledger.service import BasicTradeLedgerPositionEngine
from ai_trading_coach.modules.mcp.service import DefaultMCPToolGateway
from ai_trading_coach.modules.memory.service import ChromaLongTermMemoryService
from ai_trading_coach.modules.promptops.service import ControlledPromptOpsSelfImprovementEngine
from ai_trading_coach.modules.report.llm_report import LLMReviewReportGenerator
from ai_trading_coach.modules.report.service import StructuredReviewReportGenerator
from ai_trading_coach.modules.window.rule_based_selector import RuleBasedWindowSelector
from ai_trading_coach.orchestrator import OrchestratorModules


def build_cognition_engine(settings: Settings) -> HeuristicCognitionExtractionEngine | LLMCognitionExtractionEngine:
    heuristic = HeuristicCognitionExtractionEngine()
    if not settings.atc_enable_llm_cognition:
        return heuristic

    provider = build_llm_provider(
        settings=settings,
        model_name=settings.atc_llm_model or None,
        timeout_seconds=settings.atc_llm_timeout_seconds,
    )
    return LLMCognitionExtractionEngine(
        provider=provider,
        timeout_seconds=settings.atc_llm_timeout_seconds,
        fallback_engine=heuristic,
    )


def build_report_generator(settings: Settings) -> StructuredReviewReportGenerator | LLMReviewReportGenerator:
    heuristic = StructuredReviewReportGenerator()
    if not settings.atc_enable_llm_report:
        return heuristic

    provider = build_llm_provider(
        settings=settings,
        model_name=settings.atc_llm_model or None,
        timeout_seconds=settings.atc_llm_timeout_seconds,
    )
    return LLMReviewReportGenerator(
        provider=provider,
        timeout_seconds=settings.atc_llm_timeout_seconds,
        fallback_generator=heuristic,
    )


def build_orchestrator_modules(settings: Settings, memory_persist_dir: str | None = None) -> OrchestratorModules:
    memory_service = (
        ChromaLongTermMemoryService(persist_dir=memory_persist_dir)
        if memory_persist_dir is not None
        else ChromaLongTermMemoryService()
    )

    return OrchestratorModules(
        log_intake=MarkdownLogIntakeCanonicalizer(),
        ledger_engine=BasicTradeLedgerPositionEngine(),
        cognition_engine=build_cognition_engine(settings),
        memory_service=memory_service,
        context_builder=BaselineShortTermContextBuilder(),
        evidence_planner=ClaimDrivenEvidencePlanner(),
        mcp_gateway=DefaultMCPToolGateway(),
        window_selector=RuleBasedWindowSelector(),
        evaluator=LayeredCognitionRealityEvaluator(),
        report_generator=build_report_generator(settings),
        promptops_engine=ControlledPromptOpsSelfImprovementEngine(enable_llm=settings.atc_use_gemini),
    )


__all__ = ["build_cognition_engine", "build_report_generator", "build_orchestrator_modules"]
