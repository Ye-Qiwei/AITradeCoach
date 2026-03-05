from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.modules.cognition.service import PlaceholderCognitionExtractionEngine
from ai_trading_coach.modules.context.service import PlaceholderShortTermContextBuilder
from ai_trading_coach.modules.evaluator.service import PlaceholderCognitionRealityEvaluator
from ai_trading_coach.modules.evidence.service import PlaceholderEvidencePlanner
from ai_trading_coach.modules.intake.service import PlaceholderLogIntakeCanonicalizer
from ai_trading_coach.modules.ledger.service import PlaceholderTradeLedgerPositionEngine
from ai_trading_coach.modules.mcp.service import PlaceholderMCPToolGateway
from ai_trading_coach.modules.memory.service import PlaceholderLongTermMemoryService
from ai_trading_coach.modules.promptops.service import PlaceholderPromptOpsSelfImprovementEngine
from ai_trading_coach.modules.report.service import PlaceholderReviewReportGenerator
from ai_trading_coach.modules.window.rule_based_selector import RuleBasedWindowSelector
from ai_trading_coach.orchestrator import OrchestratorModules, PipelineOrchestrator


def test_pipeline_returns_partial_with_placeholders() -> None:
    orchestrator = PipelineOrchestrator(
        OrchestratorModules(
            log_intake=PlaceholderLogIntakeCanonicalizer(),
            ledger_engine=PlaceholderTradeLedgerPositionEngine(),
            cognition_engine=PlaceholderCognitionExtractionEngine(),
            memory_service=PlaceholderLongTermMemoryService(),
            context_builder=PlaceholderShortTermContextBuilder(),
            evidence_planner=PlaceholderEvidencePlanner(),
            mcp_gateway=PlaceholderMCPToolGateway(),
            window_selector=RuleBasedWindowSelector(),
            evaluator=PlaceholderCognitionRealityEvaluator(),
            report_generator=PlaceholderReviewReportGenerator(),
            promptops_engine=PlaceholderPromptOpsSelfImprovementEngine(),
        )
    )

    request = ReviewRunRequest(
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 3, 4),
        trigger_type=TriggerType.MANUAL,
        raw_log_text="# demo",
    )

    result = orchestrator.run(request)
    assert result.status == RunStatus.SUCCESS
    assert result.errors == []
    assert result.improvement_proposals
