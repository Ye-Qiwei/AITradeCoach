from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import EvaluationCategory
from ai_trading_coach.domain.models import ReplayCase
from ai_trading_coach.modules.cognition.service import HeuristicCognitionExtractionEngine
from ai_trading_coach.modules.context.service import BaselineShortTermContextBuilder
from ai_trading_coach.modules.evaluator.service import LayeredCognitionRealityEvaluator
from ai_trading_coach.modules.evidence.service import ClaimDrivenEvidencePlanner
from ai_trading_coach.modules.intake.service import MarkdownLogIntakeCanonicalizer
from ai_trading_coach.modules.ledger.service import BasicTradeLedgerPositionEngine
from ai_trading_coach.modules.mcp.service import DefaultMCPToolGateway
from ai_trading_coach.modules.memory.service import ChromaLongTermMemoryService
from ai_trading_coach.modules.promptops.service import ControlledPromptOpsSelfImprovementEngine
from ai_trading_coach.modules.report.service import StructuredReviewReportGenerator
from ai_trading_coach.modules.window.rule_based_selector import RuleBasedWindowSelector
from ai_trading_coach.orchestrator import OrchestratorModules, PipelineOrchestrator
from ai_trading_coach.replay import ReplayRunner


SAMPLE = """---
date: 2026-03-04
traded_tickers: [9660.HK]
---

## 状态
- emotion: 冷静
- stress: 2

## 扫描
- not_trade: 当前不追高

## 复盘
- lesson: 先看资金面
"""


def test_replay_runner_builds_predictions_and_scores(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(
        OrchestratorModules(
            log_intake=MarkdownLogIntakeCanonicalizer(),
            ledger_engine=BasicTradeLedgerPositionEngine(),
            cognition_engine=HeuristicCognitionExtractionEngine(),
            memory_service=ChromaLongTermMemoryService(persist_dir=str(tmp_path / ".chroma")),
            context_builder=BaselineShortTermContextBuilder(),
            evidence_planner=ClaimDrivenEvidencePlanner(),
            mcp_gateway=DefaultMCPToolGateway(),
            window_selector=RuleBasedWindowSelector(),
            evaluator=LayeredCognitionRealityEvaluator(),
            report_generator=StructuredReviewReportGenerator(),
            promptops_engine=ControlledPromptOpsSelfImprovementEngine(),
        )
    )
    runner = ReplayRunner(orchestrator=orchestrator)
    result = runner.run(
        [
            ReplayCase(
                case_id="c1",
                user_id="u1",
                run_date=date(2026, 3, 4),
                raw_log_text=SAMPLE,
                expected_categories=[EvaluationCategory.CORRECT],
            )
        ]
    )
    assert result.case_count == 1
    assert result.case_results
