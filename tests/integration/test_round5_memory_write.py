from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
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


SAMPLE = """---
date: 2026-03-04
traded_tickers: [9660.HK]
---

## 状态
- emotion: 焦虑
- stress: 4

## 交易记录
- 9660.HK SELL 600股 7.39HKD | reason=事件后反应弱，先降风险

## 扫描
- fomo: 4901.T 抗跌
- not_trade: 当前不追高

## 复盘
- lesson: 先看资金面
"""


def test_round5_writes_improvement_note_memory(tmp_path) -> None:
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
    request = ReviewRunRequest(
        run_id="round5_memory_write",
        user_id="u1",
        run_date=date(2026, 3, 4),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=SAMPLE,
        options={"dry_run": False},
    )

    result = orchestrator.run(request)
    assert result.status == RunStatus.SUCCESS
    collections = {item.collection for item in result.memory_write_results}
    assert "mixed" in collections
    assert "agent_improvement_notes" in collections

