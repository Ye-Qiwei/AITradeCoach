"""Replay evaluation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ai_trading_coach.config import get_settings
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

app = typer.Typer(add_completion=False)


@app.command()
def run(
    cases_file: str = typer.Option("examples/replay/replay_cases.sample.json", help="Replay cases JSON path"),
) -> None:
    settings = get_settings()
    payload = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter("Replay cases file must be a JSON list.")

    cases = [ReplayCase.model_validate(item) for item in payload]

    orchestrator = PipelineOrchestrator(
        modules=OrchestratorModules(
            log_intake=MarkdownLogIntakeCanonicalizer(),
            ledger_engine=BasicTradeLedgerPositionEngine(),
            cognition_engine=HeuristicCognitionExtractionEngine(),
            memory_service=ChromaLongTermMemoryService(),
            context_builder=BaselineShortTermContextBuilder(),
            evidence_planner=ClaimDrivenEvidencePlanner(),
            mcp_gateway=DefaultMCPToolGateway(),
            window_selector=RuleBasedWindowSelector(),
            evaluator=LayeredCognitionRealityEvaluator(),
            report_generator=StructuredReviewReportGenerator(),
            promptops_engine=ControlledPromptOpsSelfImprovementEngine(enable_llm=settings.atc_use_gemini),
        )
    )

    replay_result = ReplayRunner(orchestrator=orchestrator).run(cases)
    out_file = Path(settings.trace_output_dir) / f"{replay_result.replay_id}.json"
    out_file.write_text(replay_result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(replay_result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
