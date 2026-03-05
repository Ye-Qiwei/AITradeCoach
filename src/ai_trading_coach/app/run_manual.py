"""Manual trigger entrypoint for local development."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.enums import TriggerType
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
from ai_trading_coach.observability.tracing import save_run_trace
from ai_trading_coach.orchestrator import OrchestratorModules, PipelineOrchestrator

app = typer.Typer(add_completion=False)


@app.command()
def run(
    user_id: str = typer.Option("demo_user", help="User ID"),
    log_file: str = typer.Option("examples/logs/daily_log_sample.md", help="Path to markdown log"),
    run_date: str = typer.Option(date.today().isoformat(), help="Run date YYYY-MM-DD"),
    dry_run: bool = typer.Option(True, help="Disable memory write"),
) -> None:
    settings = get_settings()
    text = Path(log_file).read_text(encoding="utf-8")

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

    request = ReviewRunRequest(
        run_id=f"manual_{user_id}_{run_date}",
        user_id=user_id,
        run_date=date.fromisoformat(run_date),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=text,
        options={"dry_run": dry_run, "debug_mode": settings.atc_debug},
    )

    result = orchestrator.run(request)
    if result.report is not None:
        report_path = Path(settings.report_output_dir) / f"{result.run_id}.md"
        report_path.write_text(result.report.markdown_body, encoding="utf-8")
    if result.trace is not None:
        save_run_trace(result.trace, settings.trace_output_dir)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
