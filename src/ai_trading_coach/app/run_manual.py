"""Manual trigger entrypoint for local development."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from ai_trading_coach.app.factory import build_pipeline_orchestrator
from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.modules.agent.tools import get_tool_availability
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.observability.tracing import save_run_trace

app = typer.Typer(add_completion=False)

@app.callback()
def main() -> None:
    """Manual run commands."""


def _ensure_minimum_config_or_raise() -> None:
    settings = get_settings()
    settings.validate_llm_or_raise()
    available = [x for x in get_tool_availability(settings, MCPClientManager(settings=settings)) if x.available]
    if not available:
        typer.echo("configuration_error: at least one research tool must be available", err=True)
        raise typer.Exit(code=2)


@app.command()
def run(
    user_id: str = typer.Option("demo_user", help="User ID"),
    log_file: str = typer.Option("examples/logs/daily_log_sample.md", help="Path to markdown log"),
    run_date: str = typer.Option(date.today().isoformat(), help="Run date YYYY-MM-DD"),
    dry_run: bool = typer.Option(False, help="Disable memory + report/trace writes"),
) -> None:
    settings = get_settings()
    _ensure_minimum_config_or_raise()
    text = Path(log_file).read_text(encoding="utf-8")
    orchestrator = build_pipeline_orchestrator(settings)
    request = ReviewRunRequest(
        run_id=f"manual_{user_id}_{run_date}",
        user_id=user_id,
        run_date=date.fromisoformat(run_date),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=text,
        options={"dry_run": dry_run, "debug_mode": False},
    )
    result = orchestrator.run(request)
    if not dry_run and result.report is not None:
        report_path = Path(settings.report_output_dir) / f"{result.run_id}.md"
        report_path.write_text(result.report.markdown_body, encoding="utf-8")
    if not dry_run and result.trace is not None:
        save_run_trace(result.trace, settings.trace_output_dir)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
