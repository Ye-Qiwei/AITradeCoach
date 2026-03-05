"""Replay evaluation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ai_trading_coach.app.factory import build_orchestrator_modules
from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.models import ReplayCase
from ai_trading_coach.orchestrator import PipelineOrchestrator
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

    orchestrator = PipelineOrchestrator(modules=build_orchestrator_modules(settings))

    replay_result = ReplayRunner(orchestrator=orchestrator).run(cases)
    out_file = Path(settings.trace_output_dir) / f"{replay_result.replay_id}.json"
    out_file.write_text(replay_result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(replay_result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
