"""CLI entrypoint for due long-term judgement evaluations."""

from __future__ import annotations

from datetime import date

import typer

from ai_trading_coach.config import get_settings
from ai_trading_coach.modules.evaluation import DueEvaluationRunner, LongTermMemoryStore
from ai_trading_coach.prompts.prompt_store import PromptStore

app = typer.Typer(add_completion=False)


@app.command()
def run(as_of: str = typer.Option(date.today().isoformat(), help="Evaluation date YYYY-MM-DD")) -> None:
    settings = get_settings()
    runner = DueEvaluationRunner(
        memory_store=LongTermMemoryStore(),
        prompt_store=PromptStore(settings.prompt_root),
    )
    result = runner.run_due_evaluations(as_of=date.fromisoformat(as_of))
    typer.echo({"evaluated": len(result), "results": result})


if __name__ == "__main__":
    app()
