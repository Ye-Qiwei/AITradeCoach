"""Delete only trace files, preserving long-term memory."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_trading_coach.config import get_settings

app = typer.Typer(add_completion=False)


@app.command()
def run() -> None:
    settings = get_settings()
    trace_dir = Path(settings.trace_output_dir)
    deleted = 0
    if trace_dir.exists():
        for item in trace_dir.glob("*.json"):
            item.unlink()
            deleted += 1
    typer.echo({"deleted_trace_files": deleted, "trace_dir": str(trace_dir)})


if __name__ == "__main__":
    app()
