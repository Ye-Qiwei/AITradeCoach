"""Run trace persistence utilities."""

from __future__ import annotations

from pathlib import Path

from ai_trading_coach.domain.models import RunTrace


def save_run_trace(trace: RunTrace, output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / f"{trace.run_id}.json"
    out_file.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return out_file
