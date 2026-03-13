from __future__ import annotations

from datetime import date
from pathlib import Path

from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, JudgementItem, LongTermJudgementRecord
from ai_trading_coach.modules.evaluation import DueEvaluationRunner, LongTermMemoryStore
from ai_trading_coach.prompts.prompt_store import PromptStore


def test_due_evaluation_updates_record_and_prompt_overlay(tmp_path: Path) -> None:
    mem_path = tmp_path / "memory.json"
    prompt_dir = tmp_path / "prompts"
    store = LongTermMemoryStore(str(mem_path))
    prompt_store = PromptStore(str(prompt_dir))

    record = LongTermJudgementRecord(
        judgement_id="j1",
        user_id="u1",
        run_id="r1",
        run_date=date(2026, 1, 1),
        due_date=date(2026, 1, 2),
        judgement=JudgementItem(category="market_view", target="SPX", thesis="SPX up"),
        initial_feedback=DailyJudgementFeedback(initial_feedback="likely_correct", evaluation_window="1 week"),
        cycle_evidence=[{"summary": "supporting market performance"}],
    )
    store.upsert_records([record])

    outputs = DueEvaluationRunner(store, prompt_store).run_due_evaluations(date(2026, 1, 3))
    assert outputs
    saved = store.load_all()[0]
    assert saved.status == "closed"
    assert saved.final_score is not None
    assert (prompt_dir / "learned_overlays.json").exists()


def test_clear_trace_keeps_long_term_memory(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace_logs"
    trace_dir.mkdir()
    (trace_dir / "a.json").write_text("{}", encoding="utf-8")
    memory_file = tmp_path / "long_term_memory.json"
    memory_file.write_text("[]", encoding="utf-8")

    for item in trace_dir.glob("*.json"):
        item.unlink()

    assert not list(trace_dir.glob("*.json"))
    assert memory_file.exists()
