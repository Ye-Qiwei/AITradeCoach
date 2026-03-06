"""Evaluate due judgements and write prompt-learning overlays."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.modules.evaluation.long_term_store import LongTermMemoryStore
from ai_trading_coach.prompts.prompt_store import PromptStore


class DueEvaluationRunner:
    def __init__(self, memory_store: LongTermMemoryStore, prompt_store: PromptStore) -> None:
        self.memory_store = memory_store
        self.prompt_store = prompt_store

    def run_due_evaluations(self, as_of: date) -> list[dict]:
        due = self.memory_store.due_records(as_of)
        outputs: list[dict] = []
        for record in due:
            objective = record.cycle_evidence[-1]["summary"] if record.cycle_evidence else "insufficient objective data"
            final_score = 0.7 if "support" in objective.lower() else 0.4
            record.final_score = final_score
            record.final_outcome = objective
            record.final_commentary = (
                "Initial feedback aligned with later evidence. Keep prompt heuristics."
                if final_score >= 0.6
                else "Initial feedback deviated from later evidence. Tighten evidence grounding instructions."
            )
            record.status = "closed"
            if final_score >= 0.6:
                self.prompt_store.append_overlay(
                    "report_generation",
                    "When evidence is coherent across sources, state confidence explicitly.",
                    f"Judgement {record.judgement_id} final evaluation success",
                )
            else:
                self.prompt_store.append_overlay(
                    "react_research",
                    "Require one contradictory source check before final support/oppose signal.",
                    f"Judgement {record.judgement_id} final evaluation mismatch",
                )
            outputs.append({
                "judgement_id": record.judgement_id,
                "final_score": record.final_score,
                "commentary": record.final_commentary,
            })
        self.memory_store.upsert_records(due)
        return outputs
