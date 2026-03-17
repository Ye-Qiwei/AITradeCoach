"""JSON-backed long-term judgement memory (separate from run trace)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from ai_trading_coach.domain.judgement_models import LongTermJudgementRecord


class LongTermMemoryStore:
    def __init__(self, path: str = "./data/long_term_memory.json") -> None:
        self.path = Path(path)

    def upsert_records(self, records: list[LongTermJudgementRecord]) -> None:
        data = {item.judgement_id: item for item in self.load_all()}
        for record in records:
            record.updated_at = datetime.now(timezone.utc)
            data[record.judgement_id] = record
        self._write(list(data.values()))

    def append_cycle_evidence(self, judgement_id: str, evidence: dict) -> None:
        data = {item.judgement_id: item for item in self.load_all()}
        if judgement_id not in data:
            return
        data[judgement_id].cycle_evidence.append(evidence)
        data[judgement_id].updated_at = datetime.now(timezone.utc)
        self._write(list(data.values()))

    def due_records(self, as_of: date) -> list[LongTermJudgementRecord]:
        return [r for r in self.load_all() if r.status != "closed" and r.due_date <= as_of]

    def load_all(self) -> list[LongTermJudgementRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [LongTermJudgementRecord.model_validate(item) for item in payload]

    def _write(self, records: list[LongTermJudgementRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
