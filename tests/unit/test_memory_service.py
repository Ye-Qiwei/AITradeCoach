from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import MemoryRecallQuery, MemoryWriteInput
from ai_trading_coach.domain.enums import MemoryType
from ai_trading_coach.domain.models import MemoryRecord, MemoryWriteBatch
from ai_trading_coach.modules.memory.service import ChromaLongTermMemoryService


def test_chroma_memory_write_and_recall(tmp_path) -> None:
    svc = ChromaLongTermMemoryService(persist_dir=str(tmp_path / ".chroma"))

    record = MemoryRecord(
        memory_id="mem_1",
        user_id="u1",
        memory_type=MemoryType.COGNITIVE_CASE,
        source_date=date(2026, 3, 4),
        tickers=["9660.HK"],
        document_text="用户在高波动中执行了风险控制卖出",
        keywords=["风险控制", "卖出"],
    )

    write_out = svc.write(MemoryWriteInput(user_id="u1", batch=MemoryWriteBatch(records=[record])))
    assert write_out.written_memory_ids == ["mem_1"]

    recall_out = svc.recall(MemoryRecallQuery(user_id="u1", top_k=5))
    assert recall_out.relevant_memories.records
    assert recall_out.relevant_memories.records[0].memory_id == "mem_1"
