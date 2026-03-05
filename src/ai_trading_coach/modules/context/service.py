"""Short-term context builder baseline implementation."""

from __future__ import annotations

from ai_trading_coach.domain.contracts import ContextBuildInput, ContextBuildOutput
from ai_trading_coach.domain.models import ExecutionContext, RelevantMemorySet


class BaselineShortTermContextBuilder:
    """Builds four-slot execution context with bounded memory payload."""

    def __init__(self, max_history_records: int = 12) -> None:
        self.max_history_records = max_history_records

    def build(self, data: ContextBuildInput) -> ContextBuildOutput:
        records = data.relevant_memories.records[: self.max_history_records]
        related_history = RelevantMemorySet(
            records=records,
            retrieval_notes=[
                *data.relevant_memories.retrieval_notes,
                f"history_capped={len(records)}/{len(data.relevant_memories.records)}",
            ],
        )
        context = ExecutionContext(
            today_input=data.normalized_log,
            related_history=related_history,
            market_evidence=None,
            task_goals=data.task_goals or ["daily_cognition_review", "fact_interpretation_evaluation_split"],
        )
        return ContextBuildOutput(execution_context=context)


# Backward-compatible alias
PlaceholderShortTermContextBuilder = BaselineShortTermContextBuilder
