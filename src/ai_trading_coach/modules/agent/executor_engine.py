"""Executor engine that runs plan subtasks in parallel via MCP tool calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from ai_trading_coach.domain.agent_models import Plan, PlanSubTask, SubTaskExecutionTrace
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket, ToolCallTrace
from ai_trading_coach.modules.mcp.adapters import normalize_tool_output
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, tool_payload_hash


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ExecutorResult:
    evidence_packet: EvidencePacket
    tool_traces: list[ToolCallTrace]
    subtask_traces: list[SubTaskExecutionTrace]


class ExecutorEngine:
    """Execute independent subtasks concurrently and normalize results."""

    def __init__(
        self,
        *,
        mcp_manager: MCPClientManager,
        max_items_per_subtask: int = 8,
    ) -> None:
        self.mcp_manager = mcp_manager
        self.max_items_per_subtask = max_items_per_subtask

    def execute(self, *, plan: Plan, user_id: str) -> ExecutorResult:
        return asyncio.run(self._execute_async(plan=plan, user_id=user_id))

    async def _execute_async(self, *, plan: Plan, user_id: str) -> ExecutorResult:
        packet = EvidencePacket(packet_id=f"packet_{plan.plan_id}", user_id=user_id)
        if not plan.subtasks:
            return ExecutorResult(evidence_packet=packet, tool_traces=[], subtask_traces=[])

        tasks = [self._execute_subtask(subtask) for subtask in plan.subtasks]
        results = await asyncio.gather(*tasks)

        tool_traces: list[ToolCallTrace] = []
        subtask_traces: list[SubTaskExecutionTrace] = []
        successful = 0
        missing: list[str] = []
        for subtask, items, tool_trace, subtask_trace in results:
            tool_traces.append(tool_trace)
            subtask_traces.append(subtask_trace)
            if subtask_trace.success:
                successful += 1
                self._append_items(packet, items)
            else:
                missing.append(f"{subtask.subtask_id}:{tool_trace.error_message or 'unknown_error'}")

        packet.missing_requirements = missing
        packet.completeness_score = round(successful / len(plan.subtasks), 4)
        packet.extensions["tool_call_traces"] = [trace.model_dump(mode="json") for trace in tool_traces]
        return ExecutorResult(
            evidence_packet=packet,
            tool_traces=tool_traces,
            subtask_traces=subtask_traces,
        )

    async def _execute_subtask(
        self,
        subtask: PlanSubTask,
    ) -> tuple[PlanSubTask, list[EvidenceItem], ToolCallTrace, SubTaskExecutionTrace]:
        tool_ref = self.mcp_manager.resolve_tool(subtask.evidence_type)
        request_payload = {
            "objective": subtask.objective,
            "query": subtask.query,
            "tickers": subtask.tickers,
            "time_window": subtask.time_window,
            "success_criteria": subtask.success_criteria,
        }
        payload_digest = tool_payload_hash(request_payload)

        started = utc_now()
        t0 = perf_counter()
        success = True
        error_message: str | None = None
        items: list[EvidenceItem] = []
        raw_result: Any = None
        try:
            raw_result = await self.mcp_manager.call_tool(
                server_id=tool_ref.server_id,
                tool_name=tool_ref.tool_name,
                arguments=request_payload,
            )
            items = normalize_tool_output(
                server_id=tool_ref.server_id,
                tool_name=tool_ref.tool_name,
                subtask=subtask,
                raw_result=raw_result,
            )[: self.max_items_per_subtask]
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)

        latency_ms = int((perf_counter() - t0) * 1000)
        ended = utc_now()

        response_summary = f"items={len(items)}"
        if not success:
            response_summary = "items=0"

        tool_trace = ToolCallTrace(
            call_id=f"tool_{subtask.subtask_id}_{int(started.timestamp() * 1000)}",
            tool_name=tool_ref.tool_name,
            server_id=tool_ref.server_id,
            request_summary=f"evidence={subtask.evidence_type.value} tickers={subtask.tickers}",
            response_summary=response_summary,
            payload_hash=payload_digest,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
        )

        subtask_trace = SubTaskExecutionTrace(
            subtask_id=subtask.subtask_id,
            tool_ref=tool_ref.key,
            started_at=started,
            ended_at=ended,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            evidence_item_count=len(items),
        )
        return subtask, items, tool_trace, subtask_trace

    def _append_items(self, packet: EvidencePacket, items: list[EvidenceItem]) -> None:
        for item in items:
            packet.source_registry.extend(item.sources)
            self._append_by_type(packet, item)

    def _append_by_type(self, packet: EvidencePacket, item: EvidenceItem) -> None:
        evidence_type = item.evidence_type
        if evidence_type == EvidenceType.PRICE_PATH:
            packet.price_evidence.append(item)
        elif evidence_type == EvidenceType.NEWS:
            packet.news_evidence.append(item)
        elif evidence_type == EvidenceType.FILING:
            packet.filing_evidence.append(item)
        elif evidence_type == EvidenceType.MACRO:
            packet.macro_evidence.append(item)
        elif evidence_type == EvidenceType.SENTIMENT:
            packet.sentiment_evidence.append(item)
        elif evidence_type == EvidenceType.DISCUSSION:
            packet.discussion_evidence.append(item)
        elif evidence_type == EvidenceType.ANALOG_HISTORY:
            packet.analog_evidence.append(item)
        else:
            packet.market_regime_evidence.append(item)

