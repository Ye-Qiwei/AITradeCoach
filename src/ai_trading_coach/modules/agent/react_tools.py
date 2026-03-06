"""Toolbox for ReAct research stage, backed by MCP tool manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket, ToolCallTrace
from ai_trading_coach.modules.mcp.adapters import normalize_tool_output
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, tool_payload_hash


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _ToolInput(BaseModel):
    objective: str = Field(default="")
    tickers: list[str] = Field(default_factory=list)
    query: dict[str, Any] = Field(default_factory=dict)
    time_window: str | None = None


@dataclass
class ReactToolOutcome:
    evidence_items: list[EvidenceItem]
    tool_trace: ToolCallTrace
    observation_summary: str
    success: bool
    error_message: str | None = None


class ReactResearchTools:
    def __init__(self, *, mcp_manager: MCPClientManager, max_items_per_tool: int = 8) -> None:
        self.mcp_manager = mcp_manager
        self.max_items_per_tool = max_items_per_tool

    def execute(self, *, tool_name: str, arguments: dict[str, Any], step_id: str) -> ReactToolOutcome:
        return asyncio.run(self._execute_async(tool_name=tool_name, arguments=arguments, step_id=step_id))

    async def _execute_async(self, *, tool_name: str, arguments: dict[str, Any], step_id: str) -> ReactToolOutcome:
        evidence_type = _tool_evidence_type(tool_name)
        if evidence_type is None:
            trace = ToolCallTrace(
                call_id=f"tool_{step_id}",
                tool_name=tool_name,
                server_id="virtual",
                request_summary="unsupported tool",
                response_summary="items=0",
                payload_hash=tool_payload_hash(arguments),
                latency_ms=0,
                success=False,
                error_message=f"unsupported tool: {tool_name}",
            )
            return ReactToolOutcome([], trace, "unsupported tool", False, trace.error_message)

        validated = _ToolInput.model_validate(arguments)
        subtask = PlanSubTask(
            subtask_id=step_id,
            objective=validated.objective or f"ReAct fetch via {tool_name}",
            tool_category=_tool_category(evidence_type),
            evidence_type=evidence_type,
            query=validated.query,
            tickers=validated.tickers,
            time_window=validated.time_window,
        )

        tool_ref = self.mcp_manager.resolve_tool(evidence_type)
        request_payload = {
            "objective": subtask.objective,
            "query": subtask.query,
            "tickers": subtask.tickers,
            "time_window": subtask.time_window,
        }
        digest = tool_payload_hash(request_payload)
        started = utc_now()
        t0 = perf_counter()
        success = True
        err: str | None = None
        items: list[EvidenceItem] = []
        try:
            raw = await self.mcp_manager.call_tool(
                server_id=tool_ref.server_id,
                tool_name=tool_ref.tool_name,
                arguments=request_payload,
            )
            items = normalize_tool_output(
                server_id=tool_ref.server_id,
                tool_name=tool_ref.tool_name,
                subtask=subtask,
                raw_result=raw,
            )[: self.max_items_per_tool]
        except Exception as exc:  # noqa: BLE001
            success = False
            err = str(exc)

        latency_ms = int((perf_counter() - t0) * 1000)
        trace = ToolCallTrace(
            call_id=f"tool_{step_id}_{int(started.timestamp() * 1000)}",
            tool_name=tool_ref.tool_name,
            server_id=tool_ref.server_id,
            request_summary=f"react_tool={tool_name}",
            response_summary=f"items={len(items)}",
            payload_hash=digest,
            latency_ms=latency_ms,
            success=success,
            error_message=err,
        )
        observation = err if err else f"Fetched {len(items)} items from {tool_ref.key}"
        return ReactToolOutcome(items, trace, observation, success, err)


def build_evidence_packet(*, packet_id: str, user_id: str, evidence_items: list[EvidenceItem]) -> EvidencePacket:
    packet = EvidencePacket(packet_id=packet_id, user_id=user_id)
    for item in evidence_items:
        packet.source_registry.extend(item.sources)
        if item.evidence_type == EvidenceType.PRICE_PATH:
            packet.price_evidence.append(item)
        elif item.evidence_type == EvidenceType.NEWS:
            packet.news_evidence.append(item)
        elif item.evidence_type == EvidenceType.FILING:
            packet.filing_evidence.append(item)
        elif item.evidence_type == EvidenceType.MACRO:
            packet.macro_evidence.append(item)
        elif item.evidence_type == EvidenceType.SENTIMENT:
            packet.sentiment_evidence.append(item)
        elif item.evidence_type == EvidenceType.DISCUSSION:
            packet.discussion_evidence.append(item)
        elif item.evidence_type == EvidenceType.ANALOG_HISTORY:
            packet.analog_evidence.append(item)
        else:
            packet.market_regime_evidence.append(item)
    packet.completeness_score = 1.0 if evidence_items else 0.0
    return packet


def _tool_evidence_type(tool_name: str) -> EvidenceType | None:
    mapping = {
        "get_price_history": EvidenceType.PRICE_PATH,
        "search_news": EvidenceType.NEWS,
        "list_filings": EvidenceType.FILING,
        "get_macro_series": EvidenceType.MACRO,
    }
    return mapping.get(tool_name)


def _tool_category(evidence_type: EvidenceType) -> str:
    if evidence_type == EvidenceType.PRICE_PATH:
        return "market_data"
    if evidence_type == EvidenceType.NEWS:
        return "news_search"
    if evidence_type == EvidenceType.FILING:
        return "filings_financials"
    return "macro_data"
