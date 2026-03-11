"""LangChain tool wrappers over MCP backends for ReAct graph execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, ToolCallTrace
from ai_trading_coach.domain.react_models import ReActStep
from ai_trading_coach.modules.agent.web_tools import build_general_web_tools
from ai_trading_coach.modules.mcp.adapters import normalize_tool_output
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, tool_payload_hash


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MCPToolInput(BaseModel):
    objective: str = Field(default="")
    judgement_ids: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    query: dict[str, Any] = Field(default_factory=dict)
    time_window: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _coerce_query(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            return {"query": value.strip()}
        return {}


@dataclass
class MCPToolRuntime:
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    react_steps: list[ReActStep] = field(default_factory=list)


def build_langchain_mcp_tools(*, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    specs = (
        ("get_price_history", EvidenceType.PRICE_PATH),
        ("search_news", EvidenceType.NEWS),
        ("list_filings", EvidenceType.FILING),
        ("get_macro_series", EvidenceType.MACRO),
    )
    tools: list[StructuredTool] = []
    for action_name, evidence_type in specs:
        ref, _reason = mcp_manager.tool_configuration_status(evidence_type)
        if ref is None:
            continue
        tools.append(_build_tool(action_name, evidence_type, mcp_manager, runtime))
    tools.extend(build_general_web_tools(settings=mcp_manager.settings))
    return tools


def _build_tool(action_name: str, evidence_type: EvidenceType, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> StructuredTool:
    def _invoke(
        objective: str = "",
        judgement_ids: list[str] | None = None,
        tickers: list[str] | None = None,
        query: dict[str, Any] | None = None,
        time_window: str | None = None,
    ) -> str:
        validated = MCPToolInput(
            objective=objective,
            judgement_ids=judgement_ids or [],
            tickers=tickers or [],
            query=query or {},
            time_window=time_window,
        )
        return asyncio.run(
            _execute_tool_async(
                action_name=action_name,
                evidence_type=evidence_type,
                validated=validated,
                mcp_manager=mcp_manager,
                runtime=runtime,
            )
        )

    return StructuredTool.from_function(
        func=_invoke,
        name=action_name,
        description=f"Fetch {evidence_type.value} evidence via MCP. Include judgement_ids when possible.",
    )


async def _execute_tool_async(*, action_name: str, evidence_type: EvidenceType, validated: MCPToolInput, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> str:
    step_index = len(runtime.react_steps) + 1
    step = ReActStep(
        step_index=step_index,
        thought=f"Call {action_name} for judgement_ids={validated.judgement_ids}",
        action=action_name,
        action_input=validated.model_dump(mode="json"),
        started_at=utc_now(),
    )
    subtask = PlanSubTask(
        subtask_id=f"react_{action_name}_{len(runtime.tool_traces) + 1}",
        objective=validated.objective or f"ReAct fetch via {action_name}",
        tool_category=_tool_category(evidence_type),
        evidence_type=evidence_type,
        query=validated.query,
        tickers=validated.tickers,
        time_window=validated.time_window,
    )

    tool_ref = mcp_manager.resolve_tool(evidence_type)
    request_payload = {
        "objective": subtask.objective,
        "query": subtask.query,
        "tickers": subtask.tickers,
        "time_window": subtask.time_window,
        "judgement_ids": validated.judgement_ids,
    }
    digest = tool_payload_hash(request_payload)
    started = utc_now()
    t0 = perf_counter()
    success = True
    err: str | None = None
    items: list[EvidenceItem] = []
    try:
        raw = await mcp_manager.call_tool(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, arguments=request_payload)
        items = normalize_tool_output(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, subtask=subtask, raw_result=raw)
        for item in items:
            item.extensions["judgement_ids"] = validated.judgement_ids
        runtime.evidence_items.extend(items)
    except Exception as exc:  # noqa: BLE001
        success = False
        err = str(exc)

    latency_ms = int((perf_counter() - t0) * 1000)
    trace = ToolCallTrace(
        call_id=f"tool_{subtask.subtask_id}_{int(started.timestamp() * 1000)}",
        tool_name=tool_ref.tool_name,
        server_id=tool_ref.server_id,
        request_summary=f"react_tool={action_name};judgements={validated.judgement_ids}",
        response_summary=f"items={len(items)}",
        payload_hash=digest,
        latency_ms=latency_ms,
        success=success,
        error_message=err,
    )
    runtime.tool_traces.append(trace)
    step.success = success
    step.error_message = err
    step.evidence_item_ids = [item.item_id for item in items]
    step.observation_summary = err if err else f"Fetched {len(items)} items"
    step.ended_at = utc_now()
    runtime.react_steps.append(step)

    if not success:
        return f"tool_error: {err}"
    return _build_observation(tool_ref.key, items)


def _compact_data_fields(data: dict[str, Any]) -> dict[str, Any]:
    allow = {"close", "change_pct", "series_id", "filing_type", "company"}
    return {k: v for k, v in data.items() if k in allow}


def _truncate(text: str, limit: int = 280) -> str:
    plain = " ".join(str(text).split())
    return plain if len(plain) <= limit else plain[: limit - 3] + "..."


def _build_observation(tool_key: str, items: list[EvidenceItem]) -> str:
    top = items[:4]
    lines = [f"tool={tool_key}; items={len(items)}; evidence_item_ids={[item.item_id for item in items]}"]
    for item in top:
        lines.append(
            _truncate(
                f"item_id={item.item_id}; judgement_ids={item.extensions.get('judgement_ids', [])}; "
                f"title={item.title or ''}; summary={item.summary}; "
                f"source_ids={[src.source_id for src in item.sources]}; "
                f"related_tickers={item.related_tickers}; data={_compact_data_fields(item.data)}"
            )
        )
    if len(items) > len(top):
        lines.append(f"truncated={len(items) - len(top)} more items not expanded")
    return "\n".join(lines)


def _tool_category(evidence_type: EvidenceType) -> str:
    if evidence_type == EvidenceType.PRICE_PATH:
        return "market_data"
    if evidence_type == EvidenceType.NEWS:
        return "news_search"
    if evidence_type == EvidenceType.FILING:
        return "filings_financials"
    return "macro_data"


__all__ = ["MCPToolRuntime", "build_langchain_mcp_tools"]
