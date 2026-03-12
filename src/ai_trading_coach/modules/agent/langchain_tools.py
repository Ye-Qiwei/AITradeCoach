"""Build one-layer curated tools for the research agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Awaitable, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.models import EvidenceItem, ToolCallTrace
from ai_trading_coach.domain.react_models import ReActStep
from ai_trading_coach.modules.agent.curated_tools import CuratedToolDefinition
from ai_trading_coach.modules.data_sources.yahoo_japan_fund_history import get_fund_history_by_code, get_fund_history_by_url
from ai_trading_coach.modules.mcp.adapters import normalize_tool_output
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, MCPToolRef, tool_payload_hash


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CuratedToolInput(BaseModel):
    objective: str = ""
    judgement_ids: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    query: dict[str, Any] = Field(default_factory=dict)
    time_window: str | None = None
    fund_code: str | None = None
    url: str | None = None
    max_pages: int = 3


@dataclass
class MCPToolRuntime:
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    react_steps: list[ReActStep] = field(default_factory=list)


def build_agent_tools(*, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    """Backward-compatible wrapper around unified research tool builder."""
    from ai_trading_coach.modules.agent.research_tools import build_runtime_research_tools

    return build_runtime_research_tools(
        settings=mcp_manager.settings,
        mcp_manager=mcp_manager,
        runtime=runtime,
    )


def build_traced_structured_tool(*, name: str, description: str, server_id: str, runtime: MCPToolRuntime, handler: Callable[..., str]) -> StructuredTool:
    def _invoke(**kwargs: Any) -> str:
        return asyncio.run(
            _record_tool_attempt(
                runtime=runtime,
                action_name=name,
                server_id=server_id,
                tool_name=name,
                arguments=kwargs,
                run_call=lambda: _run_sync_handler(handler=handler, kwargs=kwargs),
            )
        )

    return StructuredTool.from_function(func=_invoke, name=name, description=description)


async def _run_sync_handler(*, handler: Callable[..., str], kwargs: dict[str, Any]) -> tuple[list[EvidenceItem], str | None]:
    result = handler(**kwargs)
    if isinstance(result, str) and result.strip().startswith("tool_error:"):
        return [], result.strip().split("tool_error:", 1)[1].strip() or "unknown tool error"
    return [], None


def _build_external_tool(spec: CuratedToolDefinition, tool_ref: MCPToolRef, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> StructuredTool:
    def _invoke(**kwargs: Any) -> str:
        return asyncio.run(_invoke_external(spec=spec, tool_ref=tool_ref, kwargs=kwargs, mcp_manager=mcp_manager, runtime=runtime))

    return StructuredTool.from_function(func=_invoke, name=spec.canonical_name, description=f"{spec.description} Use when: {spec.when_to_use}.")


def _build_local_tool(spec: CuratedToolDefinition, runtime: MCPToolRuntime) -> StructuredTool:
    def _invoke(**kwargs: Any) -> str:
        return asyncio.run(_invoke_local(spec=spec, kwargs=kwargs, runtime=runtime))

    return StructuredTool.from_function(func=_invoke, name=spec.canonical_name, description=f"{spec.description} Use when: {spec.when_to_use}.")


async def _invoke_external(*, spec: CuratedToolDefinition, tool_ref: MCPToolRef, kwargs: dict[str, Any], mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> str:
    validated: CuratedToolInput | None = None

    async def _run() -> tuple[list[EvidenceItem], str | None]:
        nonlocal validated
        validated = CuratedToolInput.model_validate(kwargs)
        subtask = PlanSubTask(
            subtask_id=f"react_{spec.canonical_name}_{len(runtime.tool_traces)+1}",
            objective=validated.objective or spec.description,
            tool_category=spec.tool_category,
            evidence_type=spec.evidence_type,
            query=validated.query,
            tickers=validated.tickers,
            time_window=validated.time_window,
        )
        args = {
            "objective": validated.objective,
            "query": validated.query,
            "tickers": validated.tickers,
            "time_window": validated.time_window,
            "judgement_ids": validated.judgement_ids,
        }
        raw = await mcp_manager.call_tool(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, arguments=args)
        items = normalize_tool_output(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, subtask=subtask, raw_result=raw)
        for item in items:
            item.extensions["judgement_ids"] = validated.judgement_ids
        runtime.evidence_items.extend(items)
        return items, None

    return await _record_tool_attempt(
        runtime=runtime,
        action_name=spec.canonical_name,
        server_id=tool_ref.server_id,
        tool_name=tool_ref.tool_name,
        arguments=kwargs,
        run_call=_run,
    )


async def _invoke_local(*, spec: CuratedToolDefinition, kwargs: dict[str, Any], runtime: MCPToolRuntime) -> str:
    validated: CuratedToolInput | None = None

    async def _run() -> tuple[list[EvidenceItem], str | None]:
        nonlocal validated
        validated = CuratedToolInput.model_validate(kwargs)
        if validated.fund_code:
            payload = await get_fund_history_by_code(validated.fund_code, max_pages=validated.max_pages)
        elif validated.url:
            payload = await get_fund_history_by_url(validated.url, max_pages=validated.max_pages)
        else:
            raise ValueError("Provide either fund_code or url")
        items = _fund_history_to_evidence(payload, validated)
        runtime.evidence_items.extend(items)
        return items, None

    return await _record_tool_attempt(
        runtime=runtime,
        action_name=spec.canonical_name,
        server_id="local_python",
        tool_name=spec.canonical_name,
        arguments=kwargs,
        run_call=_run,
    )


async def _record_tool_attempt(
    *,
    runtime: MCPToolRuntime,
    action_name: str,
    server_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    run_call: Callable[[], Awaitable[tuple[list[EvidenceItem], str | None]]],
) -> str:
    step = _start_step(action_name, arguments, runtime)
    started = utc_now()
    t0 = perf_counter()
    items: list[EvidenceItem] = []
    err: str | None = None
    try:
        items, explicit_err = await run_call()
        err = explicit_err
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    _finish_trace(runtime, step, started, t0, server_id, tool_name, arguments, items, err)
    return _build_observation(action_name, items, err)


def _fund_history_to_evidence(payload: dict[str, Any], validated: CuratedToolInput) -> list[EvidenceItem]:
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    source_id = f"src_yahoo_japan_fund_{validated.fund_code or 'url'}"
    return [
        EvidenceItem(
            item_id=f"ev_fund_{validated.fund_code or 'url'}",
            evidence_type="price_path",
            summary=f"fund_code={payload.get('fund_code')}; rows={len(rows)}; latest={rows[0].get('date') if rows else 'na'}",
            data={"fund_name": payload.get("fund_name"), "rows": rows[:20], "row_count": payload.get("row_count", len(rows)), "debug": payload.get("debug", [])},
            related_tickers=[validated.fund_code] if validated.fund_code else [],
            sources=[{"source_id": source_id, "source_type": "mcp", "provider": "yahoo_japan", "uri": validated.url or f"https://finance.yahoo.co.jp/quote/{validated.fund_code}/history", "title": payload.get("fund_name"), "published_at": None, "fetched_at": utc_now(), "reliability_score": 0.7}],
        )
    ]


def _start_step(action_name: str, action_input: dict[str, Any], runtime: MCPToolRuntime) -> ReActStep:
    step = ReActStep(step_index=len(runtime.react_steps)+1, thought=f"Call curated tool {action_name}", action=action_name, action_input=action_input, started_at=utc_now())
    return step


def _finish_trace(runtime: MCPToolRuntime, step: ReActStep, started: datetime, t0: float, server_id: str, tool_name: str, arguments: dict[str, Any], items: list[EvidenceItem], err: str | None) -> None:
    success = err is None
    step.success = success
    step.error_message = err
    step.evidence_item_ids = [item.item_id for item in items]
    step.observation_summary = err or f"Fetched {len(items)} items"
    step.ended_at = utc_now()
    runtime.react_steps.append(step)
    runtime.tool_traces.append(ToolCallTrace(call_id=f"tool_{tool_name}_{int(started.timestamp()*1000)}", tool_name=tool_name, server_id=server_id, request_summary=f"curated={step.action}", response_summary=f"items={len(items)}", payload_hash=tool_payload_hash(arguments), latency_ms=int((perf_counter()-t0)*1000), success=success, error_message=err))


def _build_observation(name: str, items: list[EvidenceItem], err: str | None) -> str:
    if err:
        return f"tool_error: {err}"
    return f"tool={name}; items={len(items)}; evidence_item_ids={[i.item_id for i in items][:5]}"


__all__ = [
    "MCPToolRuntime",
    "build_agent_tools",
    "build_traced_structured_tool",
    "_build_external_tool",
    "_build_local_tool",
]
