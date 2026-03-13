"""Unified agent tool specs and LangChain exposure."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, model_validator

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, ToolCallTrace
from ai_trading_coach.domain.react_models import ReActStep
from ai_trading_coach.modules.data_sources.yahoo_japan_fund_history import get_fund_history_by_code, get_fund_history_by_url
from ai_trading_coach.modules.mcp.adapters import normalize_tool_output
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, MCPToolRef, tool_payload_hash


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BraveSearchInput(BaseModel):
    query: str = Field(..., min_length=1)
    count: int = Field(default=5, ge=1, le=10)


class FirecrawlExtractInput(BaseModel):
    url: str = Field(..., min_length=3)


class PlaywrightFetchInput(BaseModel):
    url: str = Field(..., min_length=3)
    instruction: str = "extract main content"


class PriceHistoryInput(BaseModel):
    ticker: str = Field(..., min_length=1)
    period: str = "1mo"
    interval: str = "1d"


class TickerNewsInput(BaseModel):
    ticker: str = Field(..., min_length=1)


class YahooJapanFundHistoryInput(BaseModel):
    fund_code: str | None = None
    url: str | None = None
    max_pages: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_target(self) -> "YahooJapanFundHistoryInput":
        if not (self.fund_code or self.url):
            raise ValueError("Either fund_code or url is required")
        return self


@dataclass
class MCPToolRuntime:
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    react_steps: list[ReActStep] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    backend_type: str
    backend_ref: str
    invoke: Callable[[BaseModel], Any]


def make_langchain_tool(spec: RuntimeToolSpec, runtime: MCPToolRuntime) -> StructuredTool:
    def _invoke(**kwargs: Any) -> str:
        try:
            validated = spec.args_schema.model_validate(kwargs)
        except ValidationError as exc:
            return f"tool_error: invalid_input for {spec.name}: {exc.errors()}"
        return asyncio.run(
            _record_tool_attempt(
                runtime=runtime,
                action_name=spec.name,
                server_id=spec.backend_ref,
                tool_name=spec.name,
                arguments=validated.model_dump(mode="json"),
                run_call=lambda: _run_validated(spec, validated),
            )
        )

    return StructuredTool.from_function(func=_invoke, name=spec.name, description=spec.description, args_schema=spec.args_schema)


async def _run_validated(spec: RuntimeToolSpec, validated: BaseModel) -> tuple[list[EvidenceItem], str | None]:
    result = spec.invoke(validated)
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, tuple) and len(result) == 2:
        return result
    if isinstance(result, str) and result.startswith("tool_error:"):
        return [], result.split("tool_error:", 1)[1].strip()
    return [], None


def build_agent_tools(*, specs: list[RuntimeToolSpec], runtime: MCPToolRuntime) -> list[StructuredTool]:
    return [make_langchain_tool(spec, runtime) for spec in specs]


def make_mcp_price_history_spec(tool_ref: MCPToolRef, mcp_manager: MCPClientManager, *, description: str) -> RuntimeToolSpec:
    async def _invoke(payload: PriceHistoryInput) -> tuple[list[EvidenceItem], str | None]:
        arguments = payload.model_dump(mode="json")
        raw = await mcp_manager.call_tool(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, arguments=arguments)
        subtask = PlanSubTask(subtask_id=f"price_{payload.ticker}", objective=f"Get price history for {payload.ticker}", tool_category="market_data", evidence_type=EvidenceType.PRICE_PATH, query={"period": payload.period, "interval": payload.interval}, tickers=[payload.ticker])
        return normalize_tool_output(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, subtask=subtask, raw_result=raw), None

    return RuntimeToolSpec("get_price_history", description, PriceHistoryInput, "mcp", tool_ref.key, _invoke)


def make_mcp_news_spec(tool_ref: MCPToolRef, mcp_manager: MCPClientManager, *, description: str) -> RuntimeToolSpec:
    async def _invoke(payload: TickerNewsInput) -> tuple[list[EvidenceItem], str | None]:
        raw = await mcp_manager.call_tool(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, arguments=payload.model_dump(mode="json"))
        subtask = PlanSubTask(subtask_id=f"news_{payload.ticker}", objective=f"Get latest news for {payload.ticker}", tool_category="news_search", evidence_type=EvidenceType.NEWS, tickers=[payload.ticker])
        return normalize_tool_output(server_id=tool_ref.server_id, tool_name=tool_ref.tool_name, subtask=subtask, raw_result=raw), None

    return RuntimeToolSpec("search_news", description, TickerNewsInput, "mcp", tool_ref.key, _invoke)


def make_local_fund_history_spec(*, description: str) -> RuntimeToolSpec:
    async def _invoke(payload: YahooJapanFundHistoryInput) -> tuple[list[EvidenceItem], str | None]:
        if payload.fund_code:
            result = await get_fund_history_by_code(payload.fund_code, max_pages=payload.max_pages)
        else:
            result = await get_fund_history_by_url(payload.url or "", max_pages=payload.max_pages)
        rows = result.get("rows", []) if isinstance(result, dict) else []
        source_id = f"src_yahoo_japan_fund_{payload.fund_code or 'url'}"
        return [EvidenceItem(item_id=f"ev_fund_{payload.fund_code or 'url'}", evidence_type="price_path", summary=f"fund={result.get('fund_name','')}; rows={len(rows)}", data={"payload": result}, related_tickers=[payload.fund_code] if payload.fund_code else [], sources=[{"source_id": source_id, "source_type": "web", "provider": "yahoo_japan", "uri": payload.url or "", "title": result.get("fund_name", ""), "published_at": None, "fetched_at": utc_now(), "reliability_score": 0.7}])], None

    return RuntimeToolSpec("yahoo_japan_fund_history", description, YahooJapanFundHistoryInput, "python", "local:yahoo_japan", _invoke)


async def _record_tool_attempt(*, runtime: MCPToolRuntime, action_name: str, server_id: str, tool_name: str, arguments: dict[str, Any], run_call: Callable[[], Any]) -> str:
    step = ReActStep(step_index=len(runtime.react_steps) + 1, thought=f"Call {action_name}", action=action_name, action_input=arguments, started_at=utc_now())
    started = utc_now()
    t0 = perf_counter()
    items: list[EvidenceItem] = []
    err: str | None = None
    try:
        items, err = await run_call()
        runtime.evidence_items.extend(items)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    step.success = err is None
    step.error_message = err
    step.evidence_item_ids = [i.item_id for i in items]
    step.observation_summary = err or f"Fetched {len(items)} items"
    step.ended_at = utc_now()
    runtime.react_steps.append(step)
    runtime.tool_traces.append(ToolCallTrace(call_id=f"tool_{tool_name}_{int(started.timestamp()*1000)}", tool_name=tool_name, server_id=server_id, request_summary=action_name, response_summary=f"items={len(items)}", payload_hash=tool_payload_hash(arguments), latency_ms=int((perf_counter()-t0)*1000), success=(err is None), error_message=err))
    return f"tool_error: {err}" if err else f"tool={action_name}; items={len(items)}"


__all__ = ["MCPToolRuntime", "RuntimeToolSpec", "BraveSearchInput", "FirecrawlExtractInput", "PlaywrightFetchInput", "PriceHistoryInput", "TickerNewsInput", "YahooJapanFundHistoryInput", "build_agent_tools", "make_mcp_price_history_spec", "make_mcp_news_spec", "make_local_fund_history_spec"]
