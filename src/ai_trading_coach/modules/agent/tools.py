"""MVP research tools: schema, availability, invocation, traces, and LangChain exposure."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Literal
from urllib import parse, request

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, ToolCallTrace
from ai_trading_coach.domain.react_models import ReActStep
from ai_trading_coach.modules.data_sources.yahoo_japan_fund_history import get_fund_history_by_code, get_fund_history_by_url
from ai_trading_coach.modules.mcp.adapters import extract_mcp_error, normalize_tool_output, parse_yfinance_price_history_result
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, MCPToolDefinition


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class YFinanceSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Search text for company, ETF, fund, or ticker discovery.")
    search_type: Literal["all", "quotes", "news"] = Field(default="all")


class YFinanceTickerInfoInput(BaseModel):
    symbol: str = Field(..., min_length=1, description="Resolved ticker symbol such as AAPL or 7203.T.")


class YFinanceTickerNewsInput(BaseModel):
    symbol: str = Field(..., min_length=1, description="Resolved ticker symbol such as AAPL or 7203.T.")


class YFinanceGetTopInput(BaseModel):
    sector: str = Field(..., min_length=1)
    top_type: Literal["gainers", "losers", "actives"]
    top_n: int = Field(default=10, ge=1, le=100)


class YFinancePriceHistoryInput(BaseModel):
    symbol: str = Field(..., min_length=1)
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"] = "6mo"
    interval: Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"] = "1d"
    chart_type: Literal["line", "candle", "ohlc"] | None = None


class YahooJapanFundHistoryInput(BaseModel):
    fund_code: str | None = Field(default=None, description="Yahoo Japan fund code. Use this when known.")
    url: str | None = Field(default=None, description="Yahoo Japan fund page URL. Use when code is unknown.")
    max_pages: int = Field(default=3, ge=1, le=10, description="Max pagination pages to crawl.")

    @model_validator(mode="after")
    def validate_target(self) -> "YahooJapanFundHistoryInput":
        if not (self.fund_code or self.url):
            raise ValueError("Either fund_code or url is required")
        return self


class BraveSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Web search query for collecting candidate URLs.")
    count: int = Field(5, ge=1, le=10, description="Number of results to return.")


class FirecrawlExtractInput(BaseModel):
    url: str = Field(..., min_length=5, description="Target article URL selected from search results.")


class PlaywrightFetchInput(BaseModel):
    url: str = Field(..., min_length=5, description="Dynamic webpage URL when normal extraction fails.")
    instruction: str = Field("extract main content", description="Short extraction instruction for browser agent.")


@dataclass
class ToolResult:
    observation_text: str
    response_summary: str
    evidence_items: list[EvidenceItem]
    raw_output: Any
    error_message: str | None


@dataclass
class ToolRuntime:
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    react_steps: list[ReActStep] = field(default_factory=list)
    next_step_index: int = 1
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class ToolAvailability:
    name: str
    available: bool
    backend: str
    reason: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    backend: str
    invoke: Callable[[BaseModel], Any]


def get_tool_availability(settings: Settings, mcp_manager: MCPClientManager) -> list[ToolAvailability]:
    yfinance_refs = mcp_manager.list_server_tools("yfinance")
    return [
        *[ToolAvailability(name=t.tool_name, available=True, backend=f"{t.server_id}:{t.tool_name}") for t in yfinance_refs],
        ToolAvailability("yahoo_japan_fund_history", True, "python:yahoo_japan"),
        ToolAvailability("brave_search", bool(settings.brave_api_key.strip()), "http:brave", None if settings.brave_api_key.strip() else "BRAVE_API_KEY missing"),
        ToolAvailability("firecrawl_extract", bool(settings.firecrawl_api_key.strip()), "http:firecrawl", None if settings.firecrawl_api_key.strip() else "FIRECRAWL_API_KEY missing"),
        ToolAvailability("playwright_fetch", bool(settings.agent_browser_endpoint.strip()), "browser:http_bridge" if settings.agent_browser_endpoint.strip() else "browser:disabled", None if settings.agent_browser_endpoint.strip() else "AGENT_BROWSER_ENDPOINT missing"),
    ]


def _tool_specs(settings: Settings, mcp_manager: MCPClientManager) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for mcp_tool in mcp_manager.list_server_tools("yfinance"):
        args_schema = _yfinance_args_schema(mcp_tool)
        description = _yfinance_description(mcp_tool.tool_name, mcp_tool.description)

        async def _generic_mcp(payload: BaseModel, tool: MCPToolDefinition = mcp_tool) -> ToolResult:
            raw = await mcp_manager.call_tool(server_id=tool.server_id, tool_name=tool.tool_name, arguments=payload.model_dump(mode="json", exclude_none=True))
            detected_error = extract_mcp_error(raw)
            if detected_error:
                return ToolResult(observation_text=f"tool_error: {detected_error}", response_summary=f"error: {detected_error}", evidence_items=[], raw_output=raw, error_message=detected_error)
            if tool.tool_name == "yfinance_get_ticker_info":
                symbol = str(getattr(payload, "symbol", ""))
                items = normalize_tool_output(server_id=tool.server_id, tool_name=tool.tool_name, subtask=PlanSubTask(subtask_id=f"ticker_info_{symbol}", objective=f"info for {symbol}", tool_category="market_data", evidence_type=EvidenceType.PRICE_PATH, tickers=[symbol]), raw_result=raw)
                return ToolResult(observation_text=f"ticker info for {symbol}; items={len(items)}", response_summary=f"symbol=\"{symbol}\"; items={len(items)}", evidence_items=items, raw_output=raw, error_message=None)
            if tool.tool_name == "yfinance_get_ticker_news":
                symbol = str(getattr(payload, "symbol", ""))
                items = normalize_tool_output(server_id=tool.server_id, tool_name=tool.tool_name, subtask=PlanSubTask(subtask_id=f"ticker_news_{symbol}", objective=f"ticker news for {symbol}", tool_category="news", evidence_type=EvidenceType.NEWS, tickers=[symbol]), raw_result=raw)
                excerpt = items[0].summary if items else "no items"
                return ToolResult(observation_text=f"ticker news for {symbol}: {excerpt}", response_summary=f"symbol=\"{symbol}\"; items={len(items)}", evidence_items=items, raw_output=raw, error_message=None)
            if tool.tool_name == "yfinance_get_price_history":
                symbol = str(getattr(payload, "symbol", ""))
                rows = parse_yfinance_price_history_result(raw)
                if not rows:
                    message = "unable to parse price history table"
                    return ToolResult(observation_text=f"tool_error: {message}", response_summary=f"error: {message}", evidence_items=[], raw_output=raw, error_message=message)
                latest_row = rows[-1]
                first_row = rows[0]
                last_close = _to_float(latest_row.get("Close"))
                first_close = _to_float(first_row.get("Close"))
                if last_close is None:
                    message = "price history table missing numeric Close"
                    return ToolResult(observation_text=f"tool_error: {message}", response_summary=f"error: {message}", evidence_items=[], raw_output=raw, error_message=message)
                pct_change = ((last_close - first_close) / first_close * 100.0) if first_close not in (None, 0.0) else None
                pct_change_str = "n/a" if pct_change is None else f"{pct_change:.2f}%"
                item = EvidenceItem(
                    evidence_type=EvidenceType.PRICE_PATH,
                    summary=f"{symbol} price history rows={len(rows)}; first_close={_format_float(first_close)}; last_close={_format_float(last_close)}; change={pct_change_str}",
                    data={"symbol": symbol, "latest_row": latest_row, "rows": rows, "first_close": first_close, "last_close": last_close, "percent_change": pct_change},
                    related_tickers=[symbol.upper()] if symbol else [],
                    sources=[{"source_type": "market_data", "provider": "yfinance", "uri": f"https://finance.yahoo.com/quote/{parse.quote(symbol)}", "title": f"Yahoo Finance {symbol}"}],
                )
                return ToolResult(observation_text=f"price history for {symbol}: rows={len(rows)}", response_summary=f"symbol=\"{symbol}\"; rows={len(rows)}", evidence_items=[item], raw_output=raw, error_message=None)
            return ToolResult(observation_text=f"{tool.tool_name} completed", response_summary=f"tool={tool.tool_name}", evidence_items=[], raw_output=raw, error_message=None)

        specs.append(ToolSpec(mcp_tool.tool_name, description, args_schema, f"{mcp_tool.server_id}:{mcp_tool.tool_name}", _generic_mcp))

    async def _fund(payload: YahooJapanFundHistoryInput) -> ToolResult:
        raw = get_fund_history_by_code(payload.fund_code, max_pages=payload.max_pages) if payload.fund_code else get_fund_history_by_url(payload.url or "", max_pages=payload.max_pages)
        latest = raw[0] if raw else {}
        source_url = payload.url or (f"https://finance.yahoo.co.jp/quote/{payload.fund_code}" if payload.fund_code else "https://finance.yahoo.co.jp")
        item = EvidenceItem(
            evidence_type=EvidenceType.PRICE_PATH,
            summary=f"fund history rows={len(raw)}; latest_date={latest.get('date','')}; latest_nav={latest.get('基準価額','')}",
            data={"rows": raw[:100], "latest": latest, "fund_code": payload.fund_code, "source_url": source_url},
            related_tickers=[payload.fund_code] if payload.fund_code else [],
            sources=[{"source_type": "market_data", "provider": "yahoo_japan", "uri": source_url, "title": "Yahoo Japan fund history"}],
        )
        return ToolResult(observation_text=f"fund history rows={len(raw)}", response_summary=f"rows={len(raw)}", evidence_items=[item], raw_output=raw, error_message=None)

    specs.append(ToolSpec("yahoo_japan_fund_history", "Use for Yahoo Japan fund historical NAV only. Can retrieve historical NAV. Cannot retrieve company news or financial statements. For fund-like tickers such as 0331C177.T, use this tool instead of yfinance price tools. Inputs: fund_code or url, and max_pages. Outputs: historical NAV rows.", YahooJapanFundHistoryInput, "python:yahoo_japan", _fund))

    async def _brave(payload: BraveSearchInput) -> ToolResult:
        url = "https://api.search.brave.com/res/v1/web/search?" + parse.urlencode({"q": payload.query, "count": payload.count})
        req = request.Request(url, headers={"Accept": "application/json", "X-Subscription-Token": settings.brave_api_key})
        with request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode("utf-8"))
        results = raw.get("web", {}).get("results", [])[: payload.count]
        lines = [f"query=\"{payload.query}\"; results={len(results)}"]
        for i, row in enumerate(results, start=1):
            lines.append(f"[{i}] title=\"{row.get('title','')}\"; url=\"{row.get('url','')}\"")
        text = "\n".join(lines)
        return ToolResult(observation_text=text, response_summary=text, evidence_items=[], raw_output=raw, error_message=None)

    specs.append(ToolSpec("brave_search", "Use for external URL discovery. Do not use for final evidence extraction. Inputs: query, count. Outputs: search result metadata only.", BraveSearchInput, "http:brave", _brave))

    async def _firecrawl(payload: FirecrawlExtractInput) -> ToolResult:
        req = request.Request("https://api.firecrawl.dev/v1/scrape", headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.firecrawl_api_key}"}, data=json.dumps({"url": payload.url, "formats": ["markdown"]}).encode("utf-8"), method="POST")
        with request.urlopen(req, timeout=20) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode("utf-8"))
        payload_data = raw.get("data", {})
        metadata = payload_data.get("metadata", {}) if isinstance(payload_data, dict) else {}
        md = str(payload_data.get("markdown", "")) if isinstance(payload_data, dict) else ""
        title = str(metadata.get("title") or payload_data.get("title") or "") if isinstance(payload_data, dict) else ""
        item = EvidenceItem(evidence_type=EvidenceType.NEWS, summary=f"{title}; chars={len(md)}", data={"url": payload.url, "markdown": md[:8000]}, sources=[{"source_type": "web", "provider": "firecrawl", "uri": payload.url, "title": title}])
        summary = f"title=\"{title}\"; chars={len(md)}"
        return ToolResult(observation_text=f"url=\"{payload.url}\"\n{summary}", response_summary=summary, evidence_items=[item], raw_output=raw, error_message=None)

    specs.append(ToolSpec("firecrawl_extract", "Use to extract article/body content from a selected URL. Do not use for searching. Inputs: url. Outputs: markdown/content extraction.", FirecrawlExtractInput, "http:firecrawl", _firecrawl))

    async def _playwright(payload: PlaywrightFetchInput) -> ToolResult:
        req = request.Request(settings.agent_browser_endpoint, headers={"Content-Type": "application/json"}, data=json.dumps(payload.model_dump(mode="json")).encode("utf-8"), method="POST")
        with request.urlopen(req, timeout=25) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode("utf-8"))
        content = str(raw.get("content") or raw.get("markdown") or "")
        title = str(raw.get("title") or "")
        item = EvidenceItem(evidence_type=EvidenceType.NEWS, summary=f"{title}; chars={len(content)}", data={"url": payload.url, "content": content[:8000]}, sources=[{"source_type": "web", "provider": "playwright", "uri": payload.url, "title": title}])
        summary = f"title=\"{title}\"; chars={len(content)}"
        return ToolResult(observation_text=f"url=\"{payload.url}\"\n{summary}", response_summary=summary, evidence_items=[item], raw_output=raw, error_message=None)

    specs.append(ToolSpec("playwright_fetch", "Use only for JS-rendered pages when direct extraction is incomplete. Do not use for search or market prices. Inputs: url, instruction. Outputs: rendered page content.", PlaywrightFetchInput, "browser:http_bridge", _playwright))

    return specs


def build_runtime_tools(settings: Settings, mcp_manager: MCPClientManager, runtime: ToolRuntime) -> list[StructuredTool]:
    availability = {a.name: a for a in get_tool_availability(settings, mcp_manager)}
    tools: list[StructuredTool] = []
    for spec in _tool_specs(settings, mcp_manager):
        if not availability.get(spec.name, ToolAvailability(spec.name, False, spec.backend)).available:
            continue

        def _make(spec_: ToolSpec) -> Callable[..., str]:
            def _invoke(**kwargs: Any) -> str:
                try:
                    payload = spec_.args_schema.model_validate(kwargs)
                except ValidationError as exc:
                    return f"tool_error: invalid_input: {exc.errors()}"
                return asyncio.run(_record_call(runtime, spec_, payload))

            return _invoke

        tools.append(StructuredTool.from_function(func=_make(spec), name=spec.name, description=spec.description, args_schema=spec.args_schema))
    return tools


async def _record_call(runtime: ToolRuntime, spec: ToolSpec, payload: BaseModel) -> str:
    started = utc_now()
    t0 = perf_counter()
    payload_json = payload.model_dump(mode="json", by_alias=True)
    request_summary = "; ".join(f"{k}=\"{v}\"" if isinstance(v, str) else f"{k}={v}" for k, v in payload_json.items())
    async with runtime.lock:
        step_index = runtime.next_step_index
        runtime.next_step_index += 1
    step = ReActStep(step_index=step_index, thought=f"Call {spec.name}", action=spec.name, action_input=payload_json, started_at=started)
    try:
        result = spec.invoke(payload)
        if asyncio.iscoroutine(result):
            result = await result
        if not isinstance(result, ToolResult):
            raise ValueError("tool implementation must return ToolResult")
    except Exception as exc:  # noqa: BLE001
        result = ToolResult(observation_text=f"tool_error: {exc}", response_summary="error", evidence_items=[], raw_output={"error": str(exc)}, error_message=str(exc))
    runtime.evidence_items.extend(result.evidence_items)
    is_error = bool(result.error_message) or result.observation_text.startswith("tool_error:")
    step.success = not is_error
    step.error_message = result.error_message
    step.observation_summary = result.response_summary
    step.ended_at = utc_now()
    runtime.react_steps.append(step)
    runtime.tool_traces.append(
        ToolCallTrace(
            tool_name=spec.name,
            server_id=spec.backend,
            request_summary=request_summary,
            response_summary=result.response_summary,
            latency_ms=int((perf_counter() - t0) * 1000),
            success=not is_error,
            error_message=result.error_message,
            tool_input=payload_json,
            observation_text=result.observation_text,
            evidence_items=[item.model_dump(mode="json") for item in result.evidence_items],
            raw_output=result.raw_output,
        )
    )
    return result.observation_text


def _yfinance_args_schema(tool: MCPToolDefinition) -> type[BaseModel]:
    if tool.tool_name == "yfinance_search":
        return YFinanceSearchInput
    if tool.tool_name == "yfinance_get_ticker_info":
        return YFinanceTickerInfoInput
    if tool.tool_name == "yfinance_get_ticker_news":
        return YFinanceTickerNewsInput
    if tool.tool_name == "yfinance_get_top":
        return YFinanceGetTopInput
    if tool.tool_name == "yfinance_get_price_history":
        return YFinancePriceHistoryInput
    return _schema_model_from_mcp(tool.tool_name, tool.input_schema)


def _schema_model_from_mcp(tool_name: str, input_schema: dict[str, Any]) -> type[BaseModel]:
    props = input_schema.get("properties") if isinstance(input_schema, dict) else None
    required = input_schema.get("required") if isinstance(input_schema, dict) else None
    if not isinstance(props, dict):
        return create_model(f"{tool_name.title().replace('_', '')}Input")
    if not isinstance(required, list):
        required = []
    fields: dict[str, tuple[Any, Any]] = {}
    for key, schema in props.items():
        schema_type = schema.get("type") if isinstance(schema, dict) else None
        py_type: Any = str
        if schema_type == "integer":
            py_type = int
        elif schema_type == "number":
            py_type = float
        elif schema_type == "boolean":
            py_type = bool
        default = ... if key in required else None
        fields[key] = (py_type, default)
    return create_model(f"{tool_name.title().replace('_', '')}Input", **fields)


def _yfinance_description(tool_name: str, server_description: str) -> str:
    if tool_name == "yfinance_search":
        return "Use this first for ticker discovery when symbol certainty is low. Inputs: query, search_type. Outputs: candidate symbols/names. Do not use for OHLC history. Workflow: 1) yfinance_search, 2) yfinance_get_ticker_info, 3) price/news/fundamental tools."
    if tool_name == "yfinance_get_ticker_info":
        return "Use after yfinance_search to validate an exact ticker and inspect profile/fundamentals. Input: symbol. Output: ticker metadata. Do not use for historical OHLC or full article extraction."
    if tool_name == "yfinance_get_ticker_news":
        return "Use after ticker discovery for ticker-linked headlines. Input: symbol. Output: ticker news metadata. Do not use as a web crawler or for non-ticker macro discovery."
    if tool_name == "yfinance_get_price_history":
        return "Use for OHLC historical data after ticker discovery. Inputs: symbol, period, interval, chart_type. Output: historical OHLC table. Do not use to discover symbols. For Yahoo Japan fund codes like 0331C177.T, use yahoo_japan_fund_history instead."
    if tool_name == "yfinance_get_top":
        return "Use for sector top lists (gainers/losers/actives). Inputs: sector, top_type, top_n. Output: ranking list. Do not use for exact ticker discovery or OHLC history."
    return f"Use this yfinance MCP tool according to its server contract. Inputs follow the tool schema. Output is raw MCP response plus optional evidence extraction. Server description: {server_description}"


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


__all__ = ["ToolRuntime", "build_runtime_tools", "get_tool_availability"]
