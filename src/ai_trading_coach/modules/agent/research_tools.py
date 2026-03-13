"""Unified tool registry/builder for the research agent."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import (
    BraveSearchInput,
    FirecrawlExtractInput,
    MCPToolRuntime,
    PlaywrightFetchInput,
    RuntimeToolSpec,
    build_agent_tools,
    make_local_fund_history_spec,
    make_mcp_news_spec,
    make_mcp_price_history_spec,
)
from ai_trading_coach.modules.agent.web_tools import make_brave_search, make_firecrawl_extract, make_playwright_fetch, web_tool_availability, web_tool_config
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


@dataclass(frozen=True)
class ResearchToolRegistration:
    agent_name: str
    backend_name: str
    backend_type: str
    available: bool
    reason: str | None = None
    tool: StructuredTool | None = None


def _all_specs(*, settings: Settings, mcp_manager: MCPClientManager) -> list[tuple[RuntimeToolSpec, bool, str | None]]:
    web_config = web_tool_config(settings=settings)
    web_statuses = web_tool_availability(settings=settings)
    specs: list[tuple[RuntimeToolSpec, bool, str | None]] = []

    price_ref, price_reason = mcp_manager.get_tool_ref("get_price_history")
    if price_ref:
        specs.append((make_mcp_price_history_spec(price_ref, mcp_manager, description="Get ticker historical prices."), True, None))

    news_ref, news_reason = mcp_manager.get_tool_ref("search_news")
    if news_ref:
        specs.append((make_mcp_news_spec(news_ref, mcp_manager, description="Get latest ticker news."), True, None))

    specs.append((make_local_fund_history_spec(description="Fetch Yahoo Japan fund history by fund code or URL."), True, None))

    brave_status = web_statuses["brave_search"]
    specs.append((RuntimeToolSpec("brave_search", "Search the web using Brave.", BraveSearchInput, "http", brave_status.backend, lambda x: make_brave_search(api_key=web_config.brave_api_key)(x.query, x.count)), brave_status.available, brave_status.reason))

    firecrawl_status = web_statuses["firecrawl_extract"]
    specs.append((RuntimeToolSpec("firecrawl_extract", "Extract full webpage content with Firecrawl.", FirecrawlExtractInput, "http", firecrawl_status.backend, lambda x: make_firecrawl_extract(api_key=web_config.firecrawl_api_key)(x.url)), firecrawl_status.available, firecrawl_status.reason))

    playwright_status = web_statuses["playwright_fetch"]
    specs.append((RuntimeToolSpec("playwright_fetch", "Fetch dynamically rendered webpage content.", PlaywrightFetchInput, "browser", playwright_status.backend, lambda x: make_playwright_fetch(endpoint=web_config.agent_browser_endpoint)(x.url, x.instruction)), playwright_status.available, playwright_status.reason))

    # If MCP ref missing, expose diagnostic rows.
    if not price_ref:
        specs.append((RuntimeToolSpec("get_price_history", "Get ticker historical prices.", BraveSearchInput, "mcp", "mcp:missing", lambda _: "tool_error: missing MCP mapping"), False, price_reason or "missing MCP mapping"))
    if not news_ref:
        specs.append((RuntimeToolSpec("search_news", "Get latest ticker news.", BraveSearchInput, "mcp", "mcp:missing", lambda _: "tool_error: missing MCP mapping"), False, news_reason or "missing MCP mapping"))

    return specs


def resolve_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime | None = None) -> list[ResearchToolRegistration]:
    specs = _all_specs(settings=settings, mcp_manager=mcp_manager)
    tools_by_name: dict[str, StructuredTool] = {}
    if runtime is not None:
        tools_by_name = {tool.name: tool for tool in build_agent_tools(specs=[s for s, ok, _ in specs if ok], runtime=runtime)}
    return [
        ResearchToolRegistration(agent_name=s.name, backend_name=s.backend_ref, backend_type=s.backend_type, available=ok, reason=reason, tool=tools_by_name.get(s.name))
        for s, ok, reason in specs
    ]


def build_runtime_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    return [item.tool for item in resolve_research_tools(settings=settings, mcp_manager=mcp_manager, runtime=runtime) if item.available and item.tool is not None]
