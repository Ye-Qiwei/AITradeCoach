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
    PriceHistoryInput,
    RuntimeToolSpec,
    TickerNewsInput,
    build_agent_tools,
    make_local_fund_history_spec,
    make_mcp_news_spec,
    make_mcp_price_history_spec,
)
from ai_trading_coach.modules.agent.web_tools import make_brave_search, make_firecrawl_extract, make_playwright_fetch, web_tool_availability, web_tool_config
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


@dataclass(frozen=True)
class ResolvedResearchTool:
    spec: RuntimeToolSpec
    available: bool
    reason: str | None = None
    tool: StructuredTool | None = None


def _all_specs(*, settings: Settings, mcp_manager: MCPClientManager) -> list[ResolvedResearchTool]:
    web_config = web_tool_config(settings=settings)
    web_statuses = web_tool_availability(settings=settings)
    specs: list[ResolvedResearchTool] = []

    price_ref, price_reason = mcp_manager.get_tool_ref("get_price_history")
    if price_ref:
        specs.append(
            ResolvedResearchTool(
                spec=make_mcp_price_history_spec(
                    price_ref,
                    mcp_manager,
                    description="Get ticker historical prices.",
                ),
                available=True,
            )
        )
    else:
        specs.append(
            ResolvedResearchTool(
                spec=RuntimeToolSpec(
                    "get_price_history",
                    "Get ticker historical prices.",
                    PriceHistoryInput,
                    "market_data",
                    "mcp",
                    "mcp:missing",
                    lambda _: "tool_error: missing MCP mapping",
                ),
                available=False,
                reason=price_reason or "missing MCP mapping",
            )
        )

    news_ref, news_reason = mcp_manager.get_tool_ref("search_news")
    if news_ref:
        specs.append(
            ResolvedResearchTool(
                spec=make_mcp_news_spec(
                    news_ref,
                    mcp_manager,
                    description="Get latest ticker news.",
                ),
                available=True,
            )
        )
    else:
        specs.append(
            ResolvedResearchTool(
                spec=RuntimeToolSpec(
                    "search_news",
                    "Get latest ticker news.",
                    TickerNewsInput,
                    "ticker_news",
                    "mcp",
                    "mcp:missing",
                    lambda _: "tool_error: missing MCP mapping",
                ),
                available=False,
                reason=news_reason or "missing MCP mapping",
            )
        )

    specs.append(
        ResolvedResearchTool(
            spec=make_local_fund_history_spec(
                description="Fetch Yahoo Japan fund history by fund code or URL."
            ),
            available=True,
        )
    )

    brave_status = web_statuses["brave_search"]
    specs.append(
        ResolvedResearchTool(
            spec=RuntimeToolSpec(
                "brave_search",
                "Search the web using Brave.",
                BraveSearchInput,
                "general_web",
                "http",
                brave_status.backend,
                lambda x: make_brave_search(api_key=web_config.brave_api_key)(x.query, x.count),
            ),
            available=brave_status.available,
            reason=brave_status.reason,
        )
    )

    firecrawl_status = web_statuses["firecrawl_extract"]
    specs.append(
        ResolvedResearchTool(
            spec=RuntimeToolSpec(
                "firecrawl_extract",
                "Extract full webpage content with Firecrawl.",
                FirecrawlExtractInput,
                "general_web",
                "http",
                firecrawl_status.backend,
                lambda x: make_firecrawl_extract(api_key=web_config.firecrawl_api_key)(x.url),
            ),
            available=firecrawl_status.available,
            reason=firecrawl_status.reason,
        )
    )

    playwright_status = web_statuses["playwright_fetch"]
    specs.append(
        ResolvedResearchTool(
            spec=RuntimeToolSpec(
                "playwright_fetch",
                "Fetch dynamically rendered webpage content.",
                PlaywrightFetchInput,
                "general_web",
                "browser",
                playwright_status.backend,
                lambda x: make_playwright_fetch(endpoint=web_config.agent_browser_endpoint)(x.url, x.instruction),
            ),
            available=playwright_status.available,
            reason=playwright_status.reason,
        )
    )

    return specs


def resolve_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime | None = None) -> list[ResolvedResearchTool]:
    resolved = _all_specs(settings=settings, mcp_manager=mcp_manager)
    if runtime is None:
        return resolved

    enabled_specs = [item.spec for item in resolved if item.available]
    tools_by_name = {tool.name: tool for tool in build_agent_tools(specs=enabled_specs, runtime=runtime)}
    return [
        ResolvedResearchTool(
            spec=item.spec,
            available=item.available,
            reason=item.reason,
            tool=tools_by_name.get(item.spec.name),
        )
        for item in resolved
    ]


def build_runtime_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    return [item.tool for item in resolve_research_tools(settings=settings, mcp_manager=mcp_manager, runtime=runtime) if item.available and item.tool is not None]
