"""Single source of truth for research tool registration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.curated_tools import CuratedToolDefinition, enabled_curated_tools
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, _build_external_tool, _build_local_tool
from ai_trading_coach.modules.agent.web_tools import brave_search, firecrawl_extract, playwright_fetch, web_tool_availability
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager, MCPToolRef


@dataclass(frozen=True)
class ResearchToolRegistration:
    agent_name: str
    backend_name: str
    category: str
    available: bool
    reason: str | None = None
    tool: StructuredTool | None = None


def resolve_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime | None = None) -> list[ResearchToolRegistration]:
    registrations: list[ResearchToolRegistration] = []
    tool_runtime = runtime or MCPToolRuntime()

    for spec in enabled_curated_tools():
        if spec.implementation_kind == "local_python":
            registrations.append(
                ResearchToolRegistration(
                    agent_name=spec.canonical_name,
                    backend_name=f"local:{spec.implementation_ref or spec.canonical_name}",
                    category="local_python",
                    available=True,
                    tool=_build_local_tool(spec, tool_runtime) if runtime is not None else None,
                )
            )
            continue

        ref, reason = mcp_manager.curated_tool_status(spec.canonical_name)
        backend_name = ref.key if ref is not None else "mcp:unresolved"
        registrations.append(
            ResearchToolRegistration(
                agent_name=spec.canonical_name,
                backend_name=backend_name,
                category="external_mcp",
                available=ref is not None,
                reason=reason,
                tool=_build_external_tool(spec, ref, mcp_manager, tool_runtime) if (runtime is not None and ref is not None) else None,
            )
        )

    for name, handler, description in (
        ("brave_search", brave_search, "Broad web search with Brave Search API."),
        ("firecrawl_extract", firecrawl_extract, "Fetch full content from a target URL with Firecrawl API."),
        ("playwright_fetch", playwright_fetch, "Fetch dynamically rendered page content via browser runtime."),
    ):
        status = web_tool_availability(settings=settings)[name]
        registrations.append(
            ResearchToolRegistration(
                agent_name=name,
                backend_name=status.backend,
                category="web",
                available=status.available,
                reason=status.reason,
                tool=StructuredTool.from_function(func=handler, name=name, description=description) if (runtime is not None and status.available) else None,
            )
        )

    return registrations


def build_runtime_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    registrations = resolve_research_tools(settings=settings, mcp_manager=mcp_manager, runtime=runtime)
    return [item.tool for item in registrations if item.tool is not None]
