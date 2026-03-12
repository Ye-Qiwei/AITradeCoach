"""Single source of truth for research tool registration."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.curated_tools import enabled_curated_tools
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, _build_external_tool, _build_local_tool, build_traced_structured_tool
from ai_trading_coach.modules.agent.web_tools import (
    make_brave_search,
    make_firecrawl_extract,
    make_playwright_fetch,
    web_tool_availability,
    web_tool_config,
)
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


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
    web_config = web_tool_config(settings=settings)
    web_statuses = web_tool_availability(settings=settings)

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
        ("brave_search", make_brave_search(api_key=web_config.brave_api_key), "Broad web search with Brave Search API."),
        ("firecrawl_extract", make_firecrawl_extract(api_key=web_config.firecrawl_api_key), "Fetch full content from a target URL with Firecrawl API."),
        ("playwright_fetch", make_playwright_fetch(endpoint=web_config.agent_browser_endpoint), "Fetch dynamically rendered page content via browser runtime."),
    ):
        status = web_statuses[name]
        registrations.append(
            ResearchToolRegistration(
                agent_name=name,
                backend_name=status.backend,
                category="web",
                available=status.available,
                reason=status.reason,
                tool=(
                    build_traced_structured_tool(
                        name=name,
                        description=description,
                        server_id=status.backend,
                        runtime=tool_runtime,
                        handler=handler,
                    )
                    if (runtime is not None and status.available)
                    else None
                ),
            )
        )

    return registrations


def build_runtime_research_tools(*, settings: Settings, mcp_manager: MCPClientManager, runtime: MCPToolRuntime) -> list[StructuredTool]:
    registrations = resolve_research_tools(settings=settings, mcp_manager=mcp_manager, runtime=runtime)
    return [item.tool for item in registrations if item.tool is not None]
