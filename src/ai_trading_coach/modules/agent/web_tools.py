"""High-coverage web research tools for diverse judgement verification."""

from __future__ import annotations

import json
import os
from urllib import parse, request

from langchain_core.tools import StructuredTool

from ai_trading_coach.config import Settings


def build_general_web_tools(*, settings: Settings) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    status = settings.web_tool_status()
    brave_api_key = settings.brave_api_key.strip()
    firecrawl_api_key = settings.firecrawl_api_key.strip()
    agent_browser_endpoint = settings.agent_browser_endpoint.strip()

    if status["brave_search"]:
        def _brave_search(query: str, count: int = 5) -> str:
            return _brave_search_impl(query=query, count=count, api_key=brave_api_key)

        tools.append(
            StructuredTool.from_function(
                func=_brave_search,
                name="brave_search",
                description=(
                    "Broad web search with Brave Search API. "
                    "Use for clue discovery and URL candidate collection."
                ),
            )
        )
    if status["firecrawl_extract"]:
        def _firecrawl_extract(url: str) -> str:
            return _firecrawl_extract_impl(url=url, api_key=firecrawl_api_key)

        tools.append(
            StructuredTool.from_function(
                func=_firecrawl_extract,
                name="firecrawl_extract",
                description="Fetch full content from a target URL with Firecrawl API.",
            )
        )
    if status["playwright_fetch"]:
        def _playwright_fetch(url: str, instruction: str = "extract main content") -> str:
            return _playwright_fetch_impl(
                url=url,
                instruction=instruction,
                endpoint=agent_browser_endpoint,
            )

        tools.append(
            StructuredTool.from_function(
                func=_playwright_fetch,
                name="playwright_fetch",
                description="Fetch dynamically rendered page content via browser MCP/agent-browser bridge.",
            )
        )
    return tools


def brave_search(query: str, count: int = 5) -> str:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    return _brave_search_impl(query=query, count=count, api_key=api_key)


def _brave_search_impl(*, query: str, count: int, api_key: str) -> str:
    if not api_key:
        return "tool_error: BRAVE_API_KEY is missing"
    params = parse.urlencode({"q": query, "count": max(1, min(count, 10))})
    req = request.Request(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        method="GET",
    )
    with request.urlopen(req, timeout=15) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    results = data.get("web", {}).get("results", [])
    compact = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        for item in results
    ]
    return json.dumps({"query": query, "results": compact}, ensure_ascii=False)


def firecrawl_extract(url: str) -> str:
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    return _firecrawl_extract_impl(url=url, api_key=api_key)


def _firecrawl_extract_impl(*, url: str, api_key: str) -> str:
    if not api_key:
        return "tool_error: FIRECRAWL_API_KEY is missing"
    req = request.Request(
        "https://api.firecrawl.dev/v1/scrape",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        data=json.dumps({"url": url, "formats": ["markdown"]}).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    payload = data.get("data", {})
    return json.dumps(
        {
            "url": url,
            "title": payload.get("metadata", {}).get("title", ""),
            "markdown": payload.get("markdown", "")[:6000],
        },
        ensure_ascii=False,
    )


def playwright_fetch(url: str, instruction: str = "extract main content") -> str:
    endpoint = os.getenv("AGENT_BROWSER_ENDPOINT", "").strip()
    return _playwright_fetch_impl(url=url, instruction=instruction, endpoint=endpoint)


def _playwright_fetch_impl(*, url: str, instruction: str, endpoint: str) -> str:
    if not endpoint:
        return "tool_error: AGENT_BROWSER_ENDPOINT is missing"
    req = request.Request(
        endpoint,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"url": url, "instruction": instruction}).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=25) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    return json.dumps(data, ensure_ascii=False)[:8000]
