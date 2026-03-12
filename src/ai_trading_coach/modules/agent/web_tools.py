"""Web research tool implementations and availability probes."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from urllib import error, parse, request

from ai_trading_coach.config import Settings


@dataclass(frozen=True)
class WebToolAvailability:
    name: str
    backend: str
    available: bool
    reason: str | None = None


def web_tool_availability(*, settings: Settings) -> dict[str, WebToolAvailability]:
    brave_key = settings.brave_api_key.strip()
    firecrawl_key = settings.firecrawl_api_key.strip()
    endpoint = settings.agent_browser_endpoint.strip()

    brave = WebToolAvailability(
        name="brave_search",
        backend="web:brave_search_api",
        available=bool(brave_key),
        reason=None if brave_key else "BRAVE_API_KEY is missing",
    )
    firecrawl = WebToolAvailability(
        name="firecrawl_extract",
        backend="web:firecrawl_scrape_api",
        available=bool(firecrawl_key),
        reason=None if firecrawl_key else "FIRECRAWL_API_KEY is missing",
    )

    if endpoint:
        reachable, reason = _probe_http_endpoint(endpoint)
        playwright = WebToolAvailability(
            name="playwright_fetch",
            backend="browser:http_bridge",
            available=reachable,
            reason=reason,
        )
    else:
        local_ok, reason = _probe_local_playwright_runtime()
        playwright = WebToolAvailability(
            name="playwright_fetch",
            backend="browser:local_playwright",
            available=local_ok,
            reason=reason,
        )

    return {item.name: item for item in (brave, firecrawl, playwright)}


def brave_search(query: str, count: int = 5) -> str:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    return _brave_search_impl(query=query, count=count, api_key=api_key)


def firecrawl_extract(url: str) -> str:
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    return _firecrawl_extract_impl(url=url, api_key=api_key)


def playwright_fetch(url: str, instruction: str = "extract main content") -> str:
    endpoint = os.getenv("AGENT_BROWSER_ENDPOINT", "").strip()
    if endpoint:
        return _playwright_fetch_http(url=url, instruction=instruction, endpoint=endpoint)
    return asyncio.run(_playwright_fetch_local(url=url, instruction=instruction))


def _probe_http_endpoint(endpoint: str) -> tuple[bool, str | None]:
    req = request.Request(endpoint, method="GET")
    try:
        with request.urlopen(req, timeout=2):  # noqa: S310
            return True, None
    except error.HTTPError:
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"AGENT_BROWSER_ENDPOINT unreachable: {exc}"


def _probe_local_playwright_runtime() -> tuple[bool, str | None]:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"Local Playwright runtime unavailable: {exc}"


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
        {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("description", "")}
        for item in results
    ]
    return json.dumps({"query": query, "results": compact}, ensure_ascii=False)


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
    return json.dumps({"url": url, "title": payload.get("metadata", {}).get("title", ""), "markdown": payload.get("markdown", "")[:6000]}, ensure_ascii=False)


def _playwright_fetch_http(*, url: str, instruction: str, endpoint: str) -> str:
    req = request.Request(
        endpoint,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"url": url, "instruction": instruction}).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=25) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    return json.dumps(data, ensure_ascii=False)[:8000]


async def _playwright_fetch_local(*, url: str, instruction: str) -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)
            title = await page.title()
            body = await page.locator("body").inner_text()
            return json.dumps({"url": url, "instruction": instruction, "title": title, "content": body[:7000]}, ensure_ascii=False)
        finally:
            await browser.close()
