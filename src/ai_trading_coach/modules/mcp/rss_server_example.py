"""Example lightweight MCP RSS server (optional local tool server).

Run (example):
    python3 -m ai_trading_coach.modules.mcp.rss_server_example
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


def _fetch_rss(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url, timeout=12) as resp:  # noqa: S310
        xml_text = resp.read().decode("utf-8", errors="ignore")

    root = ET.fromstring(xml_text)
    out: list[dict[str, str]] = []
    for item in root.findall(".//item")[:20]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if not title and not link:
            continue
        out.append(
            {
                "title": title,
                "uri": link,
                "published_at": pub_date,
                "summary": desc[:300],
            }
        )
    return out


def _rss_search(query: str, limit: int = 10) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    feed_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    rows = _fetch_rss(feed_url)
    return rows[: max(1, min(limit, 20))]


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install 'mcp' package to run rss_server_example.") from exc

    mcp = FastMCP("rss_search")

    @mcp.tool()
    def rss_search(query: str, limit: int = 10) -> str:
        """Search public RSS feeds without API key."""

        payload = {
            "query": query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "items": _rss_search(query=query, limit=limit),
        }
        return json.dumps(payload, ensure_ascii=False)

    mcp.run()


if __name__ == "__main__":
    main()
