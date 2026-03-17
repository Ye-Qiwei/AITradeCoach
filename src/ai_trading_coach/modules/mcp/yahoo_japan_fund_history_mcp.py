"""Local MCP wrapper for Yahoo Japan fund history scraping.

This module remains as the stdio MCP bridge for environments that still prefer
MCP transport, while runtime defaults expose the same capability directly as a
local Python tool (`yahoo_japan_fund_history`).
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from ai_trading_coach.modules.data_sources.yahoo_japan_fund_history import get_fund_history_by_code, get_fund_history_by_url

mcp = FastMCP("yahoo_japan_fund_history")


@mcp.tool()
async def fund_history_from_url(url: str, max_pages: int = 3) -> str:
    payload = await get_fund_history_by_url(url=url, max_pages=max_pages)
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
async def fund_history_from_code(fund_code: str, max_pages: int = 3) -> str:
    payload = await get_fund_history_by_code(fund_code=fund_code, max_pages=max_pages)
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
