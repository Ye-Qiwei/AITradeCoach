"""Minimal MCP client manager for MVP runtime."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from ai_trading_coach.config import MCPServerDefinition, Settings
from ai_trading_coach.errors import MCPConfigurationError

DEFAULT_CURATED_MCP_TOOLS: dict[str, str] = {
    "get_price_history": "yfinance:yfinance_get_price_history",
    "search_news": "yfinance:yfinance_get_ticker_news",
}


@dataclass(frozen=True)
class MCPToolRef:
    server_id: str
    tool_name: str

    @property
    def key(self) -> str:
        return f"{self.server_id}:{self.tool_name}"


class MCPClientManager:
    def __init__(self, *, settings: Settings, invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None) -> None:
        self.settings = settings
        self.invoker = invoker
        self.server_map = {server.server_id: server for server in settings.mcp_server_definitions()}

    def get_tool_ref(self, agent_tool_name: str) -> tuple[MCPToolRef | None, str | None]:
        raw = DEFAULT_CURATED_MCP_TOOLS.get(agent_tool_name)
        if not raw:
            return None, f"unknown agent tool: {agent_tool_name}"
        server_id, tool_name = raw.split(":", 1)
        if self.invoker is None and server_id not in self.server_map:
            return None, f"MCP server '{server_id}' missing in MCP_SERVERS"
        return MCPToolRef(server_id=server_id, tool_name=tool_name), None

    async def call_tool(self, *, server_id: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self.invoker is not None:
            outcome = self.invoker(server_id, tool_name, arguments)
            if inspect.isawaitable(outcome):
                return await outcome
            return outcome
        server = self.server_map.get(server_id)
        if server is None:
            raise MCPConfigurationError(f"MCP server '{server_id}' not found in MCP_SERVERS.")
        return await self._call_with_sdk(server=server, tool_name=tool_name, arguments=arguments)

    async def _call_with_sdk(self, *, server: MCPServerDefinition, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:  # noqa: BLE001
            raise MCPConfigurationError("Install package 'mcp' to enable MCP tool calls.") from exc

        timeout = float(server.timeout_seconds or self.settings.mcp_timeout_seconds)
        retries = max(1, int(self.settings.mcp_max_retries) + 1)
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                if server.transport == "stdio":
                    if not server.command:
                        raise MCPConfigurationError(f"Server '{server.server_id}' transport=stdio requires command.")
                    command, args = _resolve_stdio_command(server)
                    params = StdioServerParameters(command=command, args=args, env=server.env or None)
                    async with stdio_client(params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)
                if server.transport == "sse":
                    if not server.url:
                        raise MCPConfigurationError(f"Server '{server.server_id}' transport=sse requires url.")
                    async with sse_client(server.url) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)
                if not server.url:
                    raise MCPConfigurationError(f"Server '{server.server_id}' transport=http requires url.")
                async with streamablehttp_client(server.url) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        return await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise MCPConfigurationError(f"Tool call failed after retries for {server.server_id}:{tool_name}: {last_error}")

    def diagnostics(self) -> dict[str, Any]:
        return {"curated_tools": DEFAULT_CURATED_MCP_TOOLS, "configured_servers": sorted(self.server_map.keys())}


def tool_payload_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_stdio_command(server: MCPServerDefinition) -> tuple[str, list[str]]:
    command = (server.command or "").strip()
    args = list(server.args)
    if command != "uv" or shutil.which("uv") is not None:
        return command, args
    parsed = _parse_uv_run_server_args(args)
    if parsed is None:
        raise MCPConfigurationError("MCP server command is 'uv' but uv is not installed.")
    server_script, workdir = parsed
    script_path = server_script if workdir is None else str((workdir / server_script).resolve())
    python = shutil.which("python3") or shutil.which("python") or "python3"
    return python, [script_path]


def _parse_uv_run_server_args(args: list[str]) -> tuple[str, Path | None] | None:
    workdir: Path | None = None
    remaining = list(args)
    if len(remaining) >= 2 and remaining[0] == "--directory":
        workdir = Path(remaining[1])
        remaining = remaining[2:]
    if len(remaining) == 2 and remaining[0] == "run" and remaining[1].endswith('.py'):
        return remaining[1], workdir
    return None
