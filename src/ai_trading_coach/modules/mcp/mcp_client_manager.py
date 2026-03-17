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

@dataclass(frozen=True)
class MCPToolDefinition:
    server_id: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


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
        self._tool_catalog: dict[str, dict[str, MCPToolDefinition]] | None = None

    def list_server_tools(self, server_id: str) -> list[MCPToolDefinition]:
        catalog = self._load_tool_catalog()
        return list(catalog.get(server_id, {}).values())

    def get_tool_ref(self, agent_tool_name: str) -> tuple[MCPToolRef | None, str | None]:
        server_id = "yfinance"
        if self.invoker is None and server_id not in self.server_map:
            return None, f"MCP server '{server_id}' missing in MCP_SERVERS"
        tools = self._load_tool_catalog().get(server_id, {})
        tool = tools.get(agent_tool_name)
        if tool is None:
            return None, f"unknown agent tool: {agent_tool_name}"
        return MCPToolRef(server_id=server_id, tool_name=tool.tool_name), None

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

    def _load_tool_catalog(self) -> dict[str, dict[str, MCPToolDefinition]]:
        if self._tool_catalog is not None:
            return self._tool_catalog
        catalog: dict[str, dict[str, MCPToolDefinition]] = {}
        for server_id, server in self.server_map.items():
            catalog[server_id] = {item.tool_name: item for item in asyncio.run(self._list_tools_with_sdk(server))}
        self._tool_catalog = catalog
        return catalog

    async def _list_tools_with_sdk(self, server: MCPServerDefinition) -> list[MCPToolDefinition]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:  # noqa: BLE001
            raise MCPConfigurationError("Install package 'mcp' to enable MCP tool calls.") from exc

        timeout = float(server.timeout_seconds or self.settings.mcp_timeout_seconds)
        if server.transport == "stdio":
            if not server.command:
                raise MCPConfigurationError(f"Server '{server.server_id}' transport=stdio requires command.")
            command, args = _resolve_stdio_command(server)
            params = StdioServerParameters(command=command, args=args, env=server.env or None)
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                    return [_tool_definition_from_sdk(server.server_id, item) for item in listed.tools]
        if server.transport == "sse":
            if not server.url:
                raise MCPConfigurationError(f"Server '{server.server_id}' transport=sse requires url.")
            async with sse_client(server.url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                    return [_tool_definition_from_sdk(server.server_id, item) for item in listed.tools]
        if not server.url:
            raise MCPConfigurationError(f"Server '{server.server_id}' transport=http requires url.")
        async with streamablehttp_client(server.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                return [_tool_definition_from_sdk(server.server_id, item) for item in listed.tools]

    def diagnostics(self) -> dict[str, Any]:
        return {"configured_servers": sorted(self.server_map.keys()), "tool_catalog": {sid: sorted(tools.keys()) for sid, tools in self._load_tool_catalog().items()}}


def _tool_definition_from_sdk(server_id: str, tool: Any) -> MCPToolDefinition:
    input_schema = getattr(tool, "inputSchema", None)
    if not isinstance(input_schema, dict):
        input_schema = {}
    return MCPToolDefinition(
        server_id=server_id,
        tool_name=str(getattr(tool, "name", "")),
        description=str(getattr(tool, "description", "")),
        input_schema=input_schema,
    )


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
