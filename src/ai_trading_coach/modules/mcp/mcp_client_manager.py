"""MCP client manager using official MCP Python SDK when available."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ai_trading_coach.config import MCPServerDefinition, Settings
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.errors import MCPConfigurationError, MCPToolNotAllowedError


@dataclass(frozen=True)
class MCPToolRef:
    server_id: str
    tool_name: str

    @property
    def key(self) -> str:
        return f"{self.server_id}:{self.tool_name}"


class MCPClientManager:
    """Manage multi-server MCP calls with allowlist enforcement."""

    def __init__(
        self,
        *,
        settings: Settings,
        invoker: Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.invoker = invoker
        self.server_map = {server.server_id: server for server in settings.mcp_server_definitions()}
        self.allowlist = settings.mcp_tool_allowlist()
        self.evidence_map = self._parse_evidence_tool_map(settings.evidence_tool_map())

    def resolve_tool(self, evidence_type: EvidenceType) -> MCPToolRef:
        ref = self.evidence_map.get(evidence_type.value)
        if ref is None:
            raise MCPConfigurationError(
                f"No evidence->tool mapping found for evidence type '{evidence_type.value}'."
            )
        return ref

    async def call_tool(self, *, server_id: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        key = f"{server_id}:{tool_name}"
        if key not in self.allowlist:
            raise MCPToolNotAllowedError(
                f"MCP tool blocked by allowlist: {key}. Configure MCP_TOOL_ALLOWLIST."
            )

        if self.invoker is not None:
            outcome = self.invoker(server_id, tool_name, arguments)
            if inspect.isawaitable(outcome):
                return await outcome
            return outcome

        server = self.server_map.get(server_id)
        if server is None:
            raise MCPConfigurationError(
                f"MCP server '{server_id}' not found in MCP_SERVERS."
            )
        return await self._call_with_sdk(server=server, tool_name=tool_name, arguments=arguments)

    async def _call_with_sdk(
        self,
        *,
        server: MCPServerDefinition,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:  # noqa: BLE001
            raise MCPConfigurationError(
                "Official MCP Python SDK is required. Install package 'mcp' to enable MCP tool calls."
            ) from exc

        timeout = float(server.timeout_seconds or self.settings.mcp_timeout_seconds)
        retries = max(1, int(self.settings.mcp_max_retries) + 1)
        last_error: Exception | None = None

        for _ in range(retries):
            try:
                if server.transport == "stdio":
                    if not server.command:
                        raise MCPConfigurationError(
                            f"Server '{server.server_id}' transport=stdio requires command."
                        )
                    params = StdioServerParameters(
                        command=server.command,
                        args=server.args,
                        env=server.env or None,
                    )
                    async with stdio_client(params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await asyncio.wait_for(
                                session.call_tool(tool_name, arguments),
                                timeout=timeout,
                            )

                if server.transport == "sse":
                    if not server.url:
                        raise MCPConfigurationError(
                            f"Server '{server.server_id}' transport=sse requires url."
                        )
                    async with sse_client(server.url) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await asyncio.wait_for(
                                session.call_tool(tool_name, arguments),
                                timeout=timeout,
                            )

                if not server.url:
                    raise MCPConfigurationError(
                        f"Server '{server.server_id}' transport=http requires url."
                    )
                async with streamablehttp_client(server.url) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        return await asyncio.wait_for(
                            session.call_tool(tool_name, arguments),
                            timeout=timeout,
                        )
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise MCPConfigurationError(
            f"Tool call failed after retries for {server.server_id}:{tool_name}: {last_error}"
        )

    def _parse_evidence_tool_map(self, payload: dict[str, str]) -> dict[str, MCPToolRef]:
        out: dict[str, MCPToolRef] = {}
        for evidence_type, value in payload.items():
            server_id, tool_name = self._split_tool_ref(value)
            out[evidence_type] = MCPToolRef(server_id=server_id, tool_name=tool_name)
        return out

    def _split_tool_ref(self, value: str) -> tuple[str, str]:
        text = value.strip()
        if ":" not in text:
            raise MCPConfigurationError(
                f"Tool mapping '{text}' must use '<server_id>:<tool_name>' format."
            )
        server_id, tool_name = text.split(":", 1)
        if not server_id or not tool_name:
            raise MCPConfigurationError(
                f"Tool mapping '{text}' must use '<server_id>:<tool_name>' format."
            )
        return server_id.strip(), tool_name.strip()


def tool_payload_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

