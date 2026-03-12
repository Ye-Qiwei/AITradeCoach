"""MCP client manager using official MCP Python SDK when available."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from ai_trading_coach.modules.agent.curated_tools import enabled_curated_tools
from pathlib import Path
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


@dataclass(frozen=True)
class RawMCPToolMetadata:
    server_id: str
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    available: bool = True

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

    def tool_configuration_status(
        self,
        evidence_type: EvidenceType,
    ) -> tuple[MCPToolRef | None, str | None]:
        ref = self.evidence_map.get(evidence_type.value)
        if ref is None:
            return None, f"No evidence->tool mapping for '{evidence_type.value}'."
        if ref.key not in self.allowlist:
            return None, f"Blocked by MCP_TOOL_ALLOWLIST: {ref.key}"
        if self.invoker is None and ref.server_id not in self.server_map:
            return None, f"Server '{ref.server_id}' is missing from MCP_SERVERS."
        return ref, None

    def prepare_tool_arguments(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return _prepare_tool_arguments(
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments,
        )

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
        prepared_arguments = self.prepare_tool_arguments(
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        return await self._call_with_sdk(
            server=server,
            tool_name=tool_name,
            arguments=prepared_arguments,
        )

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
                    command, args = _resolve_stdio_command(server)
                    params = StdioServerParameters(
                        command=command,
                        args=args,
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


    def curated_tool_mapping(self) -> dict[str, MCPToolRef]:
        mapping: dict[str, MCPToolRef] = {}
        for tool in enabled_curated_tools():
            if tool.implementation_kind != "external_mcp":
                continue
            if tool.evidence_type is None:
                continue
            ref, _ = self.tool_configuration_status(tool.evidence_type)
            if ref is not None:
                mapping[tool.canonical_name] = ref
        return mapping

    def curated_tool_status(self, canonical_name: str) -> tuple[MCPToolRef | None, str | None]:
        for tool in enabled_curated_tools():
            if tool.canonical_name != canonical_name:
                continue
            if tool.implementation_kind != "external_mcp":
                return None, "not external_mcp"
            if tool.evidence_type is None:
                return None, "missing evidence_type"
            return self.tool_configuration_status(tool.evidence_type)
        return None, f"unknown curated tool: {canonical_name}"

    def discovered_tools(self) -> list[RawMCPToolMetadata]:
        tools: list[RawMCPToolMetadata] = []
        for key in sorted(self.allowlist):
            if ":" not in key:
                continue
            server_id, tool_name = key.split(":", 1)
            tools.append(RawMCPToolMetadata(server_id=server_id, tool_name=tool_name, available=(self.invoker is not None or server_id in self.server_map)))
        return tools

    def diagnostics(self) -> dict[str, Any]:
        curated = self.curated_tool_mapping()
        discovered = self.discovered_tools()
        exposed_raw_keys = {ref.key for ref in curated.values()}
        hidden = [item.key for item in discovered if item.key not in exposed_raw_keys]
        return {
            "curated_tools": sorted(curated.keys()),
            "curated_to_raw": {name: ref.key for name, ref in curated.items()},
            "raw_discovered_tools": [item.key for item in discovered],
            "raw_not_exposed_to_agent": hidden,
        }



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
        raise MCPConfigurationError(
            "MCP server command is 'uv' but uv is not installed and args cannot be translated. "
            "Use python to launch the server directly or install uv."
        )

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


def _prepare_tool_arguments(
    *,
    server_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    builders: dict[
        tuple[str | None, str],
        Callable[[dict[str, Any]], dict[str, Any]],
    ] = {
        ("yfinance", "yfinance_get_price_history"): _build_yfinance_price_history_arguments,
        ("yfinance", "yfinance_get_ticker_news"): _build_yfinance_news_arguments,
        ("rss_search", "rss_search"): _build_rss_search_arguments,
    }
    builder = builders.get((server_id, tool_name)) or builders.get((None, tool_name))
    if builder is None:
        return arguments
    return builder(arguments)


def _build_yfinance_price_history_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    ticker = _first_ticker(arguments)
    if not ticker:
        raise MCPConfigurationError(
            "yfinance:yfinance_get_price_history requires at least one ticker."
        )

    query = _query_dict(arguments)
    period = _string_query_value(query, "period") or _time_window_to_yfinance_period(
        arguments.get("time_window")
    )
    interval = _string_query_value(query, "interval") or "1d"
    return {
        "ticker": ticker,
        "period": period or "1mo",
        "interval": interval,
    }


def _build_yfinance_news_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    ticker = _first_ticker(arguments)
    if not ticker:
        raise MCPConfigurationError(
            "yfinance:yfinance_get_ticker_news requires at least one ticker."
        )
    return {"ticker": ticker}


def _build_rss_search_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    query = _query_dict(arguments)
    text = (
        _string_query_value(query, "query")
        or _string_query_value(query, "q")
        or _build_search_query(arguments)
    )
    if not text:
        raise MCPConfigurationError("rss_search:rss_search requires a non-empty query.")
    limit = _int_query_value(query, "limit") or 10
    return {
        "query": text,
        "limit": max(1, min(limit, 20)),
    }


def _query_dict(arguments: dict[str, Any]) -> dict[str, Any]:
    value = arguments.get("query")
    return value if isinstance(value, dict) else {}


def _string_query_value(query: dict[str, Any], key: str) -> str:
    value = query.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int_query_value(query: dict[str, Any], key: str) -> int | None:
    value = query.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _first_ticker(arguments: dict[str, Any]) -> str:
    query = _query_dict(arguments)
    query_ticker = (
        _string_query_value(query, "ticker")
        or _string_query_value(query, "symbol")
        or _string_query_value(query, "asset")
    )
    if query_ticker:
        return query_ticker.upper()

    query_tickers = query.get("tickers")
    if isinstance(query_tickers, list):
        for item in query_tickers:
            if isinstance(item, str) and item.strip():
                return item.strip().upper()

    tickers = arguments.get("tickers")
    if isinstance(tickers, list):
        for item in tickers:
            if isinstance(item, str) and item.strip():
                return item.strip().upper()

    candidates = [
        _extract_ticker_from_text(str(arguments.get("objective", ""))),
        _extract_ticker_from_text(_build_search_query(arguments)),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _extract_ticker_from_text(text: str) -> str:
    if not text:
        return ""
    # Accept common ticker forms: TSLA, BRK.B, 0700.HK, $AAPL.
    pattern = re.compile(r"\$?[A-Za-z0-9]{1,6}(?:[.-][A-Za-z0-9]{1,4})?")
    for match in pattern.finditer(text):
        raw = match.group(0)
        candidate = raw.lstrip("$").upper()
        if _looks_like_ticker(candidate, raw):
            return candidate
    return ""


def _looks_like_ticker(value: str, raw: str) -> bool:
    if not value:
        return False
    has_marker = raw.startswith("$") or any(ch.isdigit() for ch in raw) or "." in raw or "-" in raw
    if has_marker:
        return True
    if raw.isupper() and raw.isalpha() and 1 <= len(raw) <= 5:
        return True
    return False


def _build_search_query(arguments: dict[str, Any]) -> str:
    objective = str(arguments.get("objective", "")).strip()
    tickers = [
        item.strip().upper()
        for item in arguments.get("tickers", [])
        if isinstance(item, str) and item.strip()
    ]
    if objective and tickers:
        suffix = " ".join(tickers)
        return objective if suffix in objective else f"{objective} {suffix}"
    if objective:
        return objective
    return " ".join(tickers)


def _time_window_to_yfinance_period(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.lower().replace("-", " ").replace("_", " ").split())
    mapping = {
        "intraday": "1d",
        "1d": "1d",
        "1 day": "1d",
        "5d": "5d",
        "5 days": "5d",
        "1w": "5d",
        "1 week": "5d",
        "2w": "1mo",
        "2 weeks": "1mo",
        "1mo": "1mo",
        "1 month": "1mo",
        "3mo": "3mo",
        "3 months": "3mo",
        "6mo": "6mo",
        "6 months": "6mo",
        "1y": "1y",
        "1 year": "1y",
        "ytd": "ytd",
        "max": "max",
    }
    return mapping.get(normalized, "")
