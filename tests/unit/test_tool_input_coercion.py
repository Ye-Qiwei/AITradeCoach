from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolInput
from ai_trading_coach.modules.agent.react_tools import ReactResearchTools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


class _Manager:
    def resolve_tool(self, _evidence_type):
        return type("ToolRef", (), {"server_id": "s", "tool_name": "t", "key": "s:t"})()

    async def call_tool(self, **_kwargs):
        return {"items": []}


def test_langchain_tool_input_coerces_string_query_to_dict() -> None:
    payload = MCPToolInput(query="macro risk off")

    assert payload.query == {"query": "macro risk off"}


def test_react_tool_accepts_string_query_arguments() -> None:
    manager = _Manager()
    tools = ReactResearchTools(mcp_manager=manager)

    result = tools.execute(
        tool_name="search_news",
        arguments={"objective": "find news", "query": "US Iran conflict market"},
        step_id="s1",
    )

    assert result.success is True


def test_uv_stdio_command_falls_back_to_python_when_uv_missing(monkeypatch) -> None:
    from ai_trading_coach.modules.mcp import mcp_client_manager as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: None if name == "uv" else "/usr/bin/python3")
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="test-key",
        mcp_servers_json='[{"server_id":"yfinance","transport":"stdio","command":"uv","args":["--directory","/tmp/yahoo-finance-mcp","run","server.py"]}]',
        mcp_tool_allowlist_csv="yfinance:get_historical_stock_prices",
        evidence_tool_map_json='{"price_path":"yfinance:get_historical_stock_prices"}',
    )
    manager = MCPClientManager(settings=settings)

    command, args = mod._resolve_stdio_command(manager.server_map["yfinance"])

    assert command == "/usr/bin/python3"
    assert args == ["/tmp/yahoo-finance-mcp/server.py"]
