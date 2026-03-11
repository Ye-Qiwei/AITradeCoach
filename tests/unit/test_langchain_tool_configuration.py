from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.langchain_tools import MCPToolRuntime, build_langchain_mcp_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_build_langchain_mcp_tools_skips_unconfigured_mcp_and_web_tools() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="test-key",
        mcp_servers_json="[]",
    )

    manager = MCPClientManager(settings=settings)
    tools = build_langchain_mcp_tools(mcp_manager=manager, runtime=MCPToolRuntime())

    assert tools == []


def test_build_langchain_mcp_tools_only_includes_configured_tools() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider_name="openai",
        openai_api_key="test-key",
        brave_api_key="brave-key",
        mcp_servers_json="[]",
        mcp_tool_allowlist_csv="mock:price_history",
        evidence_tool_map_json='{"price_path":"mock:price_history"}',
    )

    manager = MCPClientManager(settings=settings, invoker=lambda *_: {"items": []})
    tools = build_langchain_mcp_tools(mcp_manager=manager, runtime=MCPToolRuntime())

    assert [tool.name for tool in tools] == ["get_price_history", "brave_search"]
