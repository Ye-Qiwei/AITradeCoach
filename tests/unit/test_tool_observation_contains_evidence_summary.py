from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.modules.agent.research_tools import resolve_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_browser_tool_not_enabled_when_unavailable(monkeypatch) -> None:
    from ai_trading_coach.modules.agent import web_tools as mod

    monkeypatch.setattr(mod, "_probe_local_playwright_runtime", lambda: (False, "missing runtime"))
    settings = Settings(llm_provider_name="openai", openai_api_key="x")
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})

    tools = resolve_research_tools(settings=settings, mcp_manager=manager)
    browser = next(item for item in tools if item.agent_name == "playwright_fetch")
    assert browser.available is False
    assert browser.reason == "missing runtime"


def test_browser_tool_enabled_when_endpoint_reachable(monkeypatch) -> None:
    from ai_trading_coach.modules.agent import web_tools as mod

    monkeypatch.setattr(mod, "_probe_http_endpoint", lambda _endpoint: (True, None))
    settings = Settings(
        llm_provider_name="openai",
        openai_api_key="x",
        agent_browser_endpoint="http://localhost:3000/fetch",
    )
    manager = MCPClientManager(settings=settings, invoker=lambda *_: {})

    tools = resolve_research_tools(settings=settings, mcp_manager=manager)
    browser = next(item for item in tools if item.agent_name == "playwright_fetch")
    assert browser.available is True
    assert browser.backend_name == "browser:http_bridge"
