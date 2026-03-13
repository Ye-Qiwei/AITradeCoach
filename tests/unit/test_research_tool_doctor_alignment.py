from ai_trading_coach.app.run_manual import _environment_report
from ai_trading_coach.config import get_settings


def test_doctor_lists_backend_and_skipped_reason(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MCP_SERVERS", "[]")
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")
    monkeypatch.setenv("AGENT_BROWSER_ENDPOINT", "")
    get_settings.cache_clear()

    report = _environment_report()
    assert report["tools"]
    sample = report["tools"][0]
    assert "agent_name" in sample
    assert "backend" in sample
    assert "backend_kind" in sample
    assert "capability_group" in sample
    assert "available" in sample
    assert report["skipped_tools"]
