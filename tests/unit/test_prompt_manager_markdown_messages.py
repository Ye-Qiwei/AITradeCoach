from ai_trading_coach.modules.agent.prompting import PromptManager


def test_build_messages_renders_markdown_context() -> None:
    messages = PromptManager.build_messages(
        system_prompt="system",
        context={"task": "demo", "items": [{"name": "AAPL", "view": "bullish"}]},
    )
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("## Task Context")
    assert '"task"' not in messages[1]["content"]
    assert "| name | view |" in messages[1]["content"]
