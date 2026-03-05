from __future__ import annotations

import json

from ai_trading_coach.prompts.registry import PromptRegistry


def test_prompt_registry_lists_versions_and_resolves_active(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "report_generation.v1.md").write_text("v1", encoding="utf-8")
    (prompts_dir / "report_generation.v2.md").write_text("v2", encoding="utf-8")
    (prompts_dir / "window_selection.v1.md").write_text("w1", encoding="utf-8")
    (prompts_dir / "manifest.json").write_text(
        json.dumps({"active_versions": {"report_generation": "v1"}}),
        encoding="utf-8",
    )

    registry = PromptRegistry(str(prompts_dir))
    assert registry.list_prompt_names() == ["report_generation", "window_selection"]
    assert registry.list_versions("report_generation") == ["v1", "v2"]
    assert registry.get_active_version("report_generation") == "v1"

    version, content = registry.load_active("report_generation")
    assert version == "v1"
    assert content == "v1"

    registry.set_active_version("report_generation", "v2")
    assert registry.get_active_version("report_generation") == "v2"

