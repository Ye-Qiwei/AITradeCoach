from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModuleName


def test_module_model_selection_uses_override_then_default() -> None:
    settings = Settings(
        gemini_model="gemini-default",
        model_default="gemini-fallback",
        model_promptops="gemini-promptops",
        model_cognition_extraction="gemini-cognition",
    )

    assert settings.default_model() == "gemini-fallback"
    assert settings.model_for_module(ModuleName.PROMPTOPS) == "gemini-promptops"
    assert settings.model_for_module(ModuleName.COGNITION_ENGINE) == "gemini-cognition"
    assert settings.model_for_module(ModuleName.EVALUATOR) == "gemini-fallback"

