"""LLM provider registry based on runtime settings."""

from __future__ import annotations

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gemini_provider import GeminiLLMProvider
from ai_trading_coach.llm.openai_provider import OpenAILLMProvider
from ai_trading_coach.llm.provider import LLMProvider


def build_required_llm_provider(
    settings: Settings,
    model_name: str | None = None,
    timeout_seconds: float | None = None,
) -> LLMProvider:
    """Build a configured provider and raise on invalid config."""

    provider_name = settings.llm_provider()
    api_key = settings.llm_api_key()
    timeout = float(timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds)
    selected_model = model_name or settings.selected_llm_model()

    if provider_name == "gemini":
        return GeminiLLMProvider(
            model_name=selected_model,
            api_key=api_key,
            timeout_seconds=timeout,
        )

    return OpenAILLMProvider(
        model_name=selected_model,
        api_key=api_key,
        timeout_seconds=timeout,
    )


def build_llm_provider(
    settings: Settings,
    model_name: str | None = None,
    timeout_seconds: float | None = None,
) -> LLMProvider:
    """Backward-compatible alias to mandatory provider builder."""

    return build_required_llm_provider(
        settings=settings,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )


__all__ = ["build_llm_provider", "build_required_llm_provider"]
