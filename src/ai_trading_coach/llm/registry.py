"""LLM provider registry based on runtime settings."""

from __future__ import annotations

import logging

from ai_trading_coach.config import Settings
from ai_trading_coach.llm.gemini_provider import GeminiLLMProvider
from ai_trading_coach.llm.openai_provider import OpenAILLMProvider
from ai_trading_coach.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


def build_llm_provider(
    settings: Settings,
    model_name: str | None = None,
    timeout_seconds: float | None = None,
) -> LLMProvider | None:
    """Build a configured provider; return None when unavailable."""

    provider_name = settings.atc_llm_provider.strip().lower()
    timeout = float(timeout_seconds if timeout_seconds is not None else settings.atc_llm_timeout_seconds)

    if provider_name == "gemini":
        if not settings.gemini_api_key:
            logger.warning("llm_provider_unavailable provider=gemini reason=missing_api_key")
            return None
        selected_model = model_name or settings.atc_llm_model or settings.gemini_model
        return GeminiLLMProvider(
            model_name=selected_model,
            api_key=settings.gemini_api_key,
            timeout_seconds=timeout,
        )

    if provider_name == "openai":
        if not settings.openai_api_key:
            logger.warning("llm_provider_unavailable provider=openai reason=missing_api_key")
            return None
        selected_model = model_name or settings.atc_llm_model or "gpt-4o-mini"
        return OpenAILLMProvider(
            model_name=selected_model,
            api_key=settings.openai_api_key,
            timeout_seconds=timeout,
        )

    logger.warning("llm_provider_unavailable provider=%s reason=unsupported", provider_name)
    return None


__all__ = ["build_llm_provider"]
