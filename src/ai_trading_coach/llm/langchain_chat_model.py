"""Factory for provider-specific LangChain chat models."""

from __future__ import annotations

from typing import Any

from ai_trading_coach.config import Settings


def _openai_supports_temperature(model_name: str) -> bool:
    return not model_name.strip().lower().startswith("gpt-5")


def _openai_chat_kwargs(*, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
    }
    if _openai_supports_temperature(model):
        kwargs["temperature"] = 0
    return kwargs


def build_langchain_chat_model(settings: Settings, *, timeout_seconds: float | None = None) -> Any:
    provider = settings.llm_provider()
    timeout = float(timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds)
    model = settings.selected_llm_model()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.gemini_api_key,
            timeout=timeout,
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**_openai_chat_kwargs(model=model, api_key=settings.openai_api_key, timeout=timeout))


__all__ = ["build_langchain_chat_model"]
