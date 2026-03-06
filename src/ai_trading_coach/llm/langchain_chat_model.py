"""Factory for provider-specific LangChain chat models."""

from __future__ import annotations

from typing import Any

from ai_trading_coach.config import Settings


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

    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        timeout=timeout,
        temperature=0,
    )


__all__ = ["build_langchain_chat_model"]
