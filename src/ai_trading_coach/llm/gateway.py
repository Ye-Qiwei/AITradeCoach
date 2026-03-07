"""Unified LangChain-based LLM gateway for structured and text calls."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace
from ai_trading_coach.llm.langchain_chat_model import build_langchain_chat_model

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LangChainLLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = build_langchain_chat_model(settings=settings, timeout_seconds=settings.llm_timeout_seconds)

    def invoke_structured(
        self,
        *,
        schema: type[SchemaT],
        messages: list[dict[str, str]],
        purpose: ModelCallPurpose,
        prompt_version: str,
        input_summary: str,
        output_summary_builder: Callable[[SchemaT], str] | None = None,
    ) -> tuple[SchemaT, ModelCallTrace]:
        started_at = utc_now()
        t0 = time.perf_counter()
        structured_model = self.model.with_structured_output(schema)
        raw = structured_model.invoke(messages)
        result = raw if isinstance(raw, schema) else schema.model_validate(raw)
        ended_at = utc_now()
        latency = int((time.perf_counter() - t0) * 1000)
        output_summary = output_summary_builder(result) if output_summary_builder else schema.__name__
        trace = ModelCallTrace(
            purpose=purpose,
            provider_name=self.settings.llm_provider(),
            model_name=self.settings.selected_llm_model(),
            prompt_version=prompt_version,
            input_summary=input_summary,
            output_summary=output_summary,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency,
        )
        return result, trace

    def invoke_text(
        self,
        *,
        messages: list[dict[str, str]],
        purpose: ModelCallPurpose,
        prompt_version: str,
        input_summary: str,
    ) -> tuple[str, ModelCallTrace]:
        started_at = utc_now()
        t0 = time.perf_counter()
        response = self.model.invoke(messages)
        ended_at = utc_now()
        content = response.content if isinstance(response.content, str) else str(response.content)
        trace = ModelCallTrace(
            purpose=purpose,
            provider_name=self.settings.llm_provider(),
            model_name=self.settings.selected_llm_model(),
            prompt_version=prompt_version,
            input_summary=input_summary,
            output_summary=f"chars={len(content)}",
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return content, trace


__all__ = ["LangChainLLMGateway"]
