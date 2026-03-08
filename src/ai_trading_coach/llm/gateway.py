"""Unified LangChain-based LLM gateway for structured and text calls."""

from __future__ import annotations

import time
import warnings
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace
from ai_trading_coach.domain.schema_validation import validate_strict_llm_schema
from ai_trading_coach.llm.langchain_chat_model import build_langchain_chat_model

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@lru_cache(maxsize=64)
def _validate_schema_cached(schema: type[BaseModel]) -> None:
    validate_strict_llm_schema(schema)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LangChainLLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = build_langchain_chat_model(settings=settings, timeout_seconds=settings.llm_timeout_seconds)

    def _build_model_call_trace(
        self,
        *,
        purpose: ModelCallPurpose,
        started_at: datetime,
        ended_at: datetime,
        prompt_version: str,
        input_summary: str,
        output_summary: str,
        latency_ms: int,
        error_message: str | None = None,
        token_in: int | None = None,
        token_out: int | None = None,
        response_size: int | None = None,
    ) -> ModelCallTrace:
        return ModelCallTrace(
            call_id=f"model_{purpose.value}_{int(started_at.timestamp() * 1000)}",
            purpose=purpose,
            provider=self.settings.llm_provider(),
            model_name=self.settings.selected_llm_model(),
            prompt_version=prompt_version,
            input_summary=input_summary,
            output_summary=output_summary,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            error_message=error_message,
            token_in=token_in,
            token_out=token_out,
            response_size=response_size,
        )

    @staticmethod
    def _is_ignorable_structured_warning(message: str) -> bool:
        return "Pydantic serializer warnings" in message and "field_name='parsed'" in message

    def _invoke_structured_model_with_warning_filter(self, structured_model: Any, messages: list[dict[str, str]]) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            response = structured_model.invoke(messages)
        for captured in caught:
            if self._is_ignorable_structured_warning(str(captured.message)):
                continue
            warnings.showwarning(
                message=captured.message,
                category=captured.category,
                filename=captured.filename,
                lineno=captured.lineno,
                file=captured.file,
                line=captured.line,
            )
        return response

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
        _validate_schema_cached(schema)
        started_at = utc_now()
        t0 = time.perf_counter()
        try:
            structured_model = self.model.with_structured_output(schema)
            raw = self._invoke_structured_model_with_warning_filter(structured_model, messages)
            result = raw if isinstance(raw, schema) else schema.model_validate(raw)
            output_summary = output_summary_builder(result) if output_summary_builder else schema.__name__
            ended_at = utc_now()
            latency = int((time.perf_counter() - t0) * 1000)
            trace = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at,
                ended_at=ended_at,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=output_summary,
                latency_ms=latency,
            )
            return result, trace
        except Exception as exc:  # noqa: BLE001
            ended_at = utc_now()
            latency = int((time.perf_counter() - t0) * 1000)
            _ = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at,
                ended_at=ended_at,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=f"error:{exc.__class__.__name__}",
                latency_ms=latency,
                error_message=str(exc),
            )
            raise RuntimeError(
                f"Structured output failed for schema={schema.__name__}, "
                f"purpose={purpose.value}, prompt_version={prompt_version}"
            ) from exc

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
        try:
            response = self.model.invoke(messages)
            ended_at = utc_now()
            content = response.content if isinstance(response.content, str) else str(response.content)
            trace = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at,
                ended_at=ended_at,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=f"chars={len(content)}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            return content, trace
        except Exception as exc:  # noqa: BLE001
            ended_at = utc_now()
            latency = int((time.perf_counter() - t0) * 1000)
            _ = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at,
                ended_at=ended_at,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=f"error:{exc.__class__.__name__}",
                latency_ms=latency,
                error_message=str(exc),
            )
            raise RuntimeError(
                f"Text output failed for purpose={purpose.value}, prompt_version={prompt_version}"
            ) from exc


__all__ = ["LangChainLLMGateway"]
