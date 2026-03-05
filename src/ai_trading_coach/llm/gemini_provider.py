"""Gemini-backed LLM provider."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ai_trading_coach.llm.provider import LLMCallRecord, parse_json_object

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GeminiLLMProvider:
    """LLM provider implementation using langchain-google-genai."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.provider_name = "gemini"
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.last_call: LLMCallRecord | None = None

    def chat_json(
        self,
        schema_name: str,
        messages: list[dict[str, str]],
        timeout: float,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        text = self._invoke(
            messages=messages,
            timeout=timeout,
            schema_name=schema_name,
            prompt_version=prompt_version,
        )
        try:
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            self._mark_parse_error(schema_name=schema_name, error=str(exc))
            raise

    def chat_text(self, messages: list[dict[str, str]], prompt_version: str | None = None) -> str:
        return self._invoke(
            messages=messages,
            timeout=self.timeout_seconds,
            schema_name=None,
            prompt_version=prompt_version,
        )

    def _invoke(
        self,
        messages: list[dict[str, str]],
        timeout: float,
        schema_name: str | None,
        prompt_version: str | None,
    ) -> str:
        started_at = _utc_now()
        t0 = time.perf_counter()

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                timeout=timeout,
                temperature=0.1,
            )
            response = model.invoke(self._to_langchain_messages(messages, AIMessage, HumanMessage, SystemMessage))
            content = self._response_content_to_text(response.content)
            usage = self._extract_usage(response)

            self._record_call(
                schema_name=schema_name,
                prompt_version=prompt_version,
                started_at=started_at,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                response_size=len(content.encode("utf-8")),
                token_in=usage.get("input_tokens"),
                token_out=usage.get("output_tokens"),
                error=None,
            )
            return content
        except Exception as exc:  # noqa: BLE001
            self._record_call(
                schema_name=schema_name,
                prompt_version=prompt_version,
                started_at=started_at,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                response_size=0,
                token_in=None,
                token_out=None,
                error=str(exc),
            )
            raise

    def _to_langchain_messages(
        self,
        messages: list[dict[str, str]],
        ai_cls,
        human_cls,
        system_cls,
    ) -> list[Any]:
        out: list[Any] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                out.append(system_cls(content=content))
            elif role == "assistant":
                out.append(ai_cls(content=content))
            else:
                out.append(human_cls(content=content))
        return out

    def _response_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    def _extract_usage(self, response: Any) -> dict[str, int | None]:
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage_metadata", {}) if isinstance(meta, dict) else {}
        in_tokens = usage.get("input_tokens")
        out_tokens = usage.get("output_tokens")
        return {
            "input_tokens": int(in_tokens) if isinstance(in_tokens, int) else None,
            "output_tokens": int(out_tokens) if isinstance(out_tokens, int) else None,
        }

    def _mark_parse_error(self, schema_name: str, error: str) -> None:
        if self.last_call is None:
            return
        self.last_call.error = f"parse_error: {error}"
        logger.warning(
            "llm_call provider=%s model=%s schema=%s latency_ms=%s token_in=%s token_out=%s error=%s",
            self.provider_name,
            self.model_name,
            schema_name,
            self.last_call.latency_ms,
            self.last_call.token_in,
            self.last_call.token_out,
            self.last_call.error,
        )

    def _record_call(
        self,
        schema_name: str | None,
        prompt_version: str | None,
        started_at: datetime,
        latency_ms: int,
        response_size: int,
        token_in: int | None,
        token_out: int | None,
        error: str | None,
    ) -> None:
        ended_at = _utc_now()
        self.last_call = LLMCallRecord(
            provider_name=self.provider_name,
            model_name=self.model_name,
            schema_name=schema_name,
            prompt_version=prompt_version,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            response_size=response_size,
            token_in=token_in,
            token_out=token_out,
            error=error,
        )
        level = logging.INFO if error is None else logging.WARNING
        logger.log(
            level,
            "llm_call provider=%s model=%s schema=%s prompt_version=%s latency_ms=%s token_in=%s token_out=%s response_size=%s error=%s",
            self.provider_name,
            self.model_name,
            schema_name or "text",
            prompt_version or "",
            latency_ms,
            token_in,
            token_out,
            response_size,
            error or "",
        )


__all__ = ["GeminiLLMProvider"]
