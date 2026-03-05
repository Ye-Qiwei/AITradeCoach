"""OpenAI-backed LLM provider."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ai_trading_coach.llm.provider import LLMCallRecord, parse_json_object

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OpenAILLMProvider:
    """LLM provider implementation using the official OpenAI SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.provider_name = "openai"
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
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, timeout=timeout)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[self._to_openai_message(msg) for msg in messages if msg.get("content")],
                temperature=0.1,
            )
            content = self._extract_content(response)
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

    def _to_openai_message(self, item: dict[str, str]) -> dict[str, str]:
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        return {
            "role": role,
            "content": str(item.get("content", "")),
        }

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    def _extract_usage(self, response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return {
            "input_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            "output_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
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


__all__ = ["OpenAILLMProvider"]
