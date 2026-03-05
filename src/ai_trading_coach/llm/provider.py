"""LLM provider protocol and shared helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMCallRecord:
    """Trace payload for one provider call."""

    provider_name: str
    model_name: str
    schema_name: str | None
    prompt_version: str | None
    started_at: datetime
    ended_at: datetime
    latency_ms: int
    response_size: int
    token_in: int | None = None
    token_out: int | None = None
    error: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Abstraction over concrete LLM SDK clients."""

    provider_name: str
    model_name: str
    last_call: LLMCallRecord | None

    def chat_json(
        self,
        schema_name: str,
        messages: list[dict[str, str]],
        timeout: float,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        """Return parsed JSON object for the target schema."""

    def chat_text(self, messages: list[dict[str, str]], prompt_version: str | None = None) -> str:
        """Return plain text completion."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fenced code blocks."""

    payload = text.strip()
    if payload.startswith("```"):
        payload = payload[3:].strip()
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
        if payload.endswith("```"):
            payload = payload[:-3].strip()

    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("LLM JSON output must be an object")
    return loaded


__all__ = ["LLMCallRecord", "LLMProvider", "parse_json_object"]
