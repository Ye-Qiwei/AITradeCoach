"""Shared tracing helpers for agent LLM calls."""

from __future__ import annotations

from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace


def build_model_trace(
    *,
    purpose: ModelCallPurpose,
    input_summary: str,
    output_summary: str,
    provider_record,
) -> ModelCallTrace | None:
    if provider_record is None:
        return None

    return ModelCallTrace(
        call_id=f"model_{purpose.value}_{int(provider_record.started_at.timestamp() * 1000)}",
        purpose=purpose,
        model_name=provider_record.model_name,
        provider=provider_record.provider_name,
        prompt_version=provider_record.prompt_version,
        started_at=provider_record.started_at,
        ended_at=provider_record.ended_at,
        input_summary=input_summary,
        output_summary=output_summary,
        token_in=provider_record.token_in,
        token_out=provider_record.token_out,
        response_size=provider_record.response_size,
        error_message=provider_record.error,
        latency_ms=provider_record.latency_ms,
    )

