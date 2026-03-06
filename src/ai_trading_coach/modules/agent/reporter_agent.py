"""Reporter agent that drafts markdown with explicit source citations."""

from __future__ import annotations

import json

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import ReporterDraft
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import EvidencePacket, ModelCallTrace
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace


class ReporterAgent:
    schema_name = "reporter_draft.v1"
    prompt_version = "reporter.v1"

    def __init__(self, provider: LLMProvider, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        evidence_packet: EvidencePacket,
        report_context: dict[str, object],
        intent: list[str],
        rewrite_instruction: str | None = None,
    ) -> tuple[ReporterDraft, ModelCallTrace | None]:
        messages = self._build_messages(
            evidence_packet=evidence_packet,
            report_context=report_context,
            intent=intent,
            rewrite_instruction=rewrite_instruction,
        )
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            draft = ReporterDraft.model_validate(payload)
        except ValidationError as exc:
            raise self._validation_error(exc) from exc

        trace = build_model_trace(
            purpose=ModelCallPurpose.REPORT_GENERATION,
            input_summary=f"evidence_sources={len(evidence_packet.source_registry)}; intent={len(intent)}",
            output_summary=f"markdown_chars={len(draft.markdown)}; rewritten={bool(rewrite_instruction)}",
            provider_record=getattr(self.provider, "last_call", None),
        )
        return draft, trace

    def _validation_error(self, exc: ValidationError) -> LLMOutputValidationError:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()[:5]
        )
        return LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}")

    def _build_messages(
        self,
        *,
        evidence_packet: EvidencePacket,
        report_context: dict[str, object],
        intent: list[str],
        rewrite_instruction: str | None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the reporter stage for a trading review pipeline. Return JSON only with key markdown. "
            "In markdown, every factual bullet and key conclusion must include citation tag "
            "[source:<source_id>]. Do not fabricate source_id."
        )
        user_payload = {
            "intent": intent,
            "source_index": [
                {
                    "source_id": source.source_id,
                    "uri": source.uri,
                    "title": source.title,
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                }
                for source in evidence_packet.source_registry
            ],
            "report_context": report_context,
            "rewrite_instruction": rewrite_instruction,
            "format": {
                "title": "# Daily Review Report",
                "required_sections": [
                    "## Summary",
                    "## Evidence",
                    "## Key Risks",
                    "## Actions",
                ],
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

