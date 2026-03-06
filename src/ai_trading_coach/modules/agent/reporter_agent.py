"""Reporter agent generating daily feedback with evaluation windows."""

from __future__ import annotations

import json

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace
from ai_trading_coach.prompts.prompt_store import PromptStore


class ReporterAgent:
    schema_name = "reporter_output.v2"
    prompt_version = "reporter.v2"

    def __init__(self, provider: LLMProvider, timeout_seconds: float, prompt_store: PromptStore | None = None) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.prompt_store = prompt_store

    def generate(self, *, evidence_packet: EvidencePacket, report_context: dict[str, object], rewrite_instruction: str | None = None):
        messages = self._build_messages(evidence_packet=evidence_packet, report_context=report_context, rewrite_instruction=rewrite_instruction)
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            out = ReporterOutput.model_validate(payload)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
            raise LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}") from exc

        trace = build_model_trace(
            purpose=ModelCallPurpose.REPORT_GENERATION,
            input_summary=f"sources={len(evidence_packet.source_registry)}; judgements={len(report_context.get('judgements', []))}",
            output_summary=f"markdown_chars={len(out.markdown)}; feedback_items={len(out.judgement_feedback)}",
            provider_record=getattr(self.provider, "last_call", None),
        )
        return out, trace

    def _build_messages(self, *, evidence_packet: EvidencePacket, report_context: dict[str, object], rewrite_instruction: str | None) -> list[dict[str, str]]:
        default_prompt = (
            "Generate a daily trading review report. Return strict JSON with markdown + judgement_feedback. "
            "Each judgement_feedback must include evaluation_window from this set only: "
            f"{', '.join(ALLOWED_EVALUATION_WINDOWS)}. Window selection MUST be reasoned by judgement content, not rigid rules. "
            "Cite sources as [source:<id>] in markdown and include source_ids in judgement_feedback."
        )
        system_prompt = self.prompt_store.load_prompt("report_generation", default_prompt) if self.prompt_store else default_prompt
        user_payload = {
            "report_context": report_context,
            "source_index": [s.source_id for s in evidence_packet.source_registry],
            "rewrite_instruction": rewrite_instruction,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
