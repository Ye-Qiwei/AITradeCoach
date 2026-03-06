"""LLM-only parser for structured judgement extraction."""

from __future__ import annotations

import json
from datetime import date

from pydantic import ValidationError

from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ParserOutput
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace
from ai_trading_coach.prompts.prompt_store import PromptStore


class CombinedParserAgent:
    schema_name = "parser_output.v2"
    prompt_version = "parser.v2"

    def __init__(self, provider: LLMProvider, timeout_seconds: float, prompt_store: PromptStore | None = None) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.prompt_store = prompt_store

    def parse(self, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> tuple[ParserOutput, object | None]:
        messages = self._build_messages(run_id=run_id, user_id=user_id, run_date=run_date, raw_log_text=raw_log_text)
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            result = ParserOutput.model_validate(payload)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
            raise LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}") from exc

        trace = build_model_trace(
            purpose=ModelCallPurpose.LOG_UNDERSTANDING,
            input_summary=f"run_id={run_id}; chars={len(raw_log_text)}",
            output_summary=f"judgements={len(result.all_judgements())}; actions={len(result.trade_actions)}",
            provider_record=getattr(self.provider, "last_call", None),
        )
        return result, trace

    def _build_messages(self, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> list[dict[str, str]]:
        default_prompt = (
            "You extract structured trading judgements from daily logs. Return JSON only. "
            "Extract explicit + implicit views, potential opportunities, non-actions, and reflection. "
            "Every judgement must include evidence_from_user_log quotes; do not hallucinate without evidence. "
            "proposed_evaluation_window must be one of: 1 day, 1 week, 1 month, 3 months, 1 year."
        )
        system_prompt = self.prompt_store.load_prompt("log_understanding", default_prompt) if self.prompt_store else default_prompt
        user_payload = {
            "run_id": run_id,
            "user_id": user_id,
            "run_date": run_date.isoformat(),
            "raw_log_text": raw_log_text,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
