"""LLM-only parser that outputs normalized log + cognition state in one call."""

from __future__ import annotations

import json
from datetime import date

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import CombinedParseResult
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace


class CombinedParserAgent:
    """Parse raw markdown log into CombinedParseResult via one strict JSON call."""

    schema_name = "combined_parse_result.v1"
    prompt_version = "combined_parse.v1"

    def __init__(self, provider: LLMProvider, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def parse(
        self,
        *,
        run_id: str,
        user_id: str,
        run_date: date,
        raw_log_text: str,
    ) -> tuple[CombinedParseResult, ModelCallTrace | None]:
        messages = self._build_messages(
            run_id=run_id,
            user_id=user_id,
            run_date=run_date,
            raw_log_text=raw_log_text,
        )
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )

        try:
            result = CombinedParseResult.model_validate(payload)
        except ValidationError as exc:
            raise self._validation_error(exc) from exc

        input_summary = f"run_id={run_id}; chars={len(raw_log_text)}"
        output_summary = (
            f"log_date={result.normalized_log.log_date.isoformat()}; "
            f"hypotheses={len(result.cognition_state.hypotheses)}"
        )
        trace = build_model_trace(
            purpose=ModelCallPurpose.LOG_UNDERSTANDING,
            input_summary=input_summary,
            output_summary=output_summary,
            provider_record=getattr(self.provider, "last_call", None),
        )
        return result, trace

    def _validation_error(self, exc: ValidationError) -> LLMOutputValidationError:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()[:5]
        )
        return LLMOutputValidationError(
            f"Schema validation failed for {self.schema_name}: {detail or exc.__class__.__name__}"
        )

    def _build_messages(
        self,
        *,
        run_id: str,
        user_id: str,
        run_date: date,
        raw_log_text: str,
    ) -> list[dict[str, str]]:
        schema_hint = {
            "parse_id": "parse_<run_id>",
            "normalized_log": {
                "log_id": "log_<run_id>",
                "user_id": user_id,
                "log_date": run_date.isoformat(),
                "traded_tickers": ["AAPL.US"],
                "mentioned_tickers": ["AAPL.US"],
                "user_state": {"emotion": "calm", "stress": 3, "focus": 7},
                "market_context": {"regime": "range", "key_variables": ["rates"]},
                "trade_events": [],
                "trade_narratives": [],
                "scan_signals": {"anxiety": [], "fomo": [], "not_trade": []},
                "reflection": {"facts": [], "gaps": [], "lessons": []},
                "ai_directives": [],
                "raw_text": raw_log_text[:200],
                "field_errors": [],
            },
            "cognition_state": {
                "cognition_id": "cog_<run_id>",
                "log_id": "log_<run_id>",
                "user_id": user_id,
                "as_of_date": run_date.isoformat(),
                "core_judgements": [],
                "hypotheses": [],
                "risk_concerns": [],
                "outside_opportunities": [],
                "deliberate_no_trade_decisions": [],
                "explicit_rules": [],
                "fuzzy_tendencies": [],
                "fact_statements": [],
                "subjective_statements": [],
                "behavioral_signals": [],
                "emotion_signals": [],
                "user_intent_signals": [],
            },
        }
        system_prompt = (
            "You are a strict trading-log structuring engine. Return JSON only. "
            "Do not emit markdown, prose, or code fences. "
            "Output one object matching schema combined_parse_result.v1. "
            "All extracted fields must be grounded in raw input text."
        )
        user_payload = {
            "run_id": run_id,
            "user_id": user_id,
            "run_date": run_date.isoformat(),
            "raw_log_text": raw_log_text,
            "schema_example": schema_hint,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

