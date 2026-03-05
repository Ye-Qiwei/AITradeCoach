"""Planner agent: convert parsed cognition into tool-executable plan."""

from __future__ import annotations

import json

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import CombinedParseResult, Plan
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace


class PlannerAgent:
    schema_name = "plan.v1"
    prompt_version = "planner.v1"

    def __init__(self, provider: LLMProvider, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def plan(
        self,
        *,
        parse_result: CombinedParseResult,
        planner_context: dict[str, object],
    ) -> tuple[Plan, ModelCallTrace | None]:
        messages = self._build_messages(parse_result=parse_result, planner_context=planner_context)
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            plan = Plan.model_validate(payload)
        except ValidationError as exc:
            raise self._validation_error(exc) from exc

        input_summary = (
            f"tickers={len(parse_result.normalized_log.traded_tickers) + len(parse_result.normalized_log.mentioned_tickers)}; "
            f"hypotheses={len(parse_result.cognition_state.hypotheses)}"
        )
        output_summary = f"subtasks={len(plan.subtasks)}; risks={len(plan.risk_uncertainties)}"
        trace = build_model_trace(
            purpose=ModelCallPurpose.EVIDENCE_PLANNING,
            input_summary=input_summary,
            output_summary=output_summary,
            provider_record=getattr(self.provider, "last_call", None),
        )
        return plan, trace

    def _validation_error(self, exc: ValidationError) -> LLMOutputValidationError:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()[:5]
        )
        return LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}")

    def _build_messages(
        self,
        *,
        parse_result: CombinedParseResult,
        planner_context: dict[str, object],
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the planning stage of a trading review agent. Return JSON only. "
            "You must output schema plan.v1 with subtasks executable by tool_category. "
            "Use tool_category from: market_data, news_search, filings_financials, macro_data."
        )
        user_payload = {
            "normalized_log": parse_result.normalized_log.model_dump(mode="json"),
            "cognition_state": parse_result.cognition_state.model_dump(mode="json"),
            "planner_context": planner_context,
            "requirements": {
                "include_success_criteria": True,
                "include_stop_conditions": True,
                "subtask_count": "0..8",
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

