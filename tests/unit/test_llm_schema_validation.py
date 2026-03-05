from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMCallRecord
from ai_trading_coach.modules.agent.combined_parser_agent import CombinedParserAgent


class InvalidParserProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_call: LLMCallRecord | None = None

    def chat_json(self, schema_name: str, messages: list[dict[str, str]], timeout: float, prompt_version: str | None = None):
        del messages, timeout
        now = datetime.now(timezone.utc)
        self.last_call = LLMCallRecord(
            provider_name=self.provider_name,
            model_name=self.model_name,
            schema_name=schema_name,
            prompt_version=prompt_version,
            started_at=now,
            ended_at=now,
            latency_ms=1,
            response_size=18,
            token_in=1,
            token_out=1,
            error=None,
        )
        return {"not_valid": True}

    def chat_text(self, messages: list[dict[str, str]], prompt_version: str | None = None) -> str:
        del messages, prompt_version
        return ""


def test_invalid_json_schema_fails_without_fallback() -> None:
    agent = CombinedParserAgent(provider=InvalidParserProvider(), timeout_seconds=3.0)
    request = ReviewRunRequest(
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 3, 5),
        trigger_type="manual",
        raw_log_text="## 交易记录\n- AAPL BUY 10 180",
    )
    with pytest.raises(LLMOutputValidationError) as exc:
        agent.parse(
            run_id=request.run_id,
            user_id=request.user_id,
            run_date=request.run_date,
            raw_log_text=request.raw_log_text,
        )
    assert "combined_parse_result.v1" in str(exc.value)

