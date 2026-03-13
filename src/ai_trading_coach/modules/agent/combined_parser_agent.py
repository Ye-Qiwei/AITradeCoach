"""LLM parser for weakly-structured judgement extraction."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ParserOutput
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.text_output_parsing import parse_parser_output_text


class CombinedParserAgent:
    prompt_name = "log_understanding"

    def __init__(self, gateway: LangChainLLMGateway, prompt_manager: PromptManager) -> None:
        self.gateway = gateway
        self.prompt_manager = prompt_manager

    def parse(self, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> tuple[ParserOutput, object | None]:
        prompt = self.prompt_manager.load_active(self.prompt_name)
        user_payload = {
            "user_id": user_id,
            "run_date": run_date.isoformat(),
            "raw_log_text": raw_log_text,
            "extraction_targets": [
                "trade_actions",
                "judgements",
            ],
        }
        messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, payload=user_payload)
        raw_text, trace = self.gateway.invoke_text(
            messages=messages,
            purpose=ModelCallPurpose.LOG_UNDERSTANDING,
            prompt_version=f"{prompt.prompt_name}.{prompt.version}",
            input_summary=f"chars={len(raw_log_text)}",
        )
        return parse_parser_output_text(
            raw_text,
            run_id="",
            user_id=user_id,
            run_date=run_date,
            raw_log_text=raw_log_text,
        ), trace
