"""LLM-only parser for structured judgement extraction."""

from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ParserOutput
from ai_trading_coach.domain.llm_output_adapters import parser_contract_to_domain
from ai_trading_coach.domain.llm_output_contracts import ParserOutputContract
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent.prompting import PromptManager


class CombinedParserAgent:
    prompt_name = "log_understanding"

    def __init__(self, gateway: LangChainLLMGateway, prompt_manager: PromptManager) -> None:
        self.gateway = gateway
        self.prompt_manager = prompt_manager

    def parse(self, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> tuple[ParserOutput, object | None]:
        prompt = self.prompt_manager.load_active(self.prompt_name)
        user_payload = {
            "run_id": run_id,
            "user_id": user_id,
            "run_date": run_date.isoformat(),
            "raw_log_text": raw_log_text,
            "extraction_targets": [
                "trade_actions",
                "explicit_judgements",
                "implicit_judgements",
                "opportunity_judgements",
                "non_action_judgements",
                "reflection_summary",
            ],
        }
        messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, payload=user_payload)
        contract_out, trace = self.gateway.invoke_structured(
            schema=ParserOutputContract,
            messages=messages,
            purpose=ModelCallPurpose.LOG_UNDERSTANDING,
            prompt_version=f"{prompt.prompt_name}.{prompt.version}",
            input_summary=f"run_id={run_id}; chars={len(raw_log_text)}",
            output_summary_builder=lambda out: f"judgements={len(out.all_judgements())}; actions={len(out.trade_actions)}",
        )
        return parser_contract_to_domain(contract_out), trace
