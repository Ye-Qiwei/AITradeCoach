"""Reporter agent generating daily feedback with evaluation windows."""

from __future__ import annotations

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.llm_output_adapters import reporter_contract_to_domain
from ai_trading_coach.domain.llm_output_contracts import ReporterOutputContract
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent.prompting import PromptManager


class ReporterAgent:
    prompt_name = "report_generation"

    def __init__(self, gateway: LangChainLLMGateway, prompt_manager: PromptManager) -> None:
        self.gateway = gateway
        self.prompt_manager = prompt_manager

    def generate(self, *, evidence_packet: EvidencePacket, report_context: dict[str, object], rewrite_instruction: str | None = None) -> tuple[ReporterOutput, object | None]:
        prompt = self.prompt_manager.load_active(self.prompt_name)
        user_payload = {
            "report_context": report_context,
            "source_index": [s.source_id for s in evidence_packet.source_registry],
            "rewrite_instruction": rewrite_instruction,
            "constraints": {
                "must_cover_all_judgements": True,
                "must_use_source_citations": True,
                "judgement_order_must_match_input": True,
                "source_ids_must_match_markdown_citations": True,
            },
        }
        messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, payload=user_payload)
        contract_out, trace = self.gateway.invoke_structured(
            schema=ReporterOutputContract,
            messages=messages,
            purpose=ModelCallPurpose.REPORT_GENERATION,
            prompt_version=f"{prompt.prompt_name}.{prompt.version}",
            input_summary=f"sources={len(evidence_packet.source_registry)}; judgements={len(report_context.get('judgement_bundles', []))}",
            output_summary_builder=lambda out: f"markdown_chars={len(out.markdown)}; feedback_items={len(out.judgement_feedback)}",
        )
        return reporter_contract_to_domain(contract_out), trace
