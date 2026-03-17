"""Reporter agent generating daily feedback with evaluation windows."""

from __future__ import annotations

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent.prompting import PromptManager
from ai_trading_coach.modules.agent.text_output_parsing import parse_reporter_output_text


class ReporterAgent:
    prompt_name = "report_generation"

    def __init__(self, gateway: LangChainLLMGateway, prompt_manager: PromptManager) -> None:
        self.gateway = gateway
        self.prompt_manager = prompt_manager

    def generate(self, *, evidence_packet: EvidencePacket, report_context: dict[str, object], rewrite_instruction: str | None = None) -> tuple[ReporterOutput, object | None]:
        prompt = self.prompt_manager.load_active(self.prompt_name)
        user_context = {
            "report_context": report_context,
            "sources": [{"source_id": s.source_id or "", "provider": s.provider, "title": s.title or "", "uri": s.uri or ""} for s in evidence_packet.source_registry],
            "rewrite_instruction": rewrite_instruction,
            "constraints": {
                "must_cover_all_judgements": True,
                "must_use_source_citations": True,
                "judgement_order_must_match_input": True,
                "final_report_language": "zh-CN",
            },
        }
        messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, context=user_context)
        raw_text, trace = self.gateway.invoke_text(
            messages=messages,
            purpose=ModelCallPurpose.REPORT_GENERATION,
            prompt_version=prompt.prompt_name,
            input_summary=f"sources={len(evidence_packet.source_registry)}; judgements={len(report_context.get('judgement_bundles', []))}",
        )
        return parse_reporter_output_text(raw_text, len(report_context.get("judgement_bundles", []))), trace
