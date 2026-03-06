"""Build compact contexts for reporting and judging."""

from __future__ import annotations

from typing import Any

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, ParserOutput, ResearchOutput
from ai_trading_coach.domain.models import EvidencePacket


class ContextBuilderV2:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def for_reporter(
        self,
        *,
        parse_result: ParserOutput,
        research_output: ResearchOutput,
        evidence_packet: EvidencePacket,
    ) -> dict[str, Any]:
        judgement_map = {j.judgement_id: j for j in parse_result.all_judgements()}
        evidence_map = {e.judgement_id: e for e in research_output.judgement_evidence}
        return {
            "judgements": [
                {
                    "judgement_id": j.judgement_id,
                    "thesis": j.thesis,
                    "target": j.target_asset_or_topic,
                    "user_evidence": j.evidence_from_user_log,
                    "proposed_evaluation_window": j.proposed_evaluation_window,
                    "research_signal": evidence_map.get(j.judgement_id).support_signal if j.judgement_id in evidence_map else "uncertain",
                    "research_sufficiency": evidence_map.get(j.judgement_id).sufficiency_reason if j.judgement_id in evidence_map else "",
                }
                for j in judgement_map.values()
            ],
            "source_index": [source.source_id for source in evidence_packet.source_registry],
        }

    def for_judge(self, *, report_markdown: str, judgement_feedback: list[DailyJudgementFeedback]) -> dict[str, Any]:
        return {
            "report_markdown": report_markdown,
            "judgement_feedback": [item.model_dump(mode="json") for item in judgement_feedback],
        }
