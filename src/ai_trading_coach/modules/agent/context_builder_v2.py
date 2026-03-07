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
        all_items = [
            *evidence_packet.price_evidence,
            *evidence_packet.news_evidence,
            *evidence_packet.filing_evidence,
            *evidence_packet.sentiment_evidence,
            *evidence_packet.market_regime_evidence,
            *evidence_packet.discussion_evidence,
            *evidence_packet.analog_evidence,
            *evidence_packet.macro_evidence,
        ]
        evidence_index = {item.item_id: {"summary": item.summary, "source_ids": [s.source_id for s in item.sources]} for item in all_items}
        return {
            "judgements": [
                {
                    "judgement_id": j.judgement_id,
                    "thesis": j.thesis,
                    "target": j.target_asset_or_topic,
                    "user_evidence": j.evidence_from_user_log,
                    "proposed_evaluation_window": j.proposed_evaluation_window,
                    "research_signal": evidence_map[j.judgement_id].support_signal,
                    "research_sufficiency": evidence_map[j.judgement_id].sufficiency_reason,
                    "research_evidence_item_ids": evidence_map[j.judgement_id].evidence_item_ids,
                }
                for j in judgement_map.values()
            ],
            "evidence_index": evidence_index,
            "source_index": [source.source_id for source in evidence_packet.source_registry],
            "research_stop_reason": research_output.stop_reason,
        }

    def for_judge(
        self,
        *,
        report_markdown: str,
        judgement_feedback: list[DailyJudgementFeedback],
        parse_result: ParserOutput,
        research_output: ResearchOutput,
    ) -> dict[str, Any]:
        return {
            "report_markdown": report_markdown,
            "judgement_feedback": [item.model_dump(mode="json") for item in judgement_feedback],
            "expected_judgement_ids": [j.judgement_id for j in parse_result.all_judgements()],
            "research_output": research_output.model_dump(mode="json"),
        }
