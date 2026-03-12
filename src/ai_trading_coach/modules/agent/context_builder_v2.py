"""Build judgement-centric contexts for reporter and judge."""

from __future__ import annotations

from typing import Any

from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, ParserOutput, ResearchOutput
from ai_trading_coach.domain.models import EvidencePacket


class ContextBuilderV2:
    def for_reporter(self, *, parse_result: ParserOutput, research_output: ResearchOutput, evidence_packet: EvidencePacket) -> dict[str, Any]:
        all_items = [*evidence_packet.price_evidence, *evidence_packet.news_evidence, *evidence_packet.filing_evidence, *evidence_packet.sentiment_evidence, *evidence_packet.market_regime_evidence, *evidence_packet.discussion_evidence, *evidence_packet.analog_evidence, *evidence_packet.macro_evidence]
        item_map = {item.item_id: item for item in all_items}
        research_map = {item.judgement_id: item for item in research_output.judgement_evidence}
        global_source_index = {src.source_id: {"title": src.title, "provider": src.provider, "uri": src.uri, "published_at": src.published_at.isoformat() if src.published_at else None} for src in evidence_packet.source_registry}

        bundles: list[dict[str, Any]] = []
        for judgement in parse_result.all_judgements():
            rs = research_map.get(judgement.judgement_id)
            item_ids = rs.evidence_item_ids if rs else []
            bundle_items: list[dict[str, Any]] = []
            for item_id in item_ids:
                item = item_map.get(item_id)
                if not item:
                    continue
                bundle_items.append({
                    "item_id": item.item_id,
                    "evidence_type": item.evidence_type,
                    "summary": item.summary,
                    "related_tickers": item.related_tickers,
                    "source_ids": [s.source_id for s in item.sources],
                    "source_metadata": [{"source_id": s.source_id, "title": s.title, "provider": s.provider, "uri": s.uri, "published_at": s.published_at.isoformat() if s.published_at else None} for s in item.sources],
                })
            allowed_source_ids = sorted({sid for item in bundle_items for sid in item["source_ids"]})
            bundles.append({
                "judgement_id": judgement.judgement_id,
                "category": judgement.category,
                "target": judgement.target_asset_or_topic,
                "thesis": judgement.thesis,
                "evidence_from_user_log": judgement.evidence_from_user_log,
                "implicitness": judgement.implicitness,
                "proposed_evaluation_window": judgement.proposed_evaluation_window,
                "atomic_judgements": [a.model_dump(mode="json") for a in judgement.atomic_judgements],
                "research_signal": rs.support_signal if rs else "uncertain",
                "sufficiency_reason": rs.sufficiency_reason if rs else "",
                "evidence_items": bundle_items,
                "allowed_source_ids": allowed_source_ids,
            })
        return {"judgement_bundles": bundles, "global_source_index": global_source_index, "research_stop_reason": research_output.stop_reason, "coverage_summary": {"judgements": len(parse_result.all_judgements()), "with_evidence": sum(1 for item in research_output.judgement_evidence if item.evidence_item_ids)}}

    def for_judge(self, *, report_markdown: str, judgement_feedback: list[DailyJudgementFeedback], parse_result: ParserOutput, research_output: ResearchOutput, report_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_markdown": report_markdown,
            "judgement_feedback": [item.model_dump(mode="json") for item in judgement_feedback],
            "expected_judgement_ids": [j.judgement_id for j in parse_result.all_judgements()],
            "research_output": research_output.model_dump(mode="json"),
            "judgement_bundles": report_context.get("judgement_bundles", []),
            "global_source_index": report_context.get("global_source_index", {}),
        }
