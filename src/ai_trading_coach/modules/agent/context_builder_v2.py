"""Context engineering helpers for planner/reporter/judge."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from typing import Any

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import CombinedParseResult
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket


class ContextBuilderV2:
    """Build minimal, budget-aware context payloads for each agent stage."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def for_planner(
        self,
        *,
        parse_result: CombinedParseResult,
        memory_summary: str | None = None,
        positions: list[dict[str, Any]] | None = None,
        trades: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "parse": {
                "log_date": parse_result.normalized_log.log_date.isoformat(),
                "traded_tickers": parse_result.normalized_log.traded_tickers,
                "mentioned_tickers": parse_result.normalized_log.mentioned_tickers,
                "market_context": parse_result.normalized_log.market_context.model_dump(mode="json"),
                "cognition": {
                    "hypotheses": [item.model_dump(mode="json") for item in parse_result.cognition_state.hypotheses],
                    "risk_concerns": parse_result.cognition_state.risk_concerns,
                    "intent": [item.question for item in parse_result.cognition_state.user_intent_signals],
                    "time_horizon": [
                        item.timeframe_hint for item in parse_result.cognition_state.hypotheses if item.timeframe_hint
                    ],
                },
            },
            "memory_summary": memory_summary or "",
            "positions": positions or [],
            "trade_history": trades or [],
        }
        return self._fit_budget(payload, self.settings.context_budget_planner)

    def for_reporter(
        self,
        *,
        evidence_packet: EvidencePacket,
        intent: list[str],
        investigation_outline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_rows = [self._item_to_row(item) for item in self._iter_packet_items(evidence_packet)]
        selected = self._top_k_bucketed(evidence_rows, per_bucket=2, total_k=30)
        payload = {
            "investigation_outline": investigation_outline or {},
            "intent": intent,
            "evidence": selected,
            "source_index": [
                {
                    "source_id": source.source_id,
                    "uri": source.uri,
                    "title": source.title,
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                }
                for source in evidence_packet.source_registry
            ],
        }
        return self._fit_budget(payload, self.settings.context_budget_reporter)

    def for_judge(
        self,
        *,
        report_markdown: str,
        evidence_packet: EvidencePacket,
        intent: list[str],
        rewrite_instruction: str | None,
    ) -> dict[str, Any]:
        payload = {
            "report_markdown": report_markdown,
            "intent": intent,
            "source_ids": [source.source_id for source in evidence_packet.source_registry],
            "rewrite_instruction_used": rewrite_instruction,
        }
        return self._fit_budget(payload, self.settings.context_budget_judge)

    def _fit_budget(self, payload: dict[str, Any], budget: int) -> dict[str, Any]:
        if budget <= 0:
            return payload

        text = json.dumps(payload, ensure_ascii=False)
        if len(text) <= budget:
            return payload

        # Keep deterministic truncation for stability in tests.
        clipped = text[:budget]
        return {"clipped_context_json": clipped}

    def _top_k_bucketed(
        self,
        rows: list[dict[str, Any]],
        *,
        per_bucket: int,
        total_k: int,
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('ticker','*')}|{row.get('provider','unknown')}|{row.get('date_bucket','na')}"
            buckets[key].append(row)

        selected: list[dict[str, Any]] = []
        for key in sorted(buckets):
            selected.extend(buckets[key][:per_bucket])
        return selected[:total_k]

    def _item_to_row(self, item: EvidenceItem) -> dict[str, Any]:
        first_source = item.sources[0] if item.sources else None
        date_bucket = (
            item.event_time.date().isoformat()
            if item.event_time is not None
            else date.today().isoformat()
        )
        return {
            "item_id": item.item_id,
            "summary": item.summary,
            "evidence_type": item.evidence_type.value,
            "ticker": item.related_tickers[0] if item.related_tickers else "*",
            "provider": first_source.provider if first_source else "unknown",
            "source_id": first_source.source_id if first_source else "",
            "date_bucket": date_bucket,
            "data": item.data,
        }

    def _iter_packet_items(self, packet: EvidencePacket) -> list[EvidenceItem]:
        return [
            *packet.price_evidence,
            *packet.news_evidence,
            *packet.filing_evidence,
            *packet.macro_evidence,
            *packet.market_regime_evidence,
            *packet.discussion_evidence,
            *packet.sentiment_evidence,
            *packet.analog_evidence,
        ]
