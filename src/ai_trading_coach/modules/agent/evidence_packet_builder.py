"""Evidence packet assembly utilities."""

from __future__ import annotations

from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket


def build_evidence_packet(*, packet_id: str, user_id: str, evidence_items: list[EvidenceItem]) -> EvidencePacket:
    packet = EvidencePacket(packet_id=packet_id, user_id=user_id)
    for item in evidence_items:
        packet.source_registry.extend(item.sources)
        if item.evidence_type == EvidenceType.PRICE_PATH:
            packet.price_evidence.append(item)
        elif item.evidence_type == EvidenceType.NEWS:
            packet.news_evidence.append(item)
        elif item.evidence_type == EvidenceType.FILING:
            packet.filing_evidence.append(item)
        elif item.evidence_type == EvidenceType.MACRO:
            packet.macro_evidence.append(item)
        elif item.evidence_type == EvidenceType.SENTIMENT:
            packet.sentiment_evidence.append(item)
        elif item.evidence_type == EvidenceType.DISCUSSION:
            packet.discussion_evidence.append(item)
        elif item.evidence_type == EvidenceType.ANALOG_HISTORY:
            packet.analog_evidence.append(item)
        else:
            packet.market_regime_evidence.append(item)
    total_buckets = [
        packet.price_evidence,
        packet.news_evidence,
        packet.filing_evidence,
        packet.macro_evidence,
        packet.sentiment_evidence,
        packet.discussion_evidence,
        packet.analog_evidence,
        packet.market_regime_evidence,
    ]
    non_empty = sum(1 for bucket in total_buckets if bucket)
    packet.completeness_score = non_empty / len(total_buckets) if evidence_items else 0.0
    return packet
