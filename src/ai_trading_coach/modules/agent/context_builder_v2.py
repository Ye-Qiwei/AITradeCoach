"""Build judgement-centric contexts for reporter and judge."""

from __future__ import annotations

from typing import Any

from ai_trading_coach.domain.judgement_models import DailyJudgementFeedback, ParserOutput, ResearchOutput
from ai_trading_coach.domain.models import EvidencePacket


class ContextBuilderV2:
    def for_reporter(self, *, parse_result: ParserOutput, research_output: ResearchOutput, evidence_packet: EvidencePacket) -> dict[str, Any]:
        _ = parse_result
        _ = evidence_packet
        bundles = [item.model_dump(mode="json") for item in research_output.judgements]
        return {"judgement_bundles": bundles}

    def for_judge(self, *, report_markdown: str, judgement_feedback: list[DailyJudgementFeedback], parse_result: ParserOutput, research_output: ResearchOutput, report_context: dict[str, Any]) -> dict[str, Any]:
        _ = parse_result
        return {
            "report_markdown": report_markdown,
            "judgement_feedback": [item.model_dump(mode="json") for item in judgement_feedback],
            "judgement_bundles": report_context.get("judgement_bundles", [item.model_dump(mode="json") for item in research_output.judgements]),
        }
