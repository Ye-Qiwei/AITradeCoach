"""Adapter orchestrator for ReAct research stage."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_coach.domain.agent_models import CombinedParseResult
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.domain.react_models import ResearchSummary
from ai_trading_coach.modules.agent.react_research_agent import ReActResearchAgent
from ai_trading_coach.modules.agent.react_tools import build_evidence_packet


@dataclass
class ResearchStageResult:
    summary: ResearchSummary
    evidence_packet: EvidencePacket


class ReActResearchOrchestrator:
    def __init__(self, *, react_agent: ReActResearchAgent) -> None:
        self.react_agent = react_agent

    def run(self, *, run_id: str, user_id: str, parse_result: CombinedParseResult) -> ResearchStageResult:
        summary = self.react_agent.run(request_id=run_id, user_id=user_id, parse_result=parse_result)
        packet = build_evidence_packet(
            packet_id=f"packet_{run_id}",
            user_id=user_id,
            evidence_items=summary.collected_evidence,
        )
        return ResearchStageResult(summary=summary, evidence_packet=packet)
