from datetime import date

from ai_trading_coach.domain.judgement_models import JudgementEvidence, JudgementItem, ParserOutput, ResearchOutput, ResearchedJudgementItem
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.modules.agent.context_builder_v2 import ContextBuilderV2


def test_report_context_contains_ordered_judgement_bundles() -> None:
    parse = ParserOutput(user_id="u", run_date=date(2026, 1, 1), judgements=[JudgementItem(category="asset_view", target="AAPL", thesis="x")])
    research = ResearchOutput(
        judgements=[
            ResearchedJudgementItem(
                category="asset_view",
                target="AAPL",
                thesis="x",
                evidence=JudgementEvidence(evidence_summary="ok"),
            )
        ]
    )
    ctx = ContextBuilderV2().for_reporter(parse_result=parse, research_output=research, evidence_packet=EvidencePacket(packet_id="p", user_id="u"))
    assert ctx["judgement_bundles"][0]["target"] == "AAPL"
