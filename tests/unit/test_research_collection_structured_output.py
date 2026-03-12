from ai_trading_coach.modules.agent.context_builder_v2 import ContextBuilderV2
from ai_trading_coach.domain.judgement_models import JudgementEvidence, JudgementItem, ParserOutput, ResearchOutput
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket, SourceAttribution
from ai_trading_coach.domain.enums import EvidenceType
from datetime import date


def test_report_context_contains_per_judgement_bundles() -> None:
    parse = ParserOutput(parse_id="p", user_id="u", run_date=date(2026,1,1), explicit_judgements=[JudgementItem(judgement_id="j1", category="asset_view", target_asset_or_topic="AAPL", thesis="x")])
    src = SourceAttribution(source_id="s1", source_type="mcp", provider="yfinance")
    item = EvidenceItem(item_id="e1", evidence_type=EvidenceType.NEWS, summary="news", sources=[src])
    packet = EvidencePacket(packet_id="p", user_id="u", news_evidence=[item], source_registry=[src])
    research = ResearchOutput(research_id="r", judgement_evidence=[JudgementEvidence(judgement_id="j1", evidence_item_ids=["e1"], support_signal="support", sufficiency_reason="ok")], stop_reason="done")
    ctx = ContextBuilderV2().for_reporter(parse_result=parse, research_output=research, evidence_packet=packet)
    assert ctx["judgement_bundles"][0]["allowed_source_ids"] == ["s1"]
