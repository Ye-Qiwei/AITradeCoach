from __future__ import annotations

from ai_trading_coach.domain.contracts import MCPGatewayInput
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceNeed, EvidencePlan
from ai_trading_coach.modules.mcp.service import DefaultMCPToolGateway


def test_mcp_gateway_collects_mock_evidence() -> None:
    gateway = DefaultMCPToolGateway()
    plan = EvidencePlan(
        plan_id="plan_1",
        user_id="u1",
        needs=[
            EvidenceNeed(
                need_id="need_1",
                claim="验证短线事件反应",
                evidence_types=[EvidenceType.PRICE_PATH, EvidenceType.NEWS],
                tickers=["9660.HK"],
            )
        ],
    )

    out = gateway.collect(MCPGatewayInput(plan=plan))
    assert out.packet.price_evidence
    assert out.packet.news_evidence
    assert out.packet.completeness_score == 1.0
    traces = out.packet.extensions.get("tool_call_traces")
    assert isinstance(traces, list)
    assert traces
