from __future__ import annotations

from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, SourceAttribution
from ai_trading_coach.modules.agent.langchain_tools import MCPToolInput, MCPToolRuntime, _execute_tool_async


class _Manager:
    def resolve_tool(self, _evidence_type):
        return type("ToolRef", (), {"server_id": "s", "tool_name": "t", "key": "srv.t"})()

    async def call_tool(self, **_kwargs):
        return {
            "items": [
                {
                    "item_id": "e1",
                    "summary": "price moved up on earnings",
                    "title": "Earnings beat",
                    "source_ids": ["src1"],
                }
            ]
        }


def test_tool_observation_contains_compact_evidence_fields(monkeypatch) -> None:
    from ai_trading_coach.modules.agent import langchain_tools as mod

    monkeypatch.setattr(
        mod,
        "normalize_tool_output",
        lambda **_: [
            EvidenceItem(
                item_id="e1",
                evidence_type=EvidenceType.NEWS,
                summary="price moved up on earnings",
                title="Earnings beat",
                sources=[SourceAttribution(source_id="src1", source_type="news", provider="p", title="t")],
                related_tickers=["AAPL"],
                data={"close": 100, "foo": "bar"},
            )
        ],
    )

    text = __import__("asyncio").run(
        _execute_tool_async(
            action_name="search_news",
            evidence_type=EvidenceType.NEWS,
            validated=MCPToolInput(judgement_ids=["j1"]),
            mcp_manager=_Manager(),
            runtime=MCPToolRuntime(),
        )
    )

    assert "item_id=e1" in text
    assert "summary=price moved up on earnings" in text
    assert "source_ids=['src1']" in text
    assert "judgement_ids=['j1']" in text
