from __future__ import annotations

import asyncio
from time import perf_counter

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import Plan, PlanSubTask
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.modules.agent.executor_engine import ExecutorEngine
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager


def test_mcp_allowlist_blocked_call_is_traced() -> None:
    settings = Settings(
        _env_file=None,
        atc_mcp_servers='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        atc_evidence_tool_map='{"price_path":"srv:price_history"}',
        atc_mcp_tool_allowlist="srv:another_tool",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del server_id, tool_name, arguments
        return []

    manager = MCPClientManager(settings=settings, invoker=invoker)
    engine = ExecutorEngine(mcp_manager=manager)
    plan = Plan(
        plan_id="p1",
        subtasks=[
            PlanSubTask(
                subtask_id="s1",
                objective="Fetch price history",
                tool_category="market_data",
                evidence_type=EvidenceType.PRICE_PATH,
                query={"ticker": "AAPL.US"},
                tickers=["AAPL.US"],
            )
        ],
    )
    result = engine.execute(plan=plan, user_id="u1")
    assert result.tool_traces
    assert result.tool_traces[0].success is False
    assert "allowlist" in (result.tool_traces[0].error_message or "").lower()
    assert result.tool_traces[0].payload_hash


def test_executor_runs_independent_subtasks_in_parallel_with_traces() -> None:
    settings = Settings(
        _env_file=None,
        atc_mcp_servers='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        atc_evidence_tool_map='{"price_path":"srv:price_history","news":"srv:rss_search"}',
        atc_mcp_tool_allowlist="srv:price_history,srv:rss_search",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del server_id
        await asyncio.sleep(0.2)
        ticker = arguments.get("tickers", ["AAPL.US"])[0]
        return [{"title": f"{tool_name} for {ticker}", "ticker": ticker, "price": 101.5}]

    manager = MCPClientManager(settings=settings, invoker=invoker)
    engine = ExecutorEngine(mcp_manager=manager)
    plan = Plan(
        plan_id="p2",
        subtasks=[
            PlanSubTask(
                subtask_id="s_price",
                objective="Fetch price",
                tool_category="market_data",
                evidence_type=EvidenceType.PRICE_PATH,
                query={"window": "5d"},
                tickers=["AAPL.US"],
            ),
            PlanSubTask(
                subtask_id="s_news",
                objective="Fetch news",
                tool_category="news_search",
                evidence_type=EvidenceType.NEWS,
                query={"q": "AAPL"},
                tickers=["AAPL.US"],
            ),
        ],
    )

    t0 = perf_counter()
    result = engine.execute(plan=plan, user_id="u1")
    elapsed = perf_counter() - t0

    assert elapsed < 0.38
    assert len(result.tool_traces) == 2
    assert len(result.subtask_traces) == 2
    assert all(item.success for item in result.subtask_traces)
    assert result.evidence_packet.completeness_score == 1.0

