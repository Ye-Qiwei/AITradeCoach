from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.judgement_models import (
    DailyJudgementFeedback,
    JudgementEvidence,
    JudgementItem,
    ParserOutput,
    ResearchOutput,
)
from ai_trading_coach.domain.models import EvidencePacket, ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


class DummyStore:
    def __init__(self) -> None:
        self.called = False

    def upsert_records(self, records):
        self.called = True


def _runtime(max_rewrites: int, store: DummyStore) -> LangGraphNodeRuntime:
    return LangGraphNodeRuntime(
        parser_agent=None,
        reporter_agent=None,
        report_judge=None,
        context_builder=None,
        mcp_manager=None,
        chat_model=None,
        settings=SimpleNamespace(agent_max_rewrite_rounds=max_rewrites, prompt_version="v", llm_provider=lambda: "openai", selected_llm_model=lambda: "m"),
        long_term_store=store,
        llm_gateway=None,
        prompt_manager=None,
    )


def test_rewrite_round_semantics_allow_expected_extra_rewrites() -> None:
    runtime = _runtime(max_rewrites=2, store=DummyStore())
    assert runtime.route_after_judge({"judge_verdict": JudgeVerdict(passed=False), "rewrite_count": 1}) == "rewrite"
    assert runtime.route_after_judge({"judge_verdict": JudgeVerdict(passed=False), "rewrite_count": 2}) == "rewrite"
    assert runtime.route_after_judge({"judge_verdict": JudgeVerdict(passed=False), "rewrite_count": 3}) == "fail"


def test_finalize_result_respects_dry_run_memory_write() -> None:
    store = DummyStore()
    runtime = _runtime(max_rewrites=0, store=store)
    req = ReviewRunRequest(
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 3, 1),
        trigger_type=TriggerType.MANUAL,
        raw_log_text="x",
        options={"dry_run": True},
    )
    parse = ParserOutput(
        parse_id="p1",
        user_id="u1",
        run_date=date(2026, 3, 1),
        explicit_judgements=[
            JudgementItem(
                judgement_id="j1",
                category="market_view",
                target_asset_or_topic="SPX",
                thesis="SPX up",
                evidence_from_user_log=["spx up"],
                proposed_evaluation_window="1 week",
            )
        ],
    )
    state = {
        "request": req,
        "report_draft": "- a [source:s1]",
        "judgement_feedback": [
            DailyJudgementFeedback(
                judgement_id="j1",
                initial_feedback="high_uncertainty",
                evidence_summary="n/a",
                evaluation_window="1 week",
                window_rationale="n/a",
                source_ids=["s1"],
            )
        ],
        "parse_result": parse,
        "research_output": ResearchOutput(research_id="rs", judgement_evidence=[JudgementEvidence(judgement_id="j1", evidence_item_ids=[], support_signal="uncertain", sufficiency_reason="no data")]),
        "evidence_packet": EvidencePacket(packet_id="ep", user_id="u1"),
        "model_calls": [],
        "tool_calls": [],
        "react_steps": [],
        "report_context": {},
        "rewrite_count": 0,
    }
    out = runtime.finalize_result(state)
    assert store.called is False
    assert out["final_result"].memory_write_results == []
