from datetime import date

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.judgement_models import JudgementEvidence, ParserOutput, ResearchOutput, ResearchedJudgementItem, JudgementItem
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime


def test_verify_information_respects_stop_reason_thresholds() -> None:
    runtime = LangGraphNodeRuntime(None, None, None, None, None, object(), Settings(llm_provider_name="openai", openai_api_key="x", react_max_iterations=1, react_require_min_sources=2), None, None)
    state = {
        "research_retry_count": 0,
        "parse_result": ParserOutput(user_id="u", run_date=date(2026,1,1), judgements=[JudgementItem(category="asset_view", target="AAPL", thesis="up")]),
        "research_output": ResearchOutput(judgements=[ResearchedJudgementItem(category="asset_view", target="AAPL", thesis="up", evidence=JudgementEvidence())]),
        "evidence_packet": type("EP", (), {"source_registry": []})(),
    }
    out = runtime.verify_information_node(state)
    assert out["research_stop_reason"] == "max_iterations_reached"
    assert out["is_sufficient"] is False
