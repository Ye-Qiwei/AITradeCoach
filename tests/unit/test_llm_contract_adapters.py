from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.id_generation import make_judgement_id
from ai_trading_coach.domain.llm_output_adapters import (
    judge_verdict_contract_to_domain,
    parser_contract_to_domain,
    reporter_contract_to_domain,
    research_agent_contract_to_domain,
)
from ai_trading_coach.domain.llm_output_contracts import (
    DailyJudgementFeedbackContract,
    JudgeVerdictContract,
    JudgementItemContract,
    ParserOutputContract,
    ReporterOutputContract,
    ResearchAgentFinalContract,
    JudgementEvidenceContract,
    TradeActionContract,
)


def test_parser_adapter_generates_ids_and_clears_invalid_related_actions() -> None:
    contract = ParserOutputContract(
        user_id="u1",
        run_date=date(2026, 1, 1),
        trade_actions=[TradeActionContract(action="buy", target_asset="SPY", position_change="", action_time="", reason="")],
        explicit_judgements=[
            JudgementItemContract(
                category="market_view",
                target_asset_or_topic="SPX",
                thesis="SPX up",
                confidence=0.7,
                evidence_from_user_log=["SPX looks strong"],
                implicitness="explicit",
                related_actions=["invalid_free_text"],
                related_non_actions=[],
                estimated_horizon="",
                proposed_evaluation_window="1 week",
                atomic_judgements=[{
                    "id": "a1",
                    "core_thesis": "SPX up",
                    "evaluation_timeframe": "1 week",
                    "dependencies": [],
                }],
            )
        ],
        implicit_judgements=[],
        opportunity_judgements=[],
        non_action_judgements=[],
        reflection_summary=[],
    )

    domain = parser_contract_to_domain(contract, run_id="r1", raw_log_text="hello")
    assert domain.parse_id.startswith("parse_")
    assert domain.trade_actions[0].action_id.startswith("act_")
    assert domain.explicit_judgements[0].judgement_id == make_judgement_id("r1", "explicit", 1, "market_view", "SPX", "SPX up")
    assert domain.explicit_judgements[0].related_actions == []


def test_research_and_reporter_adapters_keep_values_and_validate_domain() -> None:
    research_contract = ResearchAgentFinalContract(
        judgement_evidence=[
            JudgementEvidenceContract(
                judgement_id="j1",
                evidence_item_ids=["e1"],
                support_signal="support",
                sufficiency_reason="Direct support",
            )
        ],
        stop_reason="done",
    )
    research_domain = research_agent_contract_to_domain(research_contract, run_id="run-x")
    assert research_domain.research_id.startswith("research_")

    from ai_trading_coach.domain.judgement_models import ResearchOutput

    ResearchOutput.model_validate(research_domain.model_dump(mode="json")).validate_against({"j1"}, {"e1"})

    reporter_contract = ReporterOutputContract(
        markdown="- Observation [source:s1]\n\nSufficient detail for minimum length.",
        judgement_feedback=[
            DailyJudgementFeedbackContract(
                judgement_id="j1",
                initial_feedback="likely_correct",
                evidence_summary="Supported",
                evaluation_window="1 week",
                window_rationale="Need a week for confirmation",
                followup_indicators=["SPX close"],
                source_ids=["s1"],
            )
        ],
    )
    reporter_domain = reporter_contract_to_domain(reporter_contract)
    assert reporter_domain.judgement_feedback[0].evaluation_window == "1 week"


def test_judge_verdict_adapter_restores_empty_rewrite_instruction_to_none() -> None:
    verdict_contract = JudgeVerdictContract(
        passed=False,
        reasons=["needs rewrite"],
        rewrite_instruction="",
        contradiction_flags=[],
    )
    verdict_domain = judge_verdict_contract_to_domain(verdict_contract)
    assert verdict_domain.rewrite_instruction is None
