from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.llm_output_adapters import (
    judge_verdict_contract_to_domain,
    parser_contract_to_domain,
    reporter_contract_to_domain,
    research_synthesis_contract_to_domain,
)
from ai_trading_coach.domain.llm_output_contracts import (
    DailyJudgementFeedbackContract,
    JudgeVerdictContract,
    JudgementEvidenceContract,
    JudgementItemContract,
    ParserOutputContract,
    ReporterOutputContract,
    ResearchSynthesisOutputContract,
    TradeActionContract,
)


def test_parser_adapter_restores_optional_strings_and_preserves_lists_and_enums() -> None:
    contract = ParserOutputContract(
        parse_id="p1",
        user_id="u1",
        run_date=date(2026, 1, 1),
        trade_actions=[
            TradeActionContract(
                action="buy",
                target_asset="SPY",
                position_change="",
                action_time="",
                reason="",
            )
        ],
        explicit_judgements=[
            JudgementItemContract(
                judgement_id="j1",
                category="market_view",
                target_asset_or_topic="SPX",
                thesis="SPX up",
                confidence=0.7,
                evidence_from_user_log=["SPX looks strong"],
                implicitness="explicit",
                related_actions=["a1"],
                related_non_actions=[],
                estimated_horizon="",
                proposed_evaluation_window="1 week",
            )
        ],
        implicit_judgements=[],
        opportunity_judgements=[],
        non_action_judgements=[],
        reflection_summary=[],
    )

    domain = parser_contract_to_domain(contract)
    assert domain.trade_actions[0].position_change is None
    assert domain.trade_actions[0].action_time is None
    assert domain.trade_actions[0].reason is None
    assert domain.explicit_judgements[0].estimated_horizon is None
    assert domain.explicit_judgements[0].proposed_evaluation_window == "1 week"
    assert domain.explicit_judgements[0].evidence_from_user_log == ["SPX looks strong"]


def test_research_and_reporter_adapters_keep_values_and_validate_domain() -> None:
    research_contract = ResearchSynthesisOutputContract(
        research_id="r1",
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
    research_domain = research_synthesis_contract_to_domain(research_contract)
    assert research_domain.judgement_evidence[0].support_signal == "support"

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
    assert reporter_domain.judgement_feedback[0].source_ids == ["s1"]


def test_judge_verdict_adapter_restores_empty_rewrite_instruction_to_none() -> None:
    verdict_contract = JudgeVerdictContract(
        passed=False,
        reasons=["needs rewrite"],
        rewrite_instruction="",
        contradiction_flags=[],
        citation_coverage=0.5,
    )
    verdict_domain = judge_verdict_contract_to_domain(verdict_contract)
    assert verdict_domain.rewrite_instruction is None
    assert verdict_domain.citation_coverage == 0.5
