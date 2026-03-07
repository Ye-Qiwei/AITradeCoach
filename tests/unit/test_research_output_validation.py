from __future__ import annotations

import pytest

from ai_trading_coach.domain.judgement_models import JudgementEvidence, ResearchOutput


def test_research_output_validation_passes_with_full_mapping() -> None:
    output = ResearchOutput(
        research_id="r1",
        judgement_evidence=[
            JudgementEvidence(judgement_id="j1", evidence_item_ids=["e1"], support_signal="support", sufficiency_reason="Has direct price+news support."),
            JudgementEvidence(judgement_id="j2", evidence_item_ids=[], support_signal="uncertain", sufficiency_reason="No relevant filings found."),
        ],
        stop_reason="Completed",
    )
    output.validate_against({"j1", "j2"}, {"e1", "e2"})


def test_research_output_validation_rejects_unknown_evidence() -> None:
    output = ResearchOutput(
        research_id="r1",
        judgement_evidence=[
            JudgementEvidence(judgement_id="j1", evidence_item_ids=["missing"], support_signal="support", sufficiency_reason="reason"),
        ],
    )
    with pytest.raises(ValueError, match="Unknown evidence_item_ids"):
        output.validate_against({"j1"}, {"e1"})


def test_research_output_validation_rejects_missing_judgement() -> None:
    output = ResearchOutput(
        research_id="r1",
        judgement_evidence=[
            JudgementEvidence(judgement_id="j1", evidence_item_ids=["e1"], support_signal="support", sufficiency_reason="reason"),
        ],
    )
    with pytest.raises(ValueError, match="Missing judgements"):
        output.validate_against({"j1", "j2"}, {"e1"})
