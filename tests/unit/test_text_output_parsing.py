from __future__ import annotations

from datetime import date

import pytest

from ai_trading_coach.modules.agent.text_output_parsing import (
    build_research_output,
    parse_parser_output_text,
    parse_reporter_output_text,
    parse_research_output_text,
)


def test_parse_parser_output_text_from_markdown() -> None:
    raw = """
# Trade Actions
## Action 1
- action: buy
- target_asset: AAPL

# Judgements
## Judgement 1
- category: asset_view
- target: AAPL
- thesis: Apple can outperform on earnings momentum.
- evaluation_window: 1 week
- dependencies: none
"""
    out = parse_parser_output_text(raw, run_id="r1", user_id="u1", run_date=date(2026, 1, 1), raw_log_text="log")
    assert len(out.trade_actions) == 1
    assert len(out.judgements) == 1


def test_parse_research_output_and_build_research_models() -> None:
    parser_out = parse_parser_output_text(
        """
# Trade Actions

# Judgements
## Judgement 1
- category: asset_view
- target: AAPL
- thesis: AAPL can outperform.
- evaluation_window: 1 week
- dependencies: none
""",
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 1, 1),
        raw_log_text="log",
    )
    raw = """
# Judgement Evidence
## Judgement 1
- support_signal: support
- evidence_quality: sufficient
- cited_sources:
  - src_1
- rationale: Revenue trend supports the thesis.
"""
    parsed = parse_research_output_text(raw, judgements=parser_out.judgements)
    out = build_research_output(parsed_markdown=parsed, judgements=parser_out.judgements, cited_items=[[{"summary": "e1", "evidence_type": "news", "related_tickers": ["AAPL"], "sources": [{"provider": "reuters", "source_type": "news"}]}]])
    assert out.judgements[0].evidence.support_signal == "support"
    assert len(out.judgements[0].evidence.collected_evidence_items) == 1


def test_parse_reporter_output_from_summary_table() -> None:
    raw = """
# Daily Review
## Feedback Summary

| judgement_ref | initial_feedback | evaluation_window |
|---|---|---|
| Judgement 1 | likely_correct | 1 week |

## Detailed Analysis
### Judgement 1
Thesis is supported. [source:src_1]
"""
    out = parse_reporter_output_text(raw, judgement_count=1)
    assert len(out.judgement_feedback) == 1
    assert out.judgement_feedback[0].initial_feedback == "likely_correct"


def test_markdown_parsers_reject_json() -> None:
    with pytest.raises(ValueError, match="Expected markdown"):
        parse_parser_output_text('{"judgements":[]}', run_id="r", user_id="u", run_date=date(2026, 1, 1), raw_log_text="x")
