from __future__ import annotations

from datetime import date

from ai_trading_coach.modules.agent.text_output_parsing import (
    parse_parser_output_text,
    parse_reporter_output_text,
    parse_research_output_text,
)


def test_parse_parser_output_text_supports_json() -> None:
    raw = '{"trade_actions":[{"action":"buy","target_asset":"AAPL"}],"judgements":[{"local_id":"j1","category":"asset_view","target":"AAPL","thesis":"AAPL will rise","evaluation_window":"1 week","dependencies":[]}]}'
    out = parse_parser_output_text(raw, run_id="r1", user_id="u1", run_date=date(2026, 1, 1), raw_log_text="log")
    assert len(out.trade_actions) == 1
    assert len(out.judgements) == 1
    assert out.judgements[0].category == "asset_view"


def test_parse_parser_output_text_supports_markdown_sections() -> None:
    raw = """
# TRADE_ACTIONS
- action: sell
- target_asset: TSLA

# JUDGEMENTS
- local_id: j1
- category: market_view
- target: SPX
- thesis: SPX likely volatile
- evaluation_window: 1 month
- dependencies:
"""
    out = parse_parser_output_text(raw, run_id="r1", user_id="u1", run_date=date(2026, 1, 1), raw_log_text="log")
    assert len(out.trade_actions) == 1
    assert out.trade_actions[0].action == "sell"
    assert len(out.judgements) == 1


def test_parse_research_output_text_to_domain() -> None:
    raw = """
# JUDGEMENT_EVIDENCE
- judgement_id: j1
- evidence_item_ids: e1,e2
- support_signal: support
- evidence_quality: sufficient
"""
    out = parse_research_output_text(raw, run_id="r1")
    assert len(out.judgement_evidence) == 1
    assert out.judgement_evidence[0].evidence_item_ids == ["e1", "e2"]


def test_parse_reporter_output_extracts_feedback_from_markdown() -> None:
    raw = """
## Judgement 1
judgement_id: j1
initial_feedback: likely_correct
evaluation_window: 1 week
Reasoning [source:s1]
"""
    out = parse_reporter_output_text(raw)
    assert len(out.judgement_feedback) == 1
    assert out.judgement_feedback[0].judgement_id == "j1"
