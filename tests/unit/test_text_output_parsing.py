from __future__ import annotations

from datetime import date

from ai_trading_coach.modules.agent.text_output_parsing import parse_parser_output_text, parse_reporter_output_text, parse_research_output_text


def test_parse_parser_output_text_supports_json_without_ids() -> None:
    raw = '{"trade_actions":[{"action":"buy","target_asset":"AAPL"}],"judgements":[{"category":"asset_view","target":"AAPL","thesis":"AAPL will rise","evaluation_window":"1 week","dependencies":[]}]}'
    out = parse_parser_output_text(raw, run_id="r1", user_id="u1", run_date=date(2026, 1, 1), raw_log_text="log")
    assert len(out.trade_actions) == 1
    assert len(out.judgements) == 1
    assert "judgement_id" not in out.judgements[0].model_dump(mode="json")


def test_parse_research_output_returns_enriched_judgements_in_order() -> None:
    raw = '{"judgements":[{"category":"asset_view","target":"AAPL","thesis":"AAPL will rise","evaluation_window":"1 week","dependencies":[],"evidence":{"support_signal":"support","evidence_quality":"sufficient","evidence_summary":"Earnings beat","key_points":["Revenue acceleration"],"collected_evidence_items":[{"evidence_type":"news","summary":"Reuters report","related_tickers":["AAPL"],"sources":[{"provider":"Reuters","title":"Apple earnings","uri":"https://example.com"}]}]}}]}'
    parser_out = parse_parser_output_text('{"trade_actions":[],"judgements":[{"category":"asset_view","target":"AAPL","thesis":"AAPL will rise","evaluation_window":"1 week","dependencies":[]}]}', run_id="r1", user_id="u1", run_date=date(2026, 1, 1), raw_log_text="log")
    out = parse_research_output_text(raw, judgements=parser_out.judgements)
    assert len(out.judgements) == 1
    payload = out.model_dump(mode="json")
    assert "judgement_id" not in str(payload)
    assert "evidence_item_ids" not in str(payload)


def test_parse_reporter_output_extracts_feedback_by_section_order() -> None:
    raw = """
## Judgement 1
initial_feedback: likely_correct
evaluation_window: 1 week
Reasoning [来源: Reuters, 2026-03-01]
"""
    out = parse_reporter_output_text(raw, judgement_count=1)
    assert len(out.judgement_feedback) == 1
    assert out.judgement_feedback[0].initial_feedback == "likely_correct"
