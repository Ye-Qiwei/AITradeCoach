from __future__ import annotations

from ai_trading_coach.modules.mcp.japan_fund_mcp_server import _extract_links, _parse_history_table


def test_extract_links_resolves_relative_urls() -> None:
    html_doc = (
        '<html><body>'
        '<a href="/docs/prospectus.pdf">目論見書</a>'
        '<a href="https://example.com/monthly">月次レポート</a>'
        "</body></html>"
    )
    links = _extract_links(html_doc, "https://fund.example.com/base/index.html")
    assert links[0].url == "https://fund.example.com/docs/prospectus.pdf"
    assert links[1].url == "https://example.com/monthly"


def test_parse_history_table_extracts_date_and_price_rows() -> None:
    html_doc = (
        '<table>'
        '<tr><th>日付</th><th>基準価額</th></tr>'
        '<tr><td>2025/03/01</td><td>10,123</td></tr>'
        '<tr><td>2025/02/28</td><td>10,115</td></tr>'
        '</table>'
    )
    rows = _parse_history_table(html_doc)
    assert rows == [
        {"date": "2025/03/01", "price": "10,123"},
        {"date": "2025/02/28", "price": "10,115"},
    ]
