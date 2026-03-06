"""MCP server for collecting Japanese mutual fund information.

Run:
    python -m ai_trading_coach.modules.mcp.japan_fund_mcp_server
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

GOOGLE_SEARCH_URL = "https://www.google.com/search"
YAHOO_FINANCE_JP = "https://finance.yahoo.co.jp"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

FUND_DOC_KEYWORDS = {
    "prospectus": ["目論見書", "交付目論見書", "prospectus"],
    "monthly_report": ["月次レポート", "monthly report", "運用レポート"],
    "operation_report": ["交付運用報告書", "運用報告書", "annual report"],
    "historical_nav": ["基準価額", "価格推移", "チャート", "history"],
}


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str


def _fetch_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def _clean_text(raw: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_links(raw_html: str, base_url: str) -> list[SearchResult]:
    links: list[SearchResult] = []
    for match in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", raw_html, flags=re.DOTALL):
        href = html.unescape(match.group(1)).strip()
        anchor = _clean_text(match.group(2))
        if not href:
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        links.append(SearchResult(title=anchor, url=absolute))
    return links


def _google_search(query: str, limit: int = 10) -> list[SearchResult]:
    params = urllib.parse.urlencode({"q": query, "hl": "ja", "num": min(max(limit, 1), 20)})
    url = f"{GOOGLE_SEARCH_URL}?{params}"
    raw = _fetch_text(url)
    results: list[SearchResult] = []

    pattern = re.compile(r'<a href="/url\?q=([^"&]+)[^"]*"[^>]*>(.*?)</a>', re.DOTALL)
    for match in pattern.finditer(raw):
        target = urllib.parse.unquote(html.unescape(match.group(1)))
        title = _clean_text(match.group(2))
        if not target.startswith("http"):
            continue
        if "google.com" in target:
            continue
        results.append(SearchResult(title=title, url=target))
        if len(results) >= limit:
            break
    return results


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def _find_documents(page_url: str) -> dict[str, list[dict[str, str]]]:
    raw = _fetch_text(page_url)
    links = _extract_links(raw, page_url)

    grouped: dict[str, list[dict[str, str]]] = {k: [] for k in FUND_DOC_KEYWORDS}
    for link in links:
        if not link.title and not link.url:
            continue
        target_text = f"{link.title} {link.url}"
        for key, keywords in FUND_DOC_KEYWORDS.items():
            if _contains_any(target_text, keywords):
                grouped[key].append({"title": link.title, "url": link.url})

    deduped: dict[str, list[dict[str, str]]] = {}
    for key, items in grouped.items():
        seen: set[str] = set()
        unique_items: list[dict[str, str]] = []
        for item in items:
            link_url = item.get("url", "")
            if link_url in seen:
                continue
            seen.add(link_url)
            unique_items.append(item)
        deduped[key] = unique_items[:10]
    return deduped


def _discover_yahoo_quote_url(fund_name: str) -> str | None:
    query = f"{fund_name} finance.yahoo.co.jp 投資信託"
    for result in _google_search(query=query, limit=8):
        if "finance.yahoo.co.jp" not in result.url:
            continue
        if "/quote/" in result.url:
            return result.url.split("?")[0].rstrip("/")
    return None


def _parse_history_table(raw_html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
    for row_match in row_pattern.finditer(raw_html):
        cells = [_clean_text(c) for c in cell_pattern.findall(row_match.group(1))]
        if len(cells) < 2:
            continue
        date_text = cells[0]
        nav_text = cells[1]
        if not re.search(r"\d{4}", date_text):
            continue
        if not re.search(r"\d", nav_text):
            continue
        rows.append({"date": date_text, "price": nav_text})
    return rows[:100]


def _fetch_yahoo_history(quote_url: str) -> tuple[str, list[dict[str, str]]]:
    normalized = quote_url.rstrip("/")
    history_url = f"{normalized}/history"
    raw = _fetch_text(history_url)
    rows = _parse_history_table(raw)
    return history_url, rows


def _fetch_yahoo_forum_comments(quote_url: str) -> tuple[str, list[str]]:
    forum_url = f"{quote_url.rstrip('/')}/bbs"
    raw = _fetch_text(forum_url)

    comments: list[str] = []
    for snippet in re.findall(r"<p[^>]*>(.*?)</p>", raw, flags=re.DOTALL | re.IGNORECASE):
        cleaned = _clean_text(snippet)
        if len(cleaned) < 12:
            continue
        if any(token in cleaned for token in ["返信", "投稿", "Yahoo", "利用規約"]):
            continue
        comments.append(cleaned)
        if len(comments) >= 12:
            break

    if not comments:
        plain = _clean_text(raw)
        parts = re.split(r"[。\n]", plain)
        for part in parts:
            text = part.strip()
            if len(text) < 15:
                continue
            comments.append(text)
            if len(comments) >= 8:
                break

    return forum_url, comments


def collect_japan_fund_info(fund_name: str, search_limit: int = 8) -> dict[str, object]:
    query = f"{fund_name} 投資信託 詳細"
    search_results = _google_search(query=query, limit=search_limit)

    selected_detail: str | None = None
    documents: dict[str, list[dict[str, str]]] = {k: [] for k in FUND_DOC_KEYWORDS}
    for result in search_results:
        try:
            docs = _find_documents(result.url)
        except Exception:
            continue
        total_hits = sum(len(v) for v in docs.values())
        if total_hits == 0:
            continue
        selected_detail = result.url
        documents = docs
        break

    history_rows: list[dict[str, str]] = []
    history_url: str | None = None
    if documents.get("historical_nav"):
        for candidate in documents["historical_nav"]:
            url = candidate.get("url")
            if not url:
                continue
            try:
                raw = _fetch_text(url)
            except Exception:
                continue
            rows = _parse_history_table(raw)
            if rows:
                history_rows = rows
                history_url = url
                break

    yahoo_quote_url = _discover_yahoo_quote_url(fund_name)
    forum_url: str | None = None
    forum_comments: list[str] = []

    if yahoo_quote_url:
        if not history_rows:
            try:
                history_url, history_rows = _fetch_yahoo_history(yahoo_quote_url)
            except Exception:
                history_url = None
                history_rows = []

        try:
            forum_url, forum_comments = _fetch_yahoo_forum_comments(yahoo_quote_url)
        except Exception:
            forum_url = None
            forum_comments = []

    return {
        "fund_name": fund_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "search_query": query,
        "detail_page": selected_detail,
        "documents": documents,
        "historical_nav": {
            "required_found": bool(history_rows),
            "source_url": history_url,
            "rows": history_rows,
        },
        "yahoo_finance": {
            "quote_url": yahoo_quote_url,
            "forum_url": forum_url,
            "forum_comments": forum_comments,
        },
        "search_results": [{"title": item.title, "url": item.url} for item in search_results],
    }


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install 'mcp' package to run japan_fund_mcp_server.") from exc

    mcp = FastMCP("japan_fund_search")

    @mcp.tool()
    def fund_intel_search(fund_name: str, search_limit: int = 8) -> str:
        """Collect Japanese mutual-fund documents, NAV history, and Yahoo forum comments."""

        payload = collect_japan_fund_info(fund_name=fund_name, search_limit=search_limit)
        return json.dumps(payload, ensure_ascii=False)

    mcp.run()


if __name__ == "__main__":
    main()
