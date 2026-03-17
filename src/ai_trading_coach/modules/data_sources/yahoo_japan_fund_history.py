from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any



@dataclass
class FundHistoryRow:
    date: str
    nav: int | None
    day_change: int | None
    net_assets_million_jpy: int | None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    s = text.replace(",", "").replace("円", "").replace("百万", "").replace("％", "").strip()
    if s in {"", "-", "—", "―", "–", "---"}:
        return None
    m = re.search(r"(-?[\d.]+)", s)
    return int(float(m.group(1))) if m else None


def _parse_date(text: str | None) -> str:
    if not text:
        return ""
    s = _clean_text(text).replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        return s
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _extract_rows_from_table(table: Any) -> list[FundHistoryRow]:
    rows: list[FundHistoryRow] = []
    if table is None:
        return rows
    target = table.find("tbody") or table
    for tr in target.find_all("tr"):
        cells = [_clean_text(c.get_text()) for c in tr.find_all("td")]
        if len(cells) < 2:
            continue
        date_val = _parse_date(cells[0])
        if not re.match(r"\d{4}-\d{2}-\d{2}", date_val):
            continue
        rows.append(FundHistoryRow(date=date_val, nav=_parse_int(cells[1]), day_change=_parse_int(cells[2]) if len(cells) > 2 else None, net_assets_million_jpy=_parse_int(cells[3]) if len(cells) > 3 else None))
    return rows


async def get_fund_history_by_url(url: str, max_pages: int = 3) -> dict[str, Any]:
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright
    by_date: dict[str, FundHistoryRow] = {}
    debug: list[dict[str, Any]] = []
    fund_name: str | None = None
    current_page = 1
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo")
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector("th:has-text('基準価額'), th:has-text('日付')", timeout=15000)
            header = page.locator("h1").first
            if await header.count() > 0:
                fund_name = re.sub(r"【.*?】", "", (await header.inner_text())).strip()
            while current_page <= max_pages:
                soup = BeautifulSoup(await page.content(), "html.parser")
                table = soup.find("table", {"data-it-fund-history-table": True})
                if not table:
                    for candidate in soup.find_all("table"):
                        txt = candidate.get_text()
                        if "基準価額" in txt and "日付" in txt:
                            table = candidate
                            break
                rows = _extract_rows_from_table(table)
                if not rows:
                    debug.append({"page": current_page, "warning": "no_rows"})
                    break
                for row in rows:
                    by_date[row.date] = row
                debug.append({"page": current_page, "extracted_rows": len(rows)})
                next_btn = page.locator("a:has-text('次へ'), button:has-text('次へ')").last
                if await next_btn.count() == 0 or not await next_btn.is_visible():
                    break
                await next_btn.click()
                await page.wait_for_timeout(1200)
                current_page += 1
        except Exception as exc:  # noqa: BLE001
            debug.append({"fatal_error": str(exc)})
        finally:
            await browser.close()

    rows_sorted = sorted(by_date.values(), key=lambda row: row.date, reverse=True)
    code_match = re.search(r"quote/([^/]+)", url)
    return {
        "rows": [asdict(item) for item in rows_sorted],
        "fund_name": fund_name,
        "fund_code": code_match.group(1) if code_match else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows_sorted),
        "debug": debug,
    }


async def get_fund_history_by_code(fund_code: str, max_pages: int = 3) -> dict[str, Any]:
    return await get_fund_history_by_url(f"https://finance.yahoo.co.jp/quote/{fund_code}/history", max_pages=max_pages)
