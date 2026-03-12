from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright

mcp = FastMCP("yahoo_japan_fund_history")

@dataclass
class FundHistoryRow:
    date: str
    nav: int | None
    day_change: int | None
    net_assets_million_jpy: int | None

# ---------------------------------------------------------
# 文本清理和解析函数 (保持不变)
# ---------------------------------------------------------
def _clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()

def _parse_int(text: str | None) -> int | None:
    if not text: return None
    s = text.replace(",", "").replace("円", "").replace("百万", "").replace("％", "").strip()
    if s in {"", "-", "—", "―", "–", "---"}: return None
    m = re.search(r"(-?[\d.]+)", s)
    if m:
        try: return int(float(m.group(1)))
        except ValueError: return None
    return None

def _parse_date(text: str | None) -> str:
    if not text: return ""
    s = _clean_text(text)
    s = s.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return s

def _extract_rows_from_table(table: Any) -> list[FundHistoryRow]:
    rows: list[FundHistoryRow] = []
    tbody = table.find("tbody")
    target = tbody if tbody else table
    
    for tr in target.find_all("tr"):
        cols = tr.find_all("td")
        if len(cols) < 2: continue
            
        cells = [_clean_text(c.get_text()) for c in cols]
        date_val = _parse_date(cells[0])
        if not re.match(r"\d{4}-\d{2}-\d{2}", date_val):
            continue

        rows.append(FundHistoryRow(
            date=date_val,
            nav=_parse_int(cells[1]) if len(cells) > 1 else None,
            day_change=_parse_int(cells[2]) if len(cells) > 2 else None,
            net_assets_million_jpy=_parse_int(cells[3]) if len(cells) > 3 else None,
        ))
    return rows

# ---------------------------------------------------------
# 核心重构：使用 Playwright Async API
# ---------------------------------------------------------
async def _collect_history(url: str, max_pages: int = 3) -> dict[str, Any]:
    by_date: dict[str, FundHistoryRow] = {}
    fund_name: str | None = None
    debug_info: list[dict] = []
    current_page = 1

    async with async_playwright() as p:
        # 建议测试时可以将 headless=False，这样会弹出真实的浏览器窗口让你看到加载过程
        browser = await p.chromium.launch(headless=True)
        
        # 添加更多的伪装参数，降低被反爬识别的概率
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="ja-JP",
            timezone_id="Asia/Tokyo"
        )
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded")
            
            # 【策略更改】不再死等特定 data 属性的表格
            # 改为等待包含 "基準価額" 或 "日付" 文本的表头 (th) 出现
            try:
                await page.wait_for_selector("th:has-text('基準価額'), th:has-text('日付')", timeout=15000)
            except Exception as e:
                # 【杀手锏】如果超时，拍一张网页快照保存下来
                await page.screenshot(path="error_screenshot.png", full_page=True)
                raise Exception("等待表格超时，已将当前网页状态保存为 error_screenshot.png。可能遭遇了反爬虫拦截。") from e
            
            fund_name_elem = page.locator("h1").first
            if await fund_name_elem.count() > 0:
                fund_name_text = await fund_name_elem.inner_text()
                fund_name = re.sub(r"【.*?】", "", fund_name_text).strip()

            while True:
                if current_page > max_pages:
                    break
                    
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")
                
                # 【策略更改】增强 BeautifulSoup 寻找表格的鲁棒性
                table = soup.find("table", {"data-it-fund-history-table": True})
                if not table:
                    for t in soup.find_all("table"):
                        text = t.get_text()
                        if "基準価額" in text and "日付" in text:
                            table = t
                            break
                
                rows = _extract_rows_from_table(table) if table else []
                if not rows:
                    debug_info.append({"page": current_page, "warning": "当前页没有提取到有效行数据"})
                    break
                    
                for r in rows:
                    by_date[r.date] = r
                    
                debug_info.append({"page": current_page, "extracted_rows": len(rows)})
                
                next_button = page.locator("a:has-text('次へ'), button:has-text('次へ')").last
                if await next_button.count() == 0 or not await next_button.is_visible():
                    break
                    
                try:
                    await next_button.click()
                    await asyncio.sleep(2.0)  # 稍微增加一点等待时间，确保渲染完成
                except Exception as e:
                    debug_info.append({"error": f"翻页点击失败: {str(e)}"})
                    break
                    
                current_page += 1
                
        except Exception as e:
            debug_info.append({"fatal_error": str(e)})
        finally:
            await browser.close()

    rows_sorted = sorted(by_date.values(), key=lambda r: r.date, reverse=True)
    fund_code_match = re.search(r"quote/([^/]+)", url)
    
    return {
        "fund_name": fund_name,
        "fund_code": fund_code_match.group(1) if fund_code_match else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": [asdict(r) for r in rows_sorted],
        "row_count": len(rows_sorted),
        "pages_attempted": current_page if current_page <= max_pages else max_pages,
        "debug": debug_info
    }

# 注意：MCP 工具函数也需要加上 async
@mcp.tool()
async def fund_history_from_url(url: str, max_pages: int = 3) -> str:
    """从 Yahoo!ファイナンス日本抓取动态加载的基金历史基准价数据。"""
    payload = await _collect_history(url=url, max_pages=max_pages)
    return json.dumps(payload, ensure_ascii=False, indent=2)

@mcp.tool()
async def fund_history_from_code(fund_code: str, max_pages: int = 3) -> str:
    """通过基金 code 抓取动态历史数据。"""
    url = f"https://finance.yahoo.co.jp/quote/{fund_code}/history"
    payload = await _collect_history(url=url, max_pages=max_pages)
    return json.dumps(payload, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")