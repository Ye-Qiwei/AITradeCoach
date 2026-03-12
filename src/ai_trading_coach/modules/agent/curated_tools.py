"""Single agent-facing curated tool registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ai_trading_coach.domain.enums import EvidenceType


@dataclass(frozen=True)
class CuratedToolDefinition:
    canonical_name: str
    description: str
    when_to_use: str
    when_not_to_use: str
    input_schema: dict[str, Any]
    output_summary_style: str
    tags: tuple[str, ...]
    implementation_kind: Literal["local_python", "external_mcp"]
    implementation_ref: str
    evidence_type: EvidenceType | None = None
    enabled: bool = True


CURATED_TOOL_REGISTRY: tuple[CuratedToolDefinition, ...] = (
    CuratedToolDefinition(
        canonical_name="yahoo_finance_price_history",
        description="Get ticker historical prices and summarize date range/latest close.",
        when_to_use="Need daily/weekly price path evidence for a ticker.",
        when_not_to_use="Need Japanese fund NAV history; use yahoo_japan_fund_history instead.",
        input_schema={"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}, "query": {"type": "object"}, "time_window": {"type": "string"}}},
        output_summary_style="timeseries_summary",
        tags=("price", "timeseries", "yfinance"),
        implementation_kind="external_mcp",
        implementation_ref="yfinance:yfinance_get_price_history",
        evidence_type=EvidenceType.PRICE_PATH,
    ),
    CuratedToolDefinition(
        canonical_name="yahoo_finance_ticker_news",
        description="Get latest Yahoo Finance ticker news.",
        when_to_use="Need recent company/market news for ticker thesis validation.",
        when_not_to_use="Need broad web search outside finance context.",
        input_schema={"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}}},
        output_summary_style="news_summary",
        tags=("news", "yfinance"),
        implementation_kind="external_mcp",
        implementation_ref="yfinance:yfinance_get_ticker_news",
        evidence_type=EvidenceType.NEWS,
    ),
    CuratedToolDefinition(
        canonical_name="yahoo_japan_fund_history",
        description="Fetch Yahoo Japan fund NAV history by fund code or direct URL.",
        when_to_use="Need Japanese mutual fund historical NAV rows.",
        when_not_to_use="Need stock ticker prices.",
        input_schema={"type": "object", "properties": {"fund_code": {"type": "string"}, "url": {"type": "string"}, "max_pages": {"type": "integer"}}},
        output_summary_style="fund_history_summary",
        tags=("fund", "japan", "nav"),
        implementation_kind="local_python",
        implementation_ref="ai_trading_coach.modules.data_sources.yahoo_japan_fund_history:get_fund_history",
        evidence_type=EvidenceType.PRICE_PATH,
    ),
)


def enabled_curated_tools() -> list[CuratedToolDefinition]:
    return [item for item in CURATED_TOOL_REGISTRY if item.enabled]
