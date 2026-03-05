"""MCP tool gateway with adapter abstraction and normalized evidence output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.contracts import MCPGatewayInput, MCPGatewayOutput
from ai_trading_coach.domain.enums import EvidenceType
from ai_trading_coach.domain.models import EvidenceItem, EvidencePacket, SourceAttribution, ToolCallTrace


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MCPServerAdapter(Protocol):
    """Adapter contract for a concrete MCP server."""

    adapter_id: str

    def fetch(self, need_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return normalized raw payloads for one evidence need."""


@dataclass
class MockMCPServerAdapter:
    """Local mock adapter to keep gateway runnable before external server integration."""

    adapter_id: str

    def fetch(self, need_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        evidence_type = payload.get("evidence_type")
        claim = payload.get("claim", "")
        tickers = payload.get("tickers", [])

        if evidence_type == EvidenceType.PRICE_PATH.value:
            return [
                {
                    "summary": f"Mock price path evidence for need={need_id}",
                    "data": {"claim": claim, "return_5d": None, "return_20d": None},
                    "related_tickers": tickers,
                }
            ]

        if evidence_type in {EvidenceType.NEWS.value, EvidenceType.FILING.value}:
            return [
                {
                    "summary": f"Mock event evidence for need={need_id}",
                    "data": {"claim": claim, "event_count": 0},
                    "related_tickers": tickers,
                }
            ]

        if evidence_type == EvidenceType.MACRO.value:
            return [
                {
                    "summary": f"Mock macro evidence for need={need_id}",
                    "data": {"claim": claim, "macro_signal": "neutral"},
                    "related_tickers": tickers,
                }
            ]

        return []


@dataclass
class HttpJsonMCPServerAdapter:
    """HTTP JSON adapter for real MCP servers."""

    adapter_id: str
    endpoint: str
    timeout_seconds: float = 12.0

    def fetch(self, need_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json={"need_id": need_id, "payload": payload})
            response.raise_for_status()
            body = response.json()

        if isinstance(body, list):
            return [item for item in body if isinstance(item, dict)]
        if isinstance(body, dict):
            items = body.get("items", [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []


class DefaultMCPToolGateway:
    """Collect evidence from multiple MCP adapters and normalize into EvidencePacket."""

    def __init__(self, adapters: dict[EvidenceType, list[MCPServerAdapter]] | None = None) -> None:
        settings = get_settings()
        self.retry_attempts = max(1, settings.mcp_max_retries + 1)
        self.adapters = adapters or self._build_default_adapters()

    def collect(self, data: MCPGatewayInput) -> MCPGatewayOutput:
        packet = EvidencePacket(
            packet_id=f"ep_{data.plan.plan_id}",
            user_id=data.plan.user_id,
        )

        missing: list[str] = []
        satisfied_needs = 0
        total_needs = len(data.plan.needs)
        tool_call_traces: list[ToolCallTrace] = []

        for need in data.plan.needs:
            need_satisfied = False
            for evidence_type in need.evidence_types:
                adapters = self.adapters.get(evidence_type, [])
                if not adapters:
                    missing.append(f"{need.need_id}:{evidence_type.value}:no_adapter")
                    continue

                evidence_items_from_type: list[EvidenceItem] = []
                for adapter in adapters:
                    payload = {
                        "claim": need.claim,
                        "tickers": need.tickers,
                        "indexes": need.indexes,
                        "sectors": need.sectors,
                        "macro_variables": need.macro_variables,
                        "questions": need.questions,
                        "evidence_type": evidence_type.value,
                    }
                    rows, call_trace = self._fetch_with_trace(adapter, need.need_id, payload)
                    tool_call_traces.append(call_trace)
                    if not rows:
                        continue
                    need_satisfied = True
                    evidence_items_from_type.extend(
                        self._rows_to_items(rows, need.need_id, evidence_type, adapter.adapter_id)
                    )

                if evidence_items_from_type:
                    cross_check_count = len({item.sources[0].provider for item in evidence_items_from_type if item.sources})
                    for item in evidence_items_from_type:
                        item.data["cross_check_count"] = cross_check_count
                    self._append_items(packet, evidence_type, evidence_items_from_type)

            if need_satisfied:
                satisfied_needs += 1
            else:
                missing.append(f"{need.need_id}:no_evidence")

        packet.missing_requirements = missing
        packet.completeness_score = 0.0 if total_needs == 0 else round(satisfied_needs / total_needs, 4)
        packet.extensions["tool_call_traces"] = [trace.model_dump() for trace in tool_call_traces]
        return MCPGatewayOutput(packet=packet)

    def _fetch_with_trace(
        self,
        adapter: MCPServerAdapter,
        need_id: str,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], ToolCallTrace]:
        started = utc_now()
        t0 = perf_counter()
        success = True
        error_message: str | None = None
        rows: list[dict[str, Any]] = []

        try:
            rows = self._fetch_with_retry(adapter, need_id, payload)
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)

        latency_ms = int((perf_counter() - t0) * 1000)
        call_trace = ToolCallTrace(
            call_id=f"tool_{adapter.adapter_id}_{need_id}_{int(started.timestamp() * 1000)}",
            tool_name="mcp_fetch",
            server_id=adapter.adapter_id,
            request_summary=f"type={payload.get('evidence_type')} tickers={payload.get('tickers', [])}",
            response_summary=f"rows={len(rows)}",
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
        )
        return rows, call_trace

    def _fetch_with_retry(
        self,
        adapter: MCPServerAdapter,
        need_id: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        @retry(stop=stop_after_attempt(self.retry_attempts), wait=wait_fixed(0.2), reraise=True)
        def _wrapped() -> list[dict[str, Any]]:
            return adapter.fetch(need_id, payload)

        return _wrapped()

    def _rows_to_items(
        self,
        rows: list[dict[str, Any]],
        need_id: str,
        evidence_type: EvidenceType,
        adapter_id: str,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for idx, row in enumerate(rows):
            source = SourceAttribution(
                source_id=f"src_{adapter_id}_{need_id}_{idx}",
                source_type="mcp",
                provider=adapter_id,
                uri=row.get("uri"),
                title=row.get("title"),
                published_at=row.get("published_at"),
                fetched_at=utc_now(),
                reliability_score=0.5,
            )
            item = EvidenceItem(
                item_id=f"ev_{need_id}_{adapter_id}_{idx}",
                evidence_type=evidence_type,
                summary=str(row.get("summary", "")),
                data=row.get("data", {}),
                related_tickers=row.get("related_tickers", []),
                event_time=row.get("event_time"),
                sources=[source],
            )
            items.append(item)
        return items

    def _append_items(
        self,
        packet: EvidencePacket,
        evidence_type: EvidenceType,
        items: list[EvidenceItem],
    ) -> None:
        if not items:
            return

        for item in items:
            packet.source_registry.extend(item.sources)

        if evidence_type == EvidenceType.PRICE_PATH:
            packet.price_evidence.extend(items)
        elif evidence_type == EvidenceType.NEWS:
            packet.news_evidence.extend(items)
        elif evidence_type == EvidenceType.FILING:
            packet.filing_evidence.extend(items)
        elif evidence_type == EvidenceType.SENTIMENT:
            packet.sentiment_evidence.extend(items)
        elif evidence_type == EvidenceType.DISCUSSION:
            packet.discussion_evidence.extend(items)
        elif evidence_type == EvidenceType.MACRO:
            packet.macro_evidence.extend(items)
        elif evidence_type == EvidenceType.ANALOG_HISTORY:
            packet.analog_evidence.extend(items)
        else:
            packet.market_regime_evidence.extend(items)

    def _build_default_adapters(self) -> dict[EvidenceType, list[MCPServerAdapter]]:
        adapters: dict[EvidenceType, list[MCPServerAdapter]] = {}
        for evidence_type in EvidenceType:
            adapters[evidence_type] = [MockMCPServerAdapter(adapter_id=f"mock_{evidence_type.value}")]
        return adapters


# Backward-compatible alias
PlaceholderMCPToolGateway = DefaultMCPToolGateway
