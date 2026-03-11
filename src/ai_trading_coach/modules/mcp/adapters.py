"""Normalization adapters from raw MCP tool outputs to EvidenceItem."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.models import EvidenceItem, SourceAttribution


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_tool_output(
    *,
    server_id: str,
    tool_name: str,
    subtask: PlanSubTask,
    raw_result: Any,
) -> list[EvidenceItem]:
    rows = _extract_rows(raw_result)
    out: list[EvidenceItem] = []
    for idx, row in enumerate(rows):
        uri = _lookup_text(row, "uri", "url", "link")
        title = _lookup_text(row, "title", "headline", "name")
        published_at = _to_datetime(
            _lookup_value(row, "published_at", "date", "datetime", "timestamp", "Date")
        )
        summary = _lookup_text(
            row,
            "summary",
            "snippet",
            "text",
            "title",
            "headline",
            "description",
        )
        source = SourceAttribution(
            source_id=f"src_{server_id}_{tool_name}_{subtask.subtask_id}_{idx}",
            source_type=subtask.tool_category,
            provider=server_id,
            uri=uri,
            title=title,
            published_at=published_at,
        )
        data = _safe_data(row)
        if not summary:
            summary = f"{subtask.evidence_type.value} evidence from {server_id}:{tool_name}"
        out.append(
            EvidenceItem(
                item_id=f"ev_{subtask.subtask_id}_{idx}",
                evidence_type=subtask.evidence_type,
                summary=summary[:400],
                data=data,
                related_tickers=_extract_tickers(row=row, subtask=subtask),
                event_time=_to_datetime(
                    _lookup_value(
                        row,
                        "event_time",
                        "published_at",
                        "date",
                        "datetime",
                        "timestamp",
                        "Date",
                    )
                ),
                sources=[source],
            )
        )
    return out


def _extract_rows(raw_result: Any) -> list[dict[str, Any]]:
    if isinstance(raw_result, list):
        return [item for item in raw_result if isinstance(item, dict)]
    if isinstance(raw_result, dict):
        if isinstance(raw_result.get("items"), list):
            return [item for item in raw_result["items"] if isinstance(item, dict)]
        return [raw_result]

    content = getattr(raw_result, "content", None)
    if isinstance(content, list):
        rows: list[dict[str, Any]] = []
        for chunk in content:
            text = _extract_text_chunk(chunk)
            if not text:
                continue
            loaded = _try_json(text)
            if isinstance(loaded, list):
                rows.extend(item for item in loaded if isinstance(item, dict))
            elif isinstance(loaded, dict):
                rows.append(loaded)
            else:
                rows.append({"text": text})
        return rows

    return [{"text": _to_text(raw_result)}]


def _extract_text_chunk(chunk: Any) -> str:
    if isinstance(chunk, dict):
        candidate = chunk.get("text")
        if isinstance(candidate, str):
            return candidate
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return _to_text(chunk)


def _safe_data(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "symbol",
        "ticker",
        "price",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "change_pct",
        "date",
        "window",
        "series_id",
        "value",
        "unit",
        "filing_type",
        "form",
        "accession_no",
        "company",
        "adj close",
    }
    out: dict[str, Any] = {}
    for key, value in row.items():
        canonical_key = _canonical_key(key)
        if canonical_key not in allowed:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[canonical_key.replace(" ", "_")] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            out[canonical_key.replace(" ", "_")] = value[:20]
    return out


def _extract_tickers(row: dict[str, Any], subtask: PlanSubTask) -> list[str]:
    candidates: list[str] = []
    for key in ("ticker", "symbol", "tickers"):
        value = _lookup_value(row, key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip().upper())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip().upper())

    if not candidates:
        candidates = [item.strip().upper() for item in subtask.tickers if item.strip()]
    return sorted(set(candidates))


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for parser in (_parse_iso, _parse_date):
            out = parser(text)
            if out is not None:
                return out
    return None


def _parse_iso(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date(text: str) -> datetime | None:
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.replace("\x00", " ").replace("\r", " ").strip()
        return text[:1000]
    return str(value)[:1000]


def _try_json(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _lookup_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    lowered = {_canonical_key(raw_key): value for raw_key, value in row.items()}
    for key in keys:
        candidate = lowered.get(_canonical_key(key))
        if candidate is not None:
            return candidate
    return None


def _lookup_text(row: dict[str, Any], *keys: str) -> str:
    return _to_text(_lookup_value(row, *keys))


def _canonical_key(key: Any) -> str:
    return str(key).strip().lower().replace("_", " ")
