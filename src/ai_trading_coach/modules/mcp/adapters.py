"""
Note: daily agent payloads use id-less list alignment between nodes.
Normalization adapters from raw MCP tool outputs to EvidenceItem."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ai_trading_coach.domain.agent_models import PlanSubTask
from ai_trading_coach.domain.models import EvidenceItem, SourceAttribution


_ERROR_TEXT_PATTERNS = ("error executing tool", "tool_error:", "validation error")


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


def extract_mcp_error(raw_result: Any) -> str | None:
    """Return normalized MCP error message when raw output encodes a failure."""
    if isinstance(raw_result, dict):
        for key in ("error", "error_message", "message"):
            value = raw_result.get(key)
            if isinstance(value, str) and value.strip() and _looks_like_error(value):
                return value.strip()
            if isinstance(value, dict):
                msg = _extract_error_from_mapping(value)
                if msg:
                    return msg
        if any(k in raw_result for k in ("error", "error_code", "details")):
            return _extract_error_from_mapping(raw_result)

    if isinstance(raw_result, str):
        as_json = _try_json(raw_result)
        if as_json is not None:
            parsed = extract_mcp_error(as_json)
            if parsed:
                return parsed
        if _looks_like_error(raw_result):
            return raw_result.strip()

    content = getattr(raw_result, "content", None)
    if isinstance(content, list):
        for chunk in content:
            text = _extract_text_chunk(chunk)
            if not text:
                continue
            parsed = extract_mcp_error(text)
            if parsed:
                return parsed

    text = _to_text(raw_result)
    if _looks_like_error(text):
        return text.strip()
    return None


def parse_yfinance_price_history_result(raw_result: Any) -> list[dict[str, Any]]:
    """Parse yfinance_get_price_history result markdown table into row dicts."""
    table_text = _extract_price_table_text(raw_result)
    if not table_text:
        return []

    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    header_idx = next((i for i, line in enumerate(lines) if _looks_like_markdown_row(line) and "date" in line.lower()), None)
    if header_idx is None or header_idx + 1 >= len(lines):
        return []

    headers = _split_markdown_row(lines[header_idx])
    if not headers:
        return []
    canonical_headers = [_canonical_header(col) for col in headers]

    rows: list[dict[str, Any]] = []
    for raw_line in lines[header_idx + 1 :]:
        if not _looks_like_markdown_row(raw_line) or _is_separator_row(raw_line):
            continue
        values = _split_markdown_row(raw_line)
        if len(values) != len(canonical_headers):
            continue
        parsed_row: dict[str, Any] = {}
        for key, value in zip(canonical_headers, values, strict=False):
            parsed_row[key] = _coerce_scalar(value)
        if parsed_row:
            rows.append(parsed_row)
    return rows


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


def _extract_price_table_text(raw_result: Any) -> str:
    if isinstance(raw_result, dict):
        result = raw_result.get("result")
        if isinstance(result, str):
            return result
    if isinstance(raw_result, str):
        loaded = _try_json(raw_result)
        if isinstance(loaded, dict):
            result = loaded.get("result")
            if isinstance(result, str):
                return result
        return raw_result

    content = getattr(raw_result, "content", None)
    if isinstance(content, list):
        for chunk in content:
            text = _extract_text_chunk(chunk)
            if not text:
                continue
            loaded = _try_json(text)
            if isinstance(loaded, dict) and isinstance(loaded.get("result"), str):
                return str(loaded["result"])
            if "|" in text and "date" in text.lower():
                return text

    return _to_text(raw_result)


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


def _extract_error_from_mapping(value: dict[str, Any]) -> str | None:
    chunks: list[str] = []
    for key in ("error", "error_code", "message", "details"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            chunks.append(f"{key}: {item.strip()}")
        elif isinstance(item, dict):
            nested = _extract_error_from_mapping(item)
            if nested:
                chunks.append(f"{key}: {nested}")
    return "; ".join(chunks) if chunks else None


def _looks_like_error(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(lowered) and any(pattern in lowered for pattern in _ERROR_TEXT_PATTERNS)


def _looks_like_markdown_row(line: str) -> bool:
    return line.count("|") >= 2


def _split_markdown_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def _is_separator_row(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _canonical_header(value: str) -> str:
    collapsed = " ".join(value.strip().split())
    lowered = collapsed.lower()
    mapping = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "dividends": "Dividends",
        "stock splits": "Stock Splits",
    }
    return mapping.get(lowered, collapsed)


def _coerce_scalar(value: str) -> Any:
    text = value.strip().replace(",", "")
    if not text:
        return ""
    if text in {"-", "--", "null", "None", "nan", "NaN"}:
        return None
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d*\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text
