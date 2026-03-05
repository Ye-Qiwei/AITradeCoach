"""Daily log intake and canonicalization implementation."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any

import yaml

from ai_trading_coach.domain.contracts import LogIntakeInput, LogIntakeOutput
from ai_trading_coach.domain.enums import AssetType, SourceType, TradeSide
from ai_trading_coach.domain.models import (
    DailyLogNormalized,
    DailyLogRaw,
    FieldError,
    MarketContext,
    ReflectionBlock,
    ScanSignals,
    TradeEvent,
    TradeNarrative,
    UserState,
)


class MarkdownLogIntakeCanonicalizer:
    """Parse markdown-like daily logs into normalized schema."""

    _trade_re = re.compile(
        r"(?P<ticker>\S+)\s+"
        r"(?P<side>BUY|SELL)\s+"
        r"(?P<qty>[0-9][0-9,]*(?:\.[0-9]+)?)"
        r"(?:\s*(?:股|shares?))?"
        r"(?:\s+(?P<price>[0-9][0-9,]*(?:\.[0-9]+)?))?"
        r"(?:\s*(?P<currency>[A-Za-z]{3,4}))?",
        flags=re.IGNORECASE,
    )
    _ticker_re = re.compile(r"\b(?:[0-9]{3,5}\.[A-Z]{1,4}|[A-Z]{1,5}\.[A-Z]{1,4})\b")

    def ingest(self, data: LogIntakeInput) -> LogIntakeOutput:
        text = data.raw_log_text.strip()
        raw = DailyLogRaw(
            log_id=self._build_log_id(data.user_id, data.run_date, text),
            user_id=data.user_id,
            source_type=SourceType.MARKDOWN,
            source_path=data.source_path,
            content=text,
            metadata={"run_date": data.run_date.isoformat()},
        )

        frontmatter, body = self._split_frontmatter(text)
        field_errors: list[FieldError] = []

        parsed_date = data.run_date
        front_date = frontmatter.get("date")
        if isinstance(front_date, datetime):
            parsed_date = front_date.date()
        elif isinstance(front_date, date):
            parsed_date = front_date
        elif isinstance(front_date, str):
            try:
                parsed_date = date.fromisoformat(front_date)
            except ValueError:
                field_errors.append(
                    FieldError(
                        field="date",
                        message=f"Invalid date '{front_date}', fallback to run_date",
                        severity="warning",
                    )
                )

        traded_tickers = self._normalize_string_list(frontmatter.get("traded_tickers"))
        mentioned_tickers = self._normalize_string_list(frontmatter.get("mentioned_tickers"))

        sections, directives = self._parse_sections(body)

        user_state = self._parse_user_state(sections.get("状态", []), field_errors)
        market_context = self._parse_market_context(sections.get("市场", []))
        scan_signals = self._parse_scan_signals(sections.get("扫描", []))
        reflection = self._parse_reflection(sections.get("复盘", []))

        trade_events, trade_narratives, trade_errors = self._parse_trade_records(
            sections.get("交易记录", []),
            data.user_id,
            parsed_date,
        )
        field_errors.extend(trade_errors)

        inferred_mentioned = self._extract_tickers(text)
        merged_mentioned = self._unique(mentioned_tickers + inferred_mentioned)
        merged_traded = self._unique(traded_tickers + [event.ticker for event in trade_events])

        normalized = DailyLogNormalized(
            log_id=raw.log_id,
            user_id=data.user_id,
            log_date=parsed_date,
            traded_tickers=merged_traded,
            mentioned_tickers=merged_mentioned,
            user_state=user_state,
            market_context=market_context,
            trade_events=trade_events,
            trade_narratives=trade_narratives,
            scan_signals=scan_signals,
            reflection=reflection,
            ai_directives=directives,
            raw_text=text,
            field_errors=field_errors,
        )
        return LogIntakeOutput(raw=raw, normalized=normalized)

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        if not text.startswith("---"):
            return {}, text

        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text

        _, frontmatter_raw, body = parts
        try:
            parsed = yaml.safe_load(frontmatter_raw.strip())
            if isinstance(parsed, dict):
                return parsed, body.strip()
        except yaml.YAMLError:
            return {}, text
        return {}, body.strip()

    def _parse_sections(self, body: str) -> tuple[dict[str, list[str]], list[str]]:
        sections: dict[str, list[str]] = {}
        directives: list[str] = []
        current_section: str | None = None

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("## "):
                current_section = line[3:].strip()
                sections.setdefault(current_section, [])
                continue
            if line.startswith("@AI"):
                directives.append(line)
                continue
            if current_section and line.startswith("- "):
                sections[current_section].append(line[2:].strip())

        return sections, directives

    def _parse_user_state(self, rows: list[str], errors: list[FieldError]) -> UserState:
        mapping = self._parse_key_values(rows)
        stress = self._to_int(mapping.get("stress"), "stress", errors)
        focus = self._to_int(mapping.get("focus"), "focus", errors)
        return UserState(
            emotion=mapping.get("emotion"),
            stress=stress,
            focus=focus,
        )

    def _parse_market_context(self, rows: list[str]) -> MarketContext:
        mapping = self._parse_key_values(rows)
        key_raw = mapping.get("key_var") or mapping.get("key_vars")
        key_variables = self._split_multi_value(key_raw)
        return MarketContext(regime=mapping.get("regime"), key_variables=key_variables)

    def _parse_scan_signals(self, rows: list[str]) -> ScanSignals:
        mapping = self._parse_key_values(rows)
        return ScanSignals(
            anxiety=self._split_multi_value(mapping.get("anxiety")),
            fomo=self._split_multi_value(mapping.get("fomo")),
            not_trade=self._split_multi_value(mapping.get("not_trade")),
        )

    def _parse_reflection(self, rows: list[str]) -> ReflectionBlock:
        mapping = self._parse_key_values(rows)
        return ReflectionBlock(
            facts=self._split_multi_value(mapping.get("fact")),
            gaps=self._split_multi_value(mapping.get("gap")),
            lessons=self._split_multi_value(mapping.get("lesson")),
        )

    def _parse_trade_records(
        self,
        rows: list[str],
        user_id: str,
        trade_date: date,
    ) -> tuple[list[TradeEvent], list[TradeNarrative], list[FieldError]]:
        events: list[TradeEvent] = []
        narratives: list[TradeNarrative] = []
        errors: list[FieldError] = []

        for idx, row in enumerate(rows):
            parts = [part.strip() for part in row.split("|")]
            base = parts[0]
            attrs = self._parse_attrs(parts[1:])
            match = self._trade_re.search(base)

            if not match:
                narratives.append(TradeNarrative(raw_line=row, parsed=False))
                errors.append(
                    FieldError(
                        field=f"trade_records[{idx}]",
                        message="Unable to parse trade line, preserved as raw narrative",
                        severity="warning",
                    )
                )
                continue

            ticker = match.group("ticker").upper()
            side = TradeSide.BUY if match.group("side").upper() == "BUY" else TradeSide.SELL
            quantity = self._to_float(match.group("qty"))
            unit_price = self._to_float(match.group("price")) if match.group("price") else None
            currency = (match.group("currency") or "USD").upper()

            try:
                event = TradeEvent(
                    event_id=f"te_{trade_date.isoformat()}_{idx}",
                    user_id=user_id,
                    trade_date=trade_date,
                    ticker=ticker,
                    asset_type=self._parse_asset_type(
                        attrs.get("asset_type") or attrs.get("asset") or attrs.get("type")
                    ),
                    side=side,
                    quantity=quantity,
                    unit_price=unit_price,
                    currency=currency,
                    reason=attrs.get("reason"),
                    source_tags=self._split_multi_value(attrs.get("source")),
                    trigger=attrs.get("trig") or attrs.get("trigger"),
                    moment_emotion=attrs.get("moment_emotion"),
                    risk_note=attrs.get("risk"),
                )
                events.append(event)
                narratives.append(TradeNarrative(raw_line=row, parsed=True))
            except Exception as exc:  # noqa: BLE001
                narratives.append(TradeNarrative(raw_line=row, parsed=False))
                errors.append(
                    FieldError(
                        field=f"trade_records[{idx}]",
                        message=f"Trade validation failed: {exc}",
                        severity="warning",
                    )
                )

        return events, narratives, errors

    def _parse_asset_type(self, value: str | None) -> AssetType:
        if not value:
            return AssetType.STOCK
        normalized = value.strip().lower()
        mapping = {
            "stock": AssetType.STOCK,
            "equity": AssetType.STOCK,
            "etf": AssetType.ETF,
            "fund": AssetType.FUND,
            "index": AssetType.INDEX,
            "option": AssetType.OPTION,
            "future": AssetType.FUTURE,
            "futures": AssetType.FUTURE,
            "other": AssetType.OTHER,
        }
        return mapping.get(normalized, AssetType.STOCK)

    def _parse_attrs(self, parts: list[str]) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
            elif ":" in part:
                key, value = part.split(":", 1)
            else:
                continue
            attrs[key.strip().lower()] = value.strip()
        return attrs

    def _parse_key_values(self, rows: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in rows:
            if ":" not in row:
                continue
            key, value = row.split(":", 1)
            result[key.strip().lower()] = value.strip().strip('"')
        return result

    def _normalize_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return self._split_multi_value(value)
        return []

    def _split_multi_value(self, value: str | None) -> list[str]:
        if not value:
            return []
        cleaned = value.strip().strip('"')
        if not cleaned:
            return []
        if any(sep in cleaned for sep in [";", "，", ",", "、", "/"]):
            parts = re.split(r"[;,，、/]", cleaned)
            return [part.strip() for part in parts if part.strip()]
        return [cleaned]

    def _extract_tickers(self, text: str) -> list[str]:
        return self._unique([match.group(0).upper() for match in self._ticker_re.finditer(text)])

    def _to_int(self, value: str | None, field: str, errors: list[FieldError]) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            errors.append(FieldError(field=field, message=f"Invalid int value '{value}'", severity="warning"))
            return None

    def _to_float(self, value: str) -> float:
        return float(value.replace(",", ""))

    def _build_log_id(self, user_id: str, run_date: date, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"log_{user_id}_{run_date.isoformat()}_{digest}"

    def _unique(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out


# Backward-compatible alias
PlaceholderLogIntakeCanonicalizer = MarkdownLogIntakeCanonicalizer
