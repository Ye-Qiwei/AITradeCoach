"""Lightweight parsers for weakly-structured LLM text outputs."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.judgement_models import (
    ALLOWED_EVALUATION_WINDOWS,
    CollectedEvidenceItem,
    DailyJudgementFeedback,
    EvidenceSource,
    JudgementEvidence,
    JudgementItem,
    ParserOutput,
    ResearchOutput,
    ResearchedJudgementItem,
    TradeAction,
)

_ALLOWED_ACTIONS = {"buy", "sell", "add", "reduce", "hold", "watch"}
_ALLOWED_CATEGORIES = {
    "market_view",
    "asset_view",
    "macro_view",
    "risk_view",
    "opportunity_view",
    "non_action",
    "reflection",
}
_ALLOWED_FEEDBACK = {"likely_correct", "likely_wrong", "insufficient_evidence", "high_uncertainty"}
_ALLOWED_SIGNALS = {"support", "oppose", "uncertain"}
_ALLOWED_QUALITY = {"sufficient", "insufficient", "conflicting", "stale", "indirect"}


def _normalize_window(value: Any, default: str = "1 week") -> str:
    text = str(value or "").strip()
    return text if text in ALLOWED_EVALUATION_WINDOWS else default


def _extract_json_payload(raw_text: str) -> Any | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _parse_markdown_key_values(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        out[key.strip().lower()] = value.strip().strip("`")
    return out


def parse_parser_output_text(
    raw_text: str,
    *,
    run_id: str,
    user_id: str,
    run_date: date,
    raw_log_text: str,
) -> ParserOutput:
    _ = run_id
    _ = raw_log_text
    payload = _extract_json_payload(raw_text)
    trade_raw: list[dict[str, Any]] = []
    judgement_raw: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        trade_raw = [i for i in payload.get("trade_actions", []) if isinstance(i, dict)]
        judgement_raw = [i for i in payload.get("judgements", []) if isinstance(i, dict)]

    trade_actions: list[TradeAction] = []
    for item in trade_raw:
        action = str(item.get("action", "")).strip().lower()
        target_asset = str(item.get("target_asset", item.get("target", ""))).strip()
        if action in _ALLOWED_ACTIONS and target_asset:
            trade_actions.append(TradeAction(action=action, target_asset=target_asset))

    judgements: list[JudgementItem] = []
    for item in judgement_raw:
        category = str(item.get("category", "")).strip().lower()
        target = str(item.get("target", "")).strip()
        thesis = str(item.get("thesis", "")).strip()
        if category not in _ALLOWED_CATEGORIES or not target or not thesis:
            continue
        deps = item.get("dependencies", [])
        dep_values = [str(d).strip() for d in (deps if isinstance(deps, list) else [deps]) if str(d).strip()]
        judgements.append(
            JudgementItem(
                category=category,
                target=target,
                thesis=thesis,
                evaluation_window=_normalize_window(item.get("evaluation_window")),
                dependencies=dep_values,
            )
        )

    return ParserOutput(user_id=user_id, run_date=run_date, trade_actions=trade_actions, judgements=judgements)


def _parse_research_json(item: dict[str, Any], fallback: JudgementItem | None) -> ResearchedJudgementItem | None:
    category = str(item.get("category", getattr(fallback, "category", ""))).strip().lower()
    target = str(item.get("target", getattr(fallback, "target", ""))).strip()
    thesis = str(item.get("thesis", getattr(fallback, "thesis", ""))).strip()
    if category not in _ALLOWED_CATEGORIES or not target or not thesis:
        return None
    evidence_data = item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
    raw_items = evidence_data.get("collected_evidence_items", [])
    collected: list[CollectedEvidenceItem] = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            sources_raw = raw_item.get("sources", [])
            sources = [
                EvidenceSource(
                    provider=str(src.get("provider", "")).strip() or "unknown",
                    title=str(src.get("title", "")).strip() or None,
                    uri=str(src.get("uri", "")).strip() or None,
                    published_at=str(src.get("published_at", "")).strip() or None,
                )
                for src in sources_raw
                if isinstance(src, dict)
            ]
            collected.append(
                CollectedEvidenceItem(
                    evidence_type=str(raw_item.get("evidence_type", "other")).strip() or "other",
                    summary=str(raw_item.get("summary", "")).strip() or "no summary",
                    related_tickers=[str(t).strip() for t in raw_item.get("related_tickers", []) if str(t).strip()],
                    sources=sources,
                )
            )
    signal = str(evidence_data.get("support_signal", "uncertain")).strip().lower()
    quality = str(evidence_data.get("evidence_quality", "insufficient")).strip().lower()
    evidence = JudgementEvidence(
        support_signal=signal if signal in _ALLOWED_SIGNALS else "uncertain",
        evidence_quality=quality if quality in _ALLOWED_QUALITY else "insufficient",
        evidence_summary=str(evidence_data.get("evidence_summary", "")).strip(),
        key_points=[str(k).strip() for k in evidence_data.get("key_points", []) if str(k).strip()],
        collected_evidence_items=collected,
    )
    return ResearchedJudgementItem(
        category=category,
        target=target,
        thesis=thesis,
        evaluation_window=_normalize_window(item.get("evaluation_window", getattr(fallback, "evaluation_window", "1 week"))),
        dependencies=[str(d).strip() for d in item.get("dependencies", getattr(fallback, "dependencies", [])) if str(d).strip()],
        evidence=evidence,
    )


def parse_research_output_text(raw_text: str, *, judgements: list[JudgementItem]) -> ResearchOutput:
    payload = _extract_json_payload(raw_text)
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        items = [i for i in payload.get("judgements", []) if isinstance(i, dict)]
    elif isinstance(payload, list):
        items = [i for i in payload if isinstance(i, dict)]

    parsed: list[ResearchedJudgementItem] = []
    for idx, fallback in enumerate(judgements):
        raw_item = items[idx] if idx < len(items) else {}
        obj = _parse_research_json(raw_item, fallback)
        if obj is None:
            obj = ResearchedJudgementItem(
                category=fallback.category,
                target=fallback.target,
                thesis=fallback.thesis,
                evaluation_window=fallback.evaluation_window,
                dependencies=fallback.dependencies,
            )
        parsed.append(obj)
    return ResearchOutput(judgements=parsed)


def parse_reporter_output_text(raw_text: str, judgement_count: int) -> ReporterOutput:
    feedback: list[DailyJudgementFeedback] = []
    blocks = [b for b in re.split(r"\n(?=##+\s)", raw_text) if b.strip()]
    for block in blocks[:judgement_count]:
        fields = _parse_markdown_key_values(block)
        initial_feedback = fields.get("initial_feedback", "insufficient_evidence").strip().lower()
        if initial_feedback not in _ALLOWED_FEEDBACK:
            initial_feedback = "insufficient_evidence"
        feedback.append(
            DailyJudgementFeedback(
                initial_feedback=initial_feedback,
                evaluation_window=_normalize_window(fields.get("evaluation_window")),
            )
        )
    while len(feedback) < judgement_count:
        feedback.append(DailyJudgementFeedback(initial_feedback="insufficient_evidence", evaluation_window="1 week"))
    return ReporterOutput(markdown=raw_text, judgement_feedback=feedback)
