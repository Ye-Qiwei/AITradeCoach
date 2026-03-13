"""Lightweight parsers for weakly-structured LLM text outputs."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.id_generation import make_action_id, make_judgement_id, make_parse_id, make_research_id
from ai_trading_coach.domain.judgement_models import (
    ALLOWED_EVALUATION_WINDOWS,
    DailyJudgementFeedback,
    JudgementEvidence,
    JudgementItem,
    ParserOutput,
    ResearchOutput,
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
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    start = min([idx for idx in (text.find("{"), text.find("[")) if idx != -1], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
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
    payload = _extract_json_payload(raw_text)
    trade_raw: list[dict[str, Any]] = []
    judgement_raw: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        trade_raw = [i for i in payload.get("trade_actions", []) if isinstance(i, dict)]
        judgement_raw = [i for i in payload.get("judgements", []) if isinstance(i, dict)]
    else:
        parts = re.split(r"^#{1,3}\s*(TRADE_ACTIONS|JUDGEMENTS)\s*$", raw_text, flags=re.MULTILINE | re.IGNORECASE)
        for idx in range(1, len(parts), 2):
            section = parts[idx].strip().lower()
            body = parts[idx + 1]
            blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
            parsed = [_parse_markdown_key_values(block) for block in blocks]
            if section == "trade_actions":
                trade_raw = parsed
            if section == "judgements":
                judgement_raw = parsed

    trade_actions: list[TradeAction] = []
    for idx, item in enumerate(trade_raw, start=1):
        action = str(item.get("action", "")).strip().lower()
        target_asset = str(item.get("target_asset", item.get("target", ""))).strip()
        if action not in _ALLOWED_ACTIONS or not target_asset:
            continue
        trade_actions.append(TradeAction(action_id=make_action_id(run_id, idx, item), action=action, target_asset=target_asset))

    local_to_global: dict[str, str] = {}
    for idx, item in enumerate(judgement_raw, start=1):
        category = str(item.get("category", "")).strip().lower()
        target = str(item.get("target", "")).strip()
        thesis = str(item.get("thesis", "")).strip()
        if category not in _ALLOWED_CATEGORIES or not target or not thesis:
            continue
        local_id = str(item.get("local_id", f"j{idx}")).strip() or f"j{idx}"
        local_to_global[local_id] = make_judgement_id(run_id, "judgement", idx, category, target, thesis)

    judgements: list[JudgementItem] = []
    for idx, item in enumerate(judgement_raw, start=1):
        category = str(item.get("category", "")).strip().lower()
        target = str(item.get("target", "")).strip()
        thesis = str(item.get("thesis", "")).strip()
        if category not in _ALLOWED_CATEGORIES or not target or not thesis:
            continue
        local_id = str(item.get("local_id", f"j{idx}")).strip() or f"j{idx}"
        deps = item.get("dependencies", [])
        dep_values = deps if isinstance(deps, list) else [str(deps)]
        judgements.append(
            JudgementItem(
                judgement_id=local_to_global.get(local_id, make_judgement_id(run_id, "judgement", idx, category, target, thesis)),
                category=category,
                target=target,
                thesis=thesis,
                evaluation_window=_normalize_window(item.get("evaluation_window")),
                dependencies=[local_to_global[d] for d in dep_values if isinstance(d, str) and d in local_to_global],
            )
        )

    return ParserOutput(
        parse_id=make_parse_id(run_id, raw_log_text),
        user_id=user_id,
        run_date=run_date,
        trade_actions=trade_actions,
        judgements=judgements,
    )


def parse_research_output_text(raw_text: str, *, run_id: str) -> ResearchOutput:
    payload = _extract_json_payload(raw_text)
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        items = [i for i in payload.get("judgement_evidence", []) if isinstance(i, dict)]
    elif isinstance(payload, list):
        items = [i for i in payload if isinstance(i, dict)]
    else:
        body = re.split(r"^#{1,3}\s*JUDGEMENT_EVIDENCE\s*$", raw_text, flags=re.MULTILINE | re.IGNORECASE)
        source = body[-1] if len(body) > 1 else raw_text
        blocks = [b for b in re.split(r"\n\s*\n", source) if b.strip()]
        items = [_parse_markdown_key_values(block) for block in blocks]

    parsed = [
        JudgementEvidence(
            judgement_id=str(item.get("judgement_id", "")).strip(),
            evidence_item_ids=[str(i).strip() for i in (item.get("evidence_item_ids", []) if isinstance(item.get("evidence_item_ids", []), list) else re.split(r"\s*,\s*", str(item.get("evidence_item_ids", "")))) if str(i).strip()],
            support_signal=(str(item.get("support_signal", "uncertain")).strip().lower() if str(item.get("support_signal", "uncertain")).strip().lower() in _ALLOWED_SIGNALS else "uncertain"),
            evidence_quality=(str(item.get("evidence_quality", "insufficient")).strip().lower() if str(item.get("evidence_quality", "insufficient")).strip().lower() in _ALLOWED_QUALITY else "insufficient"),
        )
        for item in items
        if str(item.get("judgement_id", "")).strip()
    ]
    return ResearchOutput(research_id=make_research_id(run_id), judgement_evidence=parsed)


def parse_reporter_output_text(raw_text: str) -> ReporterOutput:
    feedback: list[DailyJudgementFeedback] = []
    for block in [b for b in re.split(r"\n(?=##+\s)", raw_text) if b.strip()]:
        fields = _parse_markdown_key_values(block)
        judgement_id = fields.get("judgement_id", "").strip()
        if not judgement_id:
            continue
        initial_feedback = fields.get("initial_feedback", "insufficient_evidence").strip().lower()
        if initial_feedback not in _ALLOWED_FEEDBACK:
            initial_feedback = "insufficient_evidence"
        feedback.append(
            DailyJudgementFeedback(
                judgement_id=judgement_id,
                initial_feedback=initial_feedback,
                evaluation_window=_normalize_window(fields.get("evaluation_window")),
            )
        )
    return ReporterOutput(markdown=raw_text, judgement_feedback=feedback)
