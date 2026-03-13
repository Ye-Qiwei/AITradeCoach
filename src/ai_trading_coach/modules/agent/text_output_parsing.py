"""Markdown-first parsers for LLM text outputs."""

from __future__ import annotations

from datetime import date
from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag

from ai_trading_coach.domain.agent_models import ReporterOutput
from ai_trading_coach.domain.judgement_models import (
    ALLOWED_EVALUATION_WINDOWS,
    DailyJudgementFeedback,
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


def _parse_markdown(raw_text: str) -> BeautifulSoup:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("LLM output is empty.")
    if stripped.startswith("{") or stripped.startswith("["):
        raise ValueError("Expected markdown output, but JSON-like output was returned.")
    html = markdown.markdown(raw_text, extensions=["extra"])
    return BeautifulSoup(html, "html.parser")


def _normalized_key(text: str) -> str:
    return "_".join(text.strip().lower().replace("-", "_").split())


def _heading_sections(soup: BeautifulSoup, level: str) -> dict[str, Tag]:
    return {heading.get_text(strip=True).lower(): heading for heading in soup.find_all(level)}


def _subsections(parent_heading: Tag, level: str = "h2") -> list[tuple[str, Tag]]:
    found: list[tuple[str, Tag]] = []
    node = parent_heading.next_sibling
    while node is not None:
        if isinstance(node, Tag) and node.name == parent_heading.name:
            break
        if isinstance(node, Tag) and node.name == level:
            found.append((node.get_text(strip=True), node))
        node = node.next_sibling
    return found


def _heading_fields(heading: Tag) -> dict[str, str | list[str]]:
    out: dict[str, str | list[str]] = {}
    node = heading.next_sibling
    while node is not None:
        if isinstance(node, Tag) and node.name in {"h1", "h2", "h3"}:
            break
        if isinstance(node, Tag) and node.name == "ul":
            for li in node.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)
                if ":" not in text:
                    continue
                key, value = text.split(":", 1)
                normalized = _normalized_key(key)
                nested = li.find("ul")
                if nested is not None:
                    out[normalized] = [item.get_text(" ", strip=True) for item in nested.find_all("li", recursive=False)]
                else:
                    out[normalized] = value.strip()
        node = node.next_sibling
    return out


def parse_parser_output_text(raw_text: str, *, run_id: str, user_id: str, run_date: date, raw_log_text: str) -> ParserOutput:
    _ = run_id
    _ = raw_log_text
    soup = _parse_markdown(raw_text)
    sections = _heading_sections(soup, "h1")
    actions_h1 = sections.get("trade actions")
    judgements_h1 = sections.get("judgements")
    if actions_h1 is None:
        raise ValueError("Parser output missing '# Trade Actions' section.")
    if judgements_h1 is None:
        raise ValueError("Parser output missing '# Judgements' section.")

    trade_actions: list[TradeAction] = []
    for _, heading in _subsections(actions_h1, "h2"):
        fields = _heading_fields(heading)
        action = str(fields.get("action", "")).strip().lower()
        target_asset = str(fields.get("target_asset", fields.get("target", ""))).strip()
        if action in _ALLOWED_ACTIONS and target_asset:
            trade_actions.append(TradeAction(action=action, target_asset=target_asset))

    judgements: list[JudgementItem] = []
    for _, heading in _subsections(judgements_h1, "h2"):
        fields = _heading_fields(heading)
        category = str(fields.get("category", "")).strip().lower()
        target = str(fields.get("target", "")).strip()
        thesis = str(fields.get("thesis", "")).strip()
        if category not in _ALLOWED_CATEGORIES or not target or not thesis:
            continue
        dependencies = fields.get("dependencies", "")
        if isinstance(dependencies, list):
            deps = [item.strip() for item in dependencies if item.strip() and item.strip().lower() != "none"]
        else:
            deps = [part.strip() for part in str(dependencies).split(",") if part.strip() and part.strip().lower() != "none"]
        judgements.append(
            JudgementItem(
                category=category,
                target=target,
                thesis=thesis,
                evaluation_window=_normalize_window(fields.get("evaluation_window")),
                dependencies=deps,
            )
        )
    return ParserOutput(user_id=user_id, run_date=run_date, trade_actions=trade_actions, judgements=judgements)


def parse_research_output_text(raw_text: str, *, judgements: list[JudgementItem]) -> list[dict[str, Any]]:
    soup = _parse_markdown(raw_text)
    sections = _heading_sections(soup, "h1")
    evidence_h1 = sections.get("judgement evidence")
    if evidence_h1 is None:
        raise ValueError("Research output missing '# Judgement Evidence' section.")
    subsections = _subsections(evidence_h1, "h2")
    if len(subsections) != len(judgements):
        raise ValueError(
            f"Research output subsection count mismatch: expected {len(judgements)}, got {len(subsections)}"
        )

    parsed: list[dict[str, Any]] = []
    for idx, (title, heading) in enumerate(subsections):
        fields = _heading_fields(heading)
        cited_raw = fields.get("cited_sources", [])
        cited_sources = [str(item).strip() for item in (cited_raw if isinstance(cited_raw, list) else [cited_raw]) if str(item).strip()]
        parsed.append(
            {
                "judgement_ref": title,
                "support_signal": str(fields.get("support_signal", "uncertain")).strip().lower(),
                "evidence_quality": str(fields.get("evidence_quality", "insufficient")).strip().lower(),
                "rationale": str(fields.get("rationale", "")).strip(),
                "cited_sources": cited_sources,
                "fallback": judgements[idx],
            }
        )
    return parsed


def build_research_output(*, parsed_markdown: list[dict[str, Any]], judgements: list[JudgementItem], cited_items: list[list[dict[str, Any]]]) -> ResearchOutput:
    items: list[ResearchedJudgementItem] = []
    for idx, fallback in enumerate(judgements):
        section = parsed_markdown[idx] if idx < len(parsed_markdown) else {}
        support_signal = str(section.get("support_signal", "uncertain")).lower()
        evidence_quality = str(section.get("evidence_quality", "insufficient")).lower()
        rationale = str(section.get("rationale", "")).strip()
        evidence = JudgementEvidence(
            support_signal=support_signal if support_signal in _ALLOWED_SIGNALS else "uncertain",
            evidence_quality=evidence_quality if evidence_quality in _ALLOWED_QUALITY else "insufficient",
            evidence_summary=rationale,
            collected_evidence_items=cited_items[idx] if idx < len(cited_items) else [],
        )
        items.append(
            ResearchedJudgementItem(
                category=fallback.category,
                target=fallback.target,
                thesis=fallback.thesis,
                evaluation_window=fallback.evaluation_window,
                dependencies=fallback.dependencies,
                evidence=evidence,
            )
        )
    return ResearchOutput(judgements=items)


def parse_reporter_output_text(raw_text: str, judgement_count: int) -> ReporterOutput:
    soup = _parse_markdown(raw_text)
    feedback_heading = next((h for h in soup.find_all("h2") if h.get_text(strip=True).lower() == "feedback summary"), None)
    if feedback_heading is None:
        raise ValueError("Reporter output missing '## Feedback Summary' section.")
    table = feedback_heading.find_next("table")
    if table is None:
        raise ValueError("Reporter output missing feedback summary markdown table.")

    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    rows = table.find_all("tr")[1:]
    if len(rows) != judgement_count:
        raise ValueError(f"Feedback row count mismatch: expected {judgement_count}, got {len(rows)}")

    feedback: list[DailyJudgementFeedback] = []
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        data = dict(zip(headers, cols, strict=False))
        initial_feedback = str(data.get("initial_feedback", "insufficient_evidence")).strip().lower()
        if initial_feedback not in _ALLOWED_FEEDBACK:
            initial_feedback = "insufficient_evidence"
        feedback.append(
            DailyJudgementFeedback(
                initial_feedback=initial_feedback,
                evaluation_window=_normalize_window(data.get("evaluation_window")),
            )
        )
    return ReporterOutput(markdown=raw_text, judgement_feedback=feedback)
