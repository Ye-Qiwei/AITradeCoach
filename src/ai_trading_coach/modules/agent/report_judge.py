"""Judge stage with deterministic checks only."""

from __future__ import annotations

import re
from typing import Any

import markdown
from bs4 import BeautifulSoup

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.models import EvidencePacket


class ReportJudge:
    _citation_re = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)

    def __init__(self, gateway=None, prompt_manager=None) -> None:
        _ = gateway
        _ = prompt_manager

    def evaluate(self, *, report_markdown: str, judge_context: dict[str, object], evidence_packet: EvidencePacket):
        _ = evidence_packet
        rule_verdict = self._rule_check(
            report_markdown=report_markdown,
            judgement_feedback=[DailyJudgementFeedback.model_validate(i) for i in judge_context.get("judgement_feedback", [])],
            bundles=judge_context.get("judgement_bundles", []),
        )
        return rule_verdict, None

    def _rule_check(self, *, report_markdown: str, judgement_feedback: list[DailyJudgementFeedback], bundles: object) -> JudgeVerdict:
        reasons: list[str] = []
        passed = True
        bundle_list = [b for b in bundles if isinstance(b, dict)] if isinstance(bundles, list) else []
        soup = BeautifulSoup(markdown.markdown(report_markdown, extensions=["extra"]), "html.parser")
        detailed_heading = next((h for h in soup.find_all("h2") if h.get_text(strip=True).lower() == "detailed analysis"), None)
        sections = detailed_heading.find_all_next("h3") if detailed_heading is not None else []

        if len(sections) != len(bundle_list):
            passed = False
            reasons.append("Detailed analysis section count must match judgement bundle count.")

        if len(judgement_feedback) != len(bundle_list):
            passed = False
            reasons.append("Feedback count must match judgement bundle count.")

        cited = self._citation_re.findall(report_markdown)
        total_lines = [line.strip() for line in report_markdown.splitlines() if line.strip()]
        coverage = 1.0 if not total_lines else len(cited) / len(total_lines)

        for idx, section in enumerate(sections[: len(bundle_list)]):
            text = section.get_text(" ", strip=True)
            body = []
            node = section.next_sibling
            while node is not None and getattr(node, "name", None) != "h3":
                if hasattr(node, "get_text"):
                    body.append(node.get_text(" ", strip=True))
                node = node.next_sibling
            block = f"{text} {' '.join(body)}"
            citations = self._citation_re.findall(block)
            if citations and self._allowed_sources(bundle_list[idx]):
                allowed = self._allowed_sources(bundle_list[idx])
                if not all(c.strip().lower() in allowed for c in citations):
                    passed = False
                    reasons.append(f"Section {idx + 1} contains citations not grounded in source registry.")

        for idx, item in enumerate(judgement_feedback):
            if item.evaluation_window not in ALLOWED_EVALUATION_WINDOWS:
                passed = False
                reasons.append(f"Invalid evaluation window at feedback index {idx + 1}: {item.evaluation_window}")

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction="Rewrite feedback summary and detailed analysis to align with judgement order and sources." if not passed else None,
            citation_coverage=coverage,
        )

    def _allowed_sources(self, bundle: dict[str, Any]) -> set[str]:
        allowed: set[str] = set()
        evidence = bundle.get("evidence", {}) if isinstance(bundle, dict) else {}
        items = evidence.get("collected_evidence_items", []) if isinstance(evidence, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            for source in item.get("sources", []):
                if not isinstance(source, dict):
                    continue
                for key in ("source_id", "provider", "title", "uri"):
                    value = str(source.get(key, "")).strip().lower()
                    if value:
                        allowed.add(value)
        return allowed
