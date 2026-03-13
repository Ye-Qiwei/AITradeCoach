"""Judge stage with deterministic checks only."""

from __future__ import annotations

import re
from typing import Any

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.models import EvidencePacket


class ReportJudge:
    _citation_re = re.compile(r"\[来源:\s*([^\],;]+)(?:[,;]\s*([^\]]+))?\]", re.IGNORECASE)

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

    def _rule_check(
        self,
        *,
        report_markdown: str,
        judgement_feedback: list[DailyJudgementFeedback],
        bundles: object,
    ) -> JudgeVerdict:
        reasons: list[str] = []
        passed = True
        sections = [s for s in re.split(r"\n(?=##\s)", report_markdown) if s.strip()]
        bundle_list = [b for b in bundles if isinstance(b, dict)] if isinstance(bundles, list) else []

        if len(sections) != len(bundle_list):
            passed = False
            reasons.append("Section count must match judgement bundle count.")

        if len(judgement_feedback) != len(bundle_list):
            passed = False
            reasons.append("Feedback count must match judgement bundle count.")

        total_lines = [line.strip() for line in report_markdown.splitlines() if line.strip()]
        cited_lines = sum(1 for line in total_lines if self._citation_re.search(line))
        coverage = 1.0 if not total_lines else cited_lines / len(total_lines)

        for idx, section in enumerate(sections[: len(bundle_list)]):
            allowed = self._allowed_sources(bundle_list[idx])
            citations = self._citation_re.findall(section)
            if citations and allowed:
                matched = 0
                for provider, details in citations:
                    text = f"{provider} {details}".lower()
                    if any(token in text for token in allowed):
                        matched += 1
                if matched < len(citations):
                    passed = False
                    reasons.append(f"Section {idx + 1} contains citations not grounded in judgement evidence sources.")

        for idx, item in enumerate(judgement_feedback):
            if item.evaluation_window not in ALLOWED_EVALUATION_WINDOWS:
                passed = False
                reasons.append(f"Invalid evaluation window at feedback index {idx + 1}: {item.evaluation_window}")

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction="Rewrite to align section order, feedback order, and evidence-backed citations." if not passed else None,
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
                for key in ("provider", "title", "uri"):
                    value = str(source.get(key, "")).strip().lower()
                    if value:
                        allowed.add(value)
        return allowed
