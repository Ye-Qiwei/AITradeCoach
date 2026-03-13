"""Judge stage with deterministic checks only."""

from __future__ import annotations

import re

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.models import EvidencePacket


class ReportJudge:
    _citation_re = re.compile(r"\[source:([A-Za-z0-9_.:-]+)\]")

    def __init__(self, gateway=None, prompt_manager=None) -> None:
        _ = gateway
        _ = prompt_manager

    def evaluate(self, *, report_markdown: str, judge_context: dict[str, object], evidence_packet: EvidencePacket):
        rule_verdict = self._rule_check(
            report_markdown=report_markdown,
            judgement_feedback=[DailyJudgementFeedback.model_validate(i) for i in judge_context.get("judgement_feedback", [])],
            expected_judgement_ids=set(judge_context.get("expected_judgement_ids", [])),
            bundles=judge_context.get("judgement_bundles", []),
        )
        return rule_verdict, None

    def _rule_check(
        self,
        *,
        report_markdown: str,
        judgement_feedback: list[DailyJudgementFeedback],
        expected_judgement_ids: set[str],
        bundles: object,
    ) -> JudgeVerdict:
        reasons: list[str] = []
        passed = True
        lines = [line.strip() for line in report_markdown.splitlines() if line.strip()]
        cited_lines = sum(1 for line in lines if self._citation_re.findall(line))
        coverage = 1.0 if not lines else cited_lines / len(lines)
        if coverage < 0.7:
            passed = False
            reasons.append("Citation coverage below threshold.")

        bundle_map = {item.get("judgement_id"): set(item.get("allowed_source_ids", [])) for item in bundles if isinstance(item, dict)}
        section_citations: dict[str, set[str]] = {}
        current_jid: str | None = None
        for line in report_markdown.splitlines():
            m = re.search(r"judgement_id\s*[:：]\s*([A-Za-z0-9_.:-]+)", line)
            if m:
                current_jid = m.group(1)
                section_citations.setdefault(current_jid, set())
            if current_jid:
                section_citations[current_jid].update(self._citation_re.findall(line))

        if set(section_citations.keys()) != expected_judgement_ids:
            passed = False
            reasons.append("Markdown judgement sections must cover each judgement_id exactly once.")

        seen: set[str] = set()
        for item in judgement_feedback:
            if item.judgement_id in seen:
                passed = False
                reasons.append(f"Duplicate judgement feedback: {item.judgement_id}")
            seen.add(item.judgement_id)
            if item.evaluation_window not in ALLOWED_EVALUATION_WINDOWS:
                passed = False
                reasons.append(f"Invalid evaluation window: {item.evaluation_window}")
            allowed = bundle_map.get(item.judgement_id, set())
            citations = section_citations.get(item.judgement_id, set())
            if allowed and not citations.issubset(allowed):
                passed = False
                reasons.append(f"{item.judgement_id} cites sources outside allowed_source_ids")

        if seen != expected_judgement_ids:
            passed = False
            reasons.append("judgement_feedback must align 1:1 with expected_judgement_ids")

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction="Rewrite with strict judgement/source alignment." if not passed else None,
            citation_coverage=coverage,
        )
