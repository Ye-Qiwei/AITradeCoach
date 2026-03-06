"""Judge stage with hard checks for citations and judgement feedback completeness."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace


class ReportJudge:
    schema_name = "judge_verdict.v2"
    prompt_version = "judge.v2"
    _citation_re = re.compile(r"\[source:([A-Za-z0-9_.:-]+)\]")

    def __init__(self, provider: LLMProvider, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def evaluate(self, *, report_markdown: str, judge_context: dict[str, object], evidence_packet: EvidencePacket):
        rule_verdict = self._rule_check(
            report_markdown=report_markdown,
            evidence_packet=evidence_packet,
            judgement_feedback=[DailyJudgementFeedback.model_validate(i) for i in judge_context.get("judgement_feedback", [])],
        )
        messages = self._build_messages(report_markdown=report_markdown, judge_context=judge_context, rule_verdict=rule_verdict)
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            llm_verdict = JudgeVerdict.model_validate(payload)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
            raise LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}") from exc

        merged = JudgeVerdict(
            passed=rule_verdict.passed and llm_verdict.passed,
            reasons=[*rule_verdict.reasons, *llm_verdict.reasons],
            rewrite_instruction=llm_verdict.rewrite_instruction or rule_verdict.rewrite_instruction,
            contradiction_flags=[*rule_verdict.contradiction_flags, *llm_verdict.contradiction_flags],
            citation_coverage=min(rule_verdict.citation_coverage, llm_verdict.citation_coverage or 1.0),
        )
        trace = build_model_trace(
            purpose=ModelCallPurpose.COGNITION_EVALUATION,
            input_summary=f"sources={len(evidence_packet.source_registry)}",
            output_summary=f"passed={merged.passed}; coverage={merged.citation_coverage:.2f}",
            provider_record=getattr(self.provider, "last_call", None),
        )
        return merged, trace

    def _rule_check(self, *, report_markdown: str, evidence_packet: EvidencePacket, judgement_feedback: list[DailyJudgementFeedback]) -> JudgeVerdict:
        source_ids = {item.source_id for item in evidence_packet.source_registry}
        lines = [line.strip() for line in report_markdown.splitlines() if line.strip().startswith("-")]
        cited = sum(1 for line in lines if self._citation_re.findall(line))
        coverage = 1.0 if not lines else cited / len(lines)
        reasons: list[str] = []
        passed = True

        if coverage < 1.0:
            passed = False
            reasons.append("Citation coverage below 100% for bullet lines.")
        if not judgement_feedback:
            passed = False
            reasons.append("Missing judgement feedback section.")
        for item in judgement_feedback:
            if item.evaluation_window not in ALLOWED_EVALUATION_WINDOWS:
                passed = False
                reasons.append(f"Invalid evaluation window: {item.evaluation_window}")
            if not item.source_ids:
                passed = False
                reasons.append(f"{item.judgement_id} missing source ids")
            unknown = [src for src in item.source_ids if src not in source_ids]
            if unknown:
                passed = False
                reasons.append(f"{item.judgement_id} references unknown sources: {unknown}")

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction="Fix citations/judgement_feedback fields." if not passed else None,
            contradiction_flags=[],
            citation_coverage=coverage,
        )

    def _build_messages(self, *, report_markdown: str, judge_context: dict[str, object], rule_verdict: JudgeVerdict):
        payload = {
            "report_markdown": report_markdown,
            "judge_context": judge_context,
            "rule_verdict": rule_verdict.model_dump(mode="json"),
        }
        return [
            {"role": "system", "content": "Review consistency between report conclusions and evidence. Return JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
