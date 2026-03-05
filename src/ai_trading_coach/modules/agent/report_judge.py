"""Judge stage for citation coverage, intent fit, and contradiction checks."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ai_trading_coach.domain.agent_models import JudgeVerdict, Plan
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import EvidencePacket, ModelCallTrace
from ai_trading_coach.errors import LLMOutputValidationError
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.trace_utils import build_model_trace


class ReportJudge:
    schema_name = "judge_verdict.v1"
    prompt_version = "judge.v1"
    _citation_re = re.compile(r"\[source:([A-Za-z0-9_.:-]+)\]")

    def __init__(self, provider: LLMProvider, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        *,
        report_markdown: str,
        judge_context: dict[str, object],
        intent: list[str],
        evidence_packet: EvidencePacket,
        plan: Plan,
    ) -> tuple[JudgeVerdict, ModelCallTrace | None]:
        rule_verdict = self._rule_check(report_markdown=report_markdown, evidence_packet=evidence_packet)
        messages = self._build_messages(
            report_markdown=report_markdown,
            judge_context=judge_context,
            intent=intent,
            plan=plan,
            rule_verdict=rule_verdict,
        )
        payload = self.provider.chat_json(
            schema_name=self.schema_name,
            messages=messages,
            timeout=self.timeout_seconds,
            prompt_version=self.prompt_version,
        )
        try:
            llm_verdict = JudgeVerdict.model_validate(payload)
        except ValidationError as exc:
            raise self._validation_error(exc) from exc

        merged = self._merge_verdict(rule_verdict=rule_verdict, llm_verdict=llm_verdict)
        trace = build_model_trace(
            purpose=ModelCallPurpose.COGNITION_EVALUATION,
            input_summary=f"intent={len(intent)}; sources={len(evidence_packet.source_registry)}",
            output_summary=f"passed={merged.passed}; coverage={merged.citation_coverage:.2f}",
            provider_record=getattr(self.provider, "last_call", None),
        )
        return merged, trace

    def _merge_verdict(self, *, rule_verdict: JudgeVerdict, llm_verdict: JudgeVerdict) -> JudgeVerdict:
        merged_reasons = [*rule_verdict.reasons, *llm_verdict.reasons]
        rewrite_instruction = llm_verdict.rewrite_instruction or rule_verdict.rewrite_instruction
        contradiction_flags = [*rule_verdict.contradiction_flags, *llm_verdict.contradiction_flags]
        return JudgeVerdict(
            passed=rule_verdict.passed and llm_verdict.passed,
            reasons=merged_reasons,
            rewrite_instruction=rewrite_instruction,
            contradiction_flags=contradiction_flags,
            citation_coverage=min(rule_verdict.citation_coverage, llm_verdict.citation_coverage)
            if llm_verdict.citation_coverage > 0
            else rule_verdict.citation_coverage,
        )

    def _rule_check(self, *, report_markdown: str, evidence_packet: EvidencePacket) -> JudgeVerdict:
        lines = [line.strip() for line in report_markdown.splitlines() if line.strip()]
        fact_lines = [line for line in lines if line.startswith("-")]
        source_ids = {item.source_id for item in evidence_packet.source_registry}

        cited_lines = 0
        unknown_ids: set[str] = set()
        for line in fact_lines:
            refs = self._citation_re.findall(line)
            if refs:
                cited_lines += 1
            for ref in refs:
                if ref not in source_ids:
                    unknown_ids.add(ref)

        coverage = 1.0 if not fact_lines else round(cited_lines / len(fact_lines), 4)
        reasons: list[str] = []
        passed = True
        rewrite_instruction = None

        if coverage < 1.0:
            passed = False
            reasons.append("Citation coverage below 100% for bullet facts.")
            rewrite_instruction = "Add [source:<source_id>] to every factual bullet."

        if unknown_ids:
            passed = False
            reasons.append(f"Unknown source_id referenced: {sorted(unknown_ids)}")
            rewrite_instruction = "Use only source ids present in source_index."

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction=rewrite_instruction,
            contradiction_flags=[],
            citation_coverage=coverage,
        )

    def _validation_error(self, exc: ValidationError) -> LLMOutputValidationError:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()[:5]
        )
        return LLMOutputValidationError(f"Schema validation failed for {self.schema_name}: {detail}")

    def _build_messages(
        self,
        *,
        report_markdown: str,
        judge_context: dict[str, object],
        intent: list[str],
        plan: Plan,
        rule_verdict: JudgeVerdict,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the judge stage of a trading review pipeline. Return JSON only. "
            "Output schema judge_verdict.v1 with passed/reasons/rewrite_instruction. "
            "Fail when report misses user intent, has contradictory claims, or unsupported statements."
        )
        user_payload = {
            "report_markdown": report_markdown,
            "intent": intent,
            "plan": plan.model_dump(mode="json"),
            "judge_context": judge_context,
            "rule_verdict": rule_verdict.model_dump(mode="json"),
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
