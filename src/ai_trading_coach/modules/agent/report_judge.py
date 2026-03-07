"""Judge stage with hard checks for citations and judgement feedback completeness."""

from __future__ import annotations

import re

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.models import EvidencePacket
from ai_trading_coach.llm.gateway import LangChainLLMGateway
from ai_trading_coach.modules.agent.prompting import PromptManager


class ReportJudge:
    prompt_name = "report_judging"
    _citation_re = re.compile(r"\[source:([A-Za-z0-9_.:-]+)\]")

    def __init__(self, gateway: LangChainLLMGateway, prompt_manager: PromptManager) -> None:
        self.gateway = gateway
        self.prompt_manager = prompt_manager

    def evaluate(self, *, report_markdown: str, judge_context: dict[str, object], evidence_packet: EvidencePacket):
        rule_verdict = self._rule_check(
            report_markdown=report_markdown,
            evidence_packet=evidence_packet,
            judgement_feedback=[DailyJudgementFeedback.model_validate(i) for i in judge_context.get("judgement_feedback", [])],
            expected_judgement_ids=set(judge_context.get("expected_judgement_ids", [])),
        )
        prompt = self.prompt_manager.load_active(self.prompt_name)
        messages = self.prompt_manager.build_messages(
            system_prompt=prompt.system_prompt,
            payload={
                "report_markdown": report_markdown,
                "judge_context": judge_context,
                "rule_verdict": rule_verdict.model_dump(mode="json"),
            },
        )
        llm_verdict, trace = self.gateway.invoke_structured(
            schema=JudgeVerdict,
            messages=messages,
            purpose=ModelCallPurpose.COGNITION_EVALUATION,
            prompt_version=f"{prompt.prompt_name}.{prompt.version}",
            input_summary=f"sources={len(evidence_packet.source_registry)}",
            output_summary_builder=lambda out: f"passed={out.passed};coverage={out.citation_coverage:.2f}",
        )

        merged = JudgeVerdict(
            passed=rule_verdict.passed and llm_verdict.passed,
            reasons=[*rule_verdict.reasons, *llm_verdict.reasons],
            rewrite_instruction=llm_verdict.rewrite_instruction or rule_verdict.rewrite_instruction,
            contradiction_flags=[*rule_verdict.contradiction_flags, *llm_verdict.contradiction_flags],
            citation_coverage=min(rule_verdict.citation_coverage, llm_verdict.citation_coverage or 1.0),
        )
        return merged, trace

    def _rule_check(self, *, report_markdown: str, evidence_packet: EvidencePacket, judgement_feedback: list[DailyJudgementFeedback], expected_judgement_ids: set[str]) -> JudgeVerdict:
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

        seen: set[str] = set()
        for item in judgement_feedback:
            if item.judgement_id in seen:
                passed = False
                reasons.append(f"Duplicate judgement feedback: {item.judgement_id}")
            seen.add(item.judgement_id)
            if item.evaluation_window not in ALLOWED_EVALUATION_WINDOWS:
                passed = False
                reasons.append(f"Invalid evaluation window: {item.evaluation_window}")
            unknown = [src for src in item.source_ids if src not in source_ids]
            if unknown:
                passed = False
                reasons.append(f"{item.judgement_id} references unknown sources: {unknown}")

        missing = sorted(expected_judgement_ids - seen)
        if missing:
            passed = False
            reasons.append(f"Missing judgement feedback IDs: {missing}")

        return JudgeVerdict(
            passed=passed,
            reasons=reasons,
            rewrite_instruction="Fix citations and judgement_feedback alignment." if not passed else None,
            contradiction_flags=[],
            citation_coverage=coverage,
        )
