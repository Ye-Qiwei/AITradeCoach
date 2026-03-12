"""Judge stage with deterministic checks and LLM semantic validation."""

from __future__ import annotations

import re

from ai_trading_coach.domain.agent_models import JudgeVerdict
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.judgement_models import ALLOWED_EVALUATION_WINDOWS, DailyJudgementFeedback
from ai_trading_coach.domain.llm_output_adapters import judge_verdict_contract_to_domain
from ai_trading_coach.domain.llm_output_contracts import JudgeVerdictContract
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
        rule_verdict = self._rule_check(report_markdown=report_markdown, evidence_packet=evidence_packet, judgement_feedback=[DailyJudgementFeedback.model_validate(i) for i in judge_context.get("judgement_feedback", [])], expected_judgement_ids=set(judge_context.get("expected_judgement_ids", [])), bundles=judge_context.get("judgement_bundles", []), global_sources=set((judge_context.get("global_source_index", {}) or {}).keys()))
        if not rule_verdict.passed:
            return rule_verdict, None
        prompt = self.prompt_manager.load_active(self.prompt_name)
        messages = self.prompt_manager.build_messages(system_prompt=prompt.system_prompt, payload={"report_markdown": report_markdown, "judge_context": judge_context, "rule_verdict": rule_verdict.model_dump(mode="json")})
        llm_contract, trace = self.gateway.invoke_structured(schema=JudgeVerdictContract, messages=messages, purpose=ModelCallPurpose.COGNITION_EVALUATION, prompt_version=f"{prompt.prompt_name}.{prompt.version}", input_summary=f"sources={len(evidence_packet.source_registry)}", output_summary_builder=lambda out: f"passed={out.passed};reasons={len(out.reasons)}")
        llm_verdict = judge_verdict_contract_to_domain(llm_contract)
        return JudgeVerdict(passed=rule_verdict.passed and llm_verdict.passed, reasons=[*rule_verdict.reasons, *llm_verdict.reasons], rewrite_instruction=llm_verdict.rewrite_instruction or rule_verdict.rewrite_instruction, contradiction_flags=[*rule_verdict.contradiction_flags, *llm_verdict.contradiction_flags], citation_coverage=rule_verdict.citation_coverage), trace

    def _rule_check(self, *, report_markdown: str, evidence_packet: EvidencePacket, judgement_feedback: list[DailyJudgementFeedback], expected_judgement_ids: set[str], bundles: object, global_sources: set[str]) -> JudgeVerdict:
        source_ids = {item.source_id for item in evidence_packet.source_registry}
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
            if not set(item.source_ids).issubset(source_ids) or not set(item.source_ids).issubset(global_sources):
                passed = False
                reasons.append(f"{item.judgement_id} references unknown source_ids")
            if set(item.source_ids) != section_citations.get(item.judgement_id, set()):
                passed = False
                reasons.append(f"{item.judgement_id} source_ids mismatch markdown citations")
            allowed = bundle_map.get(item.judgement_id, set())
            if allowed and not set(item.source_ids).issubset(allowed):
                passed = False
                reasons.append(f"{item.judgement_id} cites sources outside allowed_source_ids")

        if seen != expected_judgement_ids:
            passed = False
            reasons.append("judgement_feedback must align 1:1 with expected_judgement_ids")

        return JudgeVerdict(passed=passed, reasons=reasons, rewrite_instruction="Rewrite with strict judgement/source alignment." if not passed else None, contradiction_flags=[], citation_coverage=coverage)
