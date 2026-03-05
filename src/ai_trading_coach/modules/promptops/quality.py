"""Report quality scoring for offline PromptOps evaluation."""

from __future__ import annotations

from ai_trading_coach.domain.models import DailyReviewReport, EvaluationResult, ReportQualityScore


class ReportQualityScorer:
    """Score report quality on structure, evidence linkage, actionability and tone."""

    REQUIRED_SECTIONS: tuple[str, ...] = (
        "今日结论",
        "关键事实",
        "你看对了什么",
        "你看错了什么",
        "哪些判断可能是超前洞察",
        "执行/仓位/风控评估",
        "今天最值得强化的一条规则",
        "今天最需要警惕的一条风险",
        "下一步观察清单",
        "策略修正建议 / 预警",
    )

    def score(self, report: DailyReviewReport, evaluation: EvaluationResult) -> ReportQualityScore:
        section_map = {section.title: section.content for section in report.sections}
        missing_sections = [title for title in self.REQUIRED_SECTIONS if title not in section_map]
        structure_score = max(0.0, 1.0 - (len(missing_sections) / len(self.REQUIRED_SECTIONS)))

        evidence_terms = 0
        key_facts = section_map.get("关键事实", "")
        if "证据完整度" in key_facts:
            evidence_terms += 1
        if "证据来源数" in key_facts:
            evidence_terms += 1
        if "交叉检查" in report.markdown_body:
            evidence_terms += 1
        layer_facts = evaluation.extensions.get("layers", {}).get("facts", [])
        if isinstance(layer_facts, list) and layer_facts:
            evidence_terms += 1
        evidence_traceability_score = evidence_terms / 4.0

        actionability_terms = 0
        if report.next_watchlist:
            actionability_terms += 1
        if report.strategy_adjustments:
            actionability_terms += 1
        if report.risk_alerts:
            actionability_terms += 1
        watch_section = section_map.get("下一步观察清单", "")
        if watch_section.count("- ") >= 2:
            actionability_terms += 1
        actionability_score = actionability_terms / 4.0

        symmetry_left = 1.0 if evaluation.strengths else 0.0
        symmetry_right = 1.0 if evaluation.mistakes else 0.0
        symmetry_score = (symmetry_left + symmetry_right) / 2.0

        risk_tone_score = 0.6 if report.risk_alerts else 0.4
        risky_words = ("必涨", "确定性100%", "无风险", "稳赚")
        if any(word in report.markdown_body for word in risky_words):
            risk_tone_score = max(0.0, risk_tone_score - 0.35)
        if evaluation.warning_flags:
            risk_tone_score = min(1.0, risk_tone_score + 0.2)

        overall_score = (
            structure_score * 0.2
            + evidence_traceability_score * 0.25
            + actionability_score * 0.2
            + symmetry_score * 0.2
            + risk_tone_score * 0.15
        )

        notes: list[str] = []
        if missing_sections:
            notes.append(f"缺失章节: {', '.join(missing_sections)}")
        if evidence_traceability_score < 0.7:
            notes.append("证据追溯性偏弱，建议增强来源链路说明。")
        if actionability_score < 0.7:
            notes.append("可执行建议密度不足，建议强化观察清单与策略动作。")
        if symmetry_score < 1.0:
            notes.append("奖惩不对称，建议同时强化优点与错误反馈。")

        return ReportQualityScore(
            score_id=f"rqs_{report.report_id}",
            overall_score=round(overall_score, 4),
            structure_score=round(structure_score, 4),
            evidence_traceability_score=round(evidence_traceability_score, 4),
            actionability_score=round(actionability_score, 4),
            symmetry_score=round(symmetry_score, 4),
            risk_tone_score=round(risk_tone_score, 4),
            missing_sections=missing_sections,
            notes=notes,
        )

