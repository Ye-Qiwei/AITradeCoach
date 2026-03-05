"""Daily review report generator implementation."""

from __future__ import annotations

from ai_trading_coach.domain.contracts import ReportGeneratorInput, ReportGeneratorOutput
from ai_trading_coach.domain.models import DailyReviewReport, ReportSection


class StructuredReviewReportGenerator:
    """Generate coach-style markdown report with fixed section structure."""

    def __init__(self, prompt_version: str = "report_generation.v1") -> None:
        self.prompt_version = prompt_version

    def generate(self, data: ReportGeneratorInput) -> ReportGeneratorOutput:
        report_date = data.evaluation.as_of_date

        sections = [
            ReportSection(title="今日结论", content=data.evaluation.summary),
            ReportSection(title="关键事实", content=self._key_facts(data)),
            ReportSection(title="你看对了什么", content=self._list_or_default(data.evaluation.strengths, "暂无明显优势点")),
            ReportSection(title="你看错了什么", content=self._list_or_default(data.evaluation.mistakes, "暂无明显错误点")),
            ReportSection(
                title="哪些判断可能是超前洞察",
                content=self._list_or_default(data.evaluation.ahead_of_market_observations, "暂无超前洞察信号"),
            ),
            ReportSection(title="执行/仓位/风控评估", content=self._execution_text(data)),
            ReportSection(title="今天最值得强化的一条规则", content=self._top_rule(data)),
            ReportSection(title="今天最需要警惕的一条风险", content=self._top_risk(data)),
            ReportSection(title="下一步观察清单", content=self._list_or_default(data.evaluation.follow_up_signals, "暂无")),
            ReportSection(title="策略修正建议 / 预警", content=self._strategy_adjustments(data)),
        ]

        key_takeaways = [
            data.evaluation.summary,
            self._top_rule(data),
            self._top_risk(data),
        ]

        markdown_body = self._render_markdown(report_date.isoformat(), sections)

        report = DailyReviewReport(
            report_id=f"report_{data.evaluation.evaluation_id}",
            user_id=data.evaluation.user_id,
            report_date=report_date,
            title=f"Daily Trading Cognition Review - {report_date.isoformat()}",
            sections=sections,
            key_takeaways=key_takeaways,
            next_watchlist=data.evaluation.follow_up_signals[:8],
            strategy_adjustments=data.evaluation.execution_assessment.notes,
            risk_alerts=data.evaluation.warning_flags,
            generated_prompt_version=self.prompt_version,
            markdown_body=markdown_body,
        )
        return ReportGeneratorOutput(report=report)

    def _key_facts(self, data: ReportGeneratorInput) -> str:
        facts: list[str] = []
        facts.append(f"证据完整度: {data.evidence_packet.completeness_score:.2f}")
        facts.append(f"分析窗口数: {len(data.window_decision.selected_windows)}")
        facts.append(f"持仓数量: {len(data.position_snapshot.holdings)}")
        facts.append(f"证据来源数: {len(data.evidence_packet.source_registry)}")

        if data.pnl_snapshot.total_pnl is not None:
            facts.append(f"总盈亏: {data.pnl_snapshot.total_pnl:.2f} {data.pnl_snapshot.currency}")
        elif data.pnl_snapshot.realized_pnl:
            facts.append(f"已实现盈亏: {data.pnl_snapshot.realized_pnl:.2f} {data.pnl_snapshot.currency}")

        if data.evidence_packet.missing_requirements:
            facts.append(f"证据缺口: {', '.join(data.evidence_packet.missing_requirements[:3])}")
        layered_facts = data.evaluation.extensions.get("layers", {}).get("facts", [])
        if isinstance(layered_facts, list):
            facts.extend(str(item) for item in layered_facts[:3])

        return self._to_bullets(facts)

    def _execution_text(self, data: ReportGeneratorInput) -> str:
        execution = data.evaluation.execution_assessment
        lines = [
            f"纪律性评分: {execution.discipline_score:.2f}",
            f"仓位合理性评分: {execution.position_sizing_score:.2f}",
            f"风控评分: {execution.risk_control_score:.2f}",
            *execution.notes,
        ]
        return self._to_bullets(lines)

    def _top_rule(self, data: ReportGeneratorInput) -> str:
        if data.evaluation.strengths:
            return data.evaluation.strengths[0]
        return "优先使用事实-解释-评价三层复盘，避免情绪先行。"

    def _top_risk(self, data: ReportGeneratorInput) -> str:
        if data.evaluation.warning_flags:
            return data.evaluation.warning_flags[0]
        return "短期价格波动可能误导长期 thesis 判断。"

    def _strategy_adjustments(self, data: ReportGeneratorInput) -> str:
        suggestions: list[str] = []
        if data.window_decision.follow_up_needed:
            suggestions.append("当前为暂判，请在推荐复盘日再次验证核心 thesis。")
        if data.evidence_packet.missing_requirements:
            suggestions.append("先补齐缺失证据，再做仓位放大或策略切换。")
        if not suggestions:
            suggestions.append("保持当前策略节奏，优先执行既定风控规则。")
        if data.evidence_packet.source_registry:
            source_text = ", ".join(sorted({src.provider for src in data.evidence_packet.source_registry})[:4])
            suggestions.append(f"证据来源交叉检查: {source_text}")
        return self._to_bullets(suggestions)

    def _list_or_default(self, items: list[str], default: str) -> str:
        if not items:
            return f"- {default}"
        return self._to_bullets(items)

    def _to_bullets(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if item)

    def _render_markdown(self, date_text: str, sections: list[ReportSection]) -> str:
        lines = [f"# Daily Review Report - {date_text}"]
        for idx, section in enumerate(sections, start=1):
            lines.append(f"\n## {idx}. {section.title}")
            lines.append(section.content)
        return "\n".join(lines).strip() + "\n"


# Backward-compatible alias
PlaceholderReviewReportGenerator = StructuredReviewReportGenerator
