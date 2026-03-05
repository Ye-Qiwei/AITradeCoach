"""LLM review report generator with strict markdown contract and fallback."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timezone

from pydantic import BaseModel, Field

from ai_trading_coach.domain.contracts import ReportGeneratorInput, ReportGeneratorOutput
from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import DailyReviewReport, ModelCallTrace, ReportSection
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.report.service import StructuredReviewReportGenerator

logger = logging.getLogger(__name__)


class MarkdownSectionContract(BaseModel):
    index: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class MarkdownReportContract(BaseModel):
    report_date: date
    sections: list[MarkdownSectionContract] = Field(min_length=10, max_length=10)


class LLMReviewReportGenerator:
    """Generate markdown report with LLM and deterministic heuristic fallback."""

    _required_titles: list[str] = [
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
    ]

    _factual_titles: set[str] = {
        "关键事实",
        "你看对了什么",
        "你看错了什么",
        "哪些判断可能是超前洞察",
        "执行/仓位/风控评估",
        "下一步观察清单",
        "策略修正建议 / 预警",
    }

    _citation_pattern = re.compile(r"\[source:[^\]|]+\|[^\]]+\]")

    def __init__(
        self,
        provider: LLMProvider | None,
        timeout_seconds: float,
        fallback_generator: StructuredReviewReportGenerator | None = None,
        prompt_version: str = "report_generation.llm_v1",
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.fallback_generator = fallback_generator or StructuredReviewReportGenerator()
        self.prompt_version = prompt_version

    def generate(self, data: ReportGeneratorInput) -> ReportGeneratorOutput:
        if self.provider is None:
            return self._fallback(data, reason="llm_provider_unavailable")

        input_summary = self._input_summary(data)
        try:
            messages = self._build_messages(data)
            markdown = self.provider.chat_text(messages)
            report = self._build_report(data, markdown)
            output = ReportGeneratorOutput(report=report)
            output.extensions["llm_engine"] = "enabled"
            self._attach_trace(
                output=output,
                purpose=ModelCallPurpose.REPORT_GENERATION,
                input_summary=input_summary,
                output_summary=f"ok; markdown_chars={len(markdown)}",
            )
            return output
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_report_fallback reason=%s", exc)
            return self._fallback(data, reason=str(exc), input_summary=input_summary)

    def _fallback(
        self,
        data: ReportGeneratorInput,
        reason: str,
        input_summary: str | None = None,
    ) -> ReportGeneratorOutput:
        output = self.fallback_generator.generate(data)
        output.extensions["llm_engine"] = "fallback"
        output.extensions["llm_fallback_reason"] = reason
        self._attach_trace(
            output=output,
            purpose=ModelCallPurpose.REPORT_GENERATION,
            input_summary=input_summary or self._input_summary(data),
            output_summary=f"fallback; reason={reason}",
        )
        return output

    def _build_messages(self, data: ReportGeneratorInput) -> list[dict[str, str]]:
        source_index = [
            {
                "source_id": source.source_id,
                "uri": source.uri,
                "provider": source.provider,
                "title": source.title,
            }
            for source in data.evidence_packet.source_registry
        ]

        payload = {
            "evaluation": data.evaluation.model_dump(mode="json"),
            "position_snapshot": data.position_snapshot.model_dump(mode="json"),
            "pnl_snapshot": data.pnl_snapshot.model_dump(mode="json"),
            "window_decision": data.window_decision.model_dump(mode="json"),
            "trade_ledger": data.trade_ledger.model_dump(mode="json") if data.trade_ledger else None,
            "recalled_memories": (
                [record.model_dump(mode="json") for record in data.relevant_memories.records]
                if data.relevant_memories is not None
                else []
            ),
            "focus_points": data.user_focus_points,
            "evidence_sources": source_index,
        }

        section_title_lines = "\n".join(
            f"{idx}. {title}" for idx, title in enumerate(self._required_titles, start=1)
        )

        system_prompt = (
            "You are a trading cognition coach. Return markdown only, no JSON and no extra commentary.\n"
            "Use exactly this structure and section order:\n"
            f"{section_title_lines}\n"
            "Markdown contract:\n"
            "1) Start with '# Daily Review Report - YYYY-MM-DD'.\n"
            "2) Use exact section headings in order as '## N. 标题'.\n"
            "3) For each factual bullet/statement, append citation tag '[source:source_id|uri]'.\n"
            "4) Do not invent source_id or uri; only use provided evidence_sources.\n"
            "5) Keep content actionable and concise."
        )

        user_prompt = (
            "Generate the report from this payload:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_report(self, data: ReportGeneratorInput, markdown: str) -> DailyReviewReport:
        sections = self._validate_and_parse_markdown(markdown)

        section_map = {section.title: section for section in sections}
        key_takeaways = [
            self._first_line(section_map["今日结论"].content),
            self._first_line(section_map["今天最值得强化的一条规则"].content),
            self._first_line(section_map["今天最需要警惕的一条风险"].content),
        ]
        next_watchlist = self._extract_bullets(section_map["下一步观察清单"].content)
        strategy_adjustments = self._extract_bullets(section_map["策略修正建议 / 预警"].content)

        risk_alerts = data.evaluation.warning_flags[:]
        if not risk_alerts:
            risk_line = self._first_line(section_map["今天最需要警惕的一条风险"].content)
            if risk_line:
                risk_alerts = [risk_line]

        return DailyReviewReport(
            report_id=f"report_{data.evaluation.evaluation_id}",
            user_id=data.evaluation.user_id,
            report_date=data.evaluation.as_of_date,
            title=f"Daily Trading Cognition Review - {data.evaluation.as_of_date.isoformat()}",
            sections=sections,
            key_takeaways=[line for line in key_takeaways if line],
            next_watchlist=next_watchlist,
            strategy_adjustments=strategy_adjustments,
            risk_alerts=risk_alerts,
            generated_prompt_version=self.prompt_version,
            markdown_body=markdown.strip() + "\n",
        )

    def _validate_and_parse_markdown(self, markdown: str) -> list[ReportSection]:
        text = markdown.strip()
        if len(text) < 240:
            raise ValueError("markdown_output_too_short")

        lines = text.splitlines()
        if not lines or not lines[0].startswith("# Daily Review Report - "):
            raise ValueError("missing_report_title")
        report_date_text = lines[0].replace("# Daily Review Report - ", "", 1).strip()

        heading_pattern = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
        headings: list[tuple[int, str, int]] = []
        for idx, line in enumerate(lines):
            match = heading_pattern.match(line.strip())
            if match:
                headings.append((int(match.group(1)), match.group(2).strip(), idx))

        if len(headings) != len(self._required_titles):
            raise ValueError("missing_required_sections")

        ordered = sorted(headings, key=lambda item: item[0])
        contract = MarkdownReportContract.model_validate(
            {
                "report_date": report_date_text,
                "sections": [
                    {
                        "index": index,
                        "title": title,
                        "content": "\n".join(
                            lines[(line_no + 1) : (ordered[pos + 1][2] if pos + 1 < len(ordered) else len(lines))]
                        ).strip(),
                    }
                    for pos, (index, title, line_no) in enumerate(ordered)
                ],
            }
        )
        ordered_titles = [section.title for section in contract.sections]
        if ordered_titles != self._required_titles:
            raise ValueError("section_titles_mismatch")

        sections: list[ReportSection] = []
        for section in contract.sections:
            self._validate_citations(title=section.title, content=section.content)
            sections.append(ReportSection(title=section.title, content=section.content))

        return sections

    def _validate_citations(self, title: str, content: str) -> None:
        if title not in self._factual_titles:
            return

        bullet_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("-")]
        if not bullet_lines:
            if not self._citation_pattern.search(content):
                raise ValueError(f"citation_missing:{title}")
            return

        for line in bullet_lines:
            if self._citation_pattern.search(line) is None:
                raise ValueError(f"citation_missing:{title}")

    def _extract_bullets(self, content: str) -> list[str]:
        items: list[str] = []
        for line in content.splitlines():
            cleaned = line.strip()
            if not cleaned.startswith("-"):
                continue
            text = cleaned[1:].strip()
            text = self._citation_pattern.sub("", text).strip()
            if text:
                items.append(text)
        return items

    def _first_line(self, content: str) -> str:
        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith("-"):
                text = text[1:].strip()
            text = self._citation_pattern.sub("", text).strip()
            if text:
                return text
        return ""

    def _input_summary(self, data: ReportGeneratorInput) -> str:
        return (
            f"evaluation_id={data.evaluation.evaluation_id}; "
            f"evidence_sources={len(data.evidence_packet.source_registry)}; "
            f"focus_points={len(data.user_focus_points)}"
        )

    def _attach_trace(
        self,
        output: ReportGeneratorOutput,
        purpose: ModelCallPurpose,
        input_summary: str,
        output_summary: str,
    ) -> None:
        if self.provider is None:
            return
        record = getattr(self.provider, "last_call", None)
        if record is None:
            return

        call_id = f"model_{purpose.value}_{int(record.started_at.timestamp() * 1000)}"
        trace = ModelCallTrace(
            call_id=call_id,
            purpose=purpose,
            model_name=record.model_name,
            started_at=record.started_at.astimezone(timezone.utc),
            ended_at=record.ended_at.astimezone(timezone.utc),
            input_summary=input_summary,
            output_summary=output_summary if not record.error else f"{output_summary}; error={record.error}",
            token_in=record.token_in,
            token_out=record.token_out,
            latency_ms=record.latency_ms,
        )
        output.extensions["model_call_traces"] = [trace.model_dump(mode="json")]


__all__ = ["LLMReviewReportGenerator"]
