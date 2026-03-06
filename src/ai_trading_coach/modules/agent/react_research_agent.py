"""ReAct-style dynamic research agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.agent_models import CombinedParseResult
from ai_trading_coach.domain.react_models import ReActStep, ResearchSummary
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.agent.react_tools import ReactResearchTools


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _ActionDecision(BaseModel):
    thought: str = Field(default="")
    action: str = Field(default="finish_research")
    action_input: dict[str, Any] = Field(default_factory=dict)


class _SummaryPayload(BaseModel):
    investigation_summary: str = Field(default="")
    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ReActResearchAgent:
    schema_decision = "react_research_decision.v1"
    schema_summary = "react_research_summary.v1"
    prompt_version = "react-research.v1"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools: ReactResearchTools,
        settings: Settings,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.settings = settings

    def run(self, *, request_id: str, user_id: str, parse_result: CombinedParseResult) -> ResearchSummary:
        max_iterations = max(1, self.settings.react_max_iterations)
        max_failures = max(1, self.settings.react_max_tool_failures)
        steps: list[ReActStep] = []
        evidence_items = []
        failures = 0

        for idx in range(1, max_iterations + 1):
            decision = self._decide(
                parse_result=parse_result,
                steps=steps,
                evidence_count=len(evidence_items),
            )
            step = ReActStep(
                step_index=idx,
                thought=decision.thought,
                action=decision.action,
                action_input=decision.action_input,
                started_at=utc_now(),
            )

            if decision.action == "finish_research":
                if len(evidence_items) < self.settings.react_require_min_sources:
                    step.success = False
                    step.error_message = "insufficient_evidence"
                    step.observation_summary = (
                        f"Need >= {self.settings.react_require_min_sources} sources before finish."
                    )
                else:
                    step.observation_summary = "finish_research accepted"
                    step.ended_at = utc_now()
                    steps.append(step)
                    break
            else:
                outcome = self.tools.execute(
                    tool_name=decision.action,
                    arguments=decision.action_input,
                    step_id=f"{request_id}_react_{idx}",
                )
                step.success = outcome.success
                step.error_message = outcome.error_message
                step.observation_summary = outcome.observation_summary
                step.evidence_item_ids = [item.item_id for item in outcome.evidence_items]
                evidence_items.extend(outcome.evidence_items)
                if not outcome.success:
                    failures += 1
                    if failures >= max_failures:
                        step.observation_summary += " | max tool failures reached"
                        step.ended_at = utc_now()
                        steps.append(step)
                        break

            step.ended_at = utc_now()
            steps.append(step)

        summary = self._summarize(parse_result=parse_result, steps=steps, evidence_items=evidence_items)
        summary.research_id = f"research_{request_id}"
        return summary

    def _decide(
        self,
        *,
        parse_result: CombinedParseResult,
        steps: list[ReActStep],
        evidence_count: int,
    ) -> _ActionDecision:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a ReAct research agent. Choose one action from: "
                    "get_price_history, search_news, list_filings, get_macro_series, "
                    "finish_research. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "parse_result": parse_result.model_dump(mode="json"),
                        "evidence_count": evidence_count,
                        "recent_steps": [s.model_dump(mode="json") for s in steps[-4:]],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = self.provider.chat_json(
            schema_name=self.schema_decision,
            messages=messages,
            timeout=self.settings.llm_timeout_seconds,
            prompt_version=self.prompt_version,
        )
        return _ActionDecision.model_validate(payload)

    def _summarize(
        self,
        *,
        parse_result: CombinedParseResult,
        steps: list[ReActStep],
        evidence_items: list[Any],
    ) -> ResearchSummary:
        payload = self.provider.chat_json(
            schema_name=self.schema_summary,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize investigation into concise structured JSON.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "intent": [item.question for item in parse_result.cognition_state.user_intent_signals],
                            "step_observations": [s.observation_summary for s in steps],
                            "evidence_count": len(evidence_items),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            timeout=self.settings.llm_timeout_seconds,
            prompt_version=self.prompt_version,
        )
        summary_text = _SummaryPayload.model_validate(payload)
        return ResearchSummary(
            research_id="",
            investigation_summary=summary_text.investigation_summary,
            key_findings=summary_text.key_findings,
            open_questions=summary_text.open_questions,
            collected_evidence=evidence_items,
            evidence_item_ids=[item.item_id for item in evidence_items],
            tool_steps=steps,
        )
