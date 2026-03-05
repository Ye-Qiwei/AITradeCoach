"""Optional Gemini advisor for PromptOps proposal refinement."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from ai_trading_coach.domain.enums import ModelCallPurpose
from ai_trading_coach.domain.models import ModelCallTrace
from ai_trading_coach.prompts.registry import PromptRegistry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GeminiPromptOpsAdvisor:
    """Refine proposal text with Gemini; safe to disable and fall back heuristics."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: int,
        prompt_registry: PromptRegistry,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.prompt_registry = prompt_registry

    def suggest(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, ModelCallTrace]:
        started_at = _utc_now()
        started_perf = time.perf_counter()
        call_id = f"model_promptops_{int(started_at.timestamp() * 1000)}"
        input_summary = (
            f"keys={sorted(payload.keys())}; "
            f"report_quality={payload.get('report_quality_score')}; "
            f"scope={payload.get('scope_hint')}"
        )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            version, prompt_template = self.prompt_registry.load_active("self_improvement")
            prompt = self._build_prompt(prompt_template, payload)

            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                timeout=self.timeout_seconds,
                temperature=0.1,
            )
            response = model.invoke([HumanMessage(content=prompt)])
            content = response.content if isinstance(response.content, str) else str(response.content)
            parsed = self._parse_json_payload(content)
            usage = self._extract_usage(response)
            latency_ms = int((time.perf_counter() - started_perf) * 1000)
            trace = ModelCallTrace(
                call_id=call_id,
                purpose=ModelCallPurpose.IMPROVEMENT_PROPOSAL,
                model_name=self.model_name,
                started_at=started_at,
                ended_at=_utc_now(),
                input_summary=input_summary,
                output_summary=f"ok; prompt={version}; keys={sorted(parsed.keys()) if parsed else []}",
                token_in=usage.get("input_tokens"),
                token_out=usage.get("output_tokens"),
                latency_ms=latency_ms,
            )
            return parsed, trace
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started_perf) * 1000)
            trace = ModelCallTrace(
                call_id=call_id,
                purpose=ModelCallPurpose.IMPROVEMENT_PROPOSAL,
                model_name=self.model_name,
                started_at=started_at,
                ended_at=_utc_now(),
                input_summary=input_summary,
                output_summary=f"error: {exc}",
                token_in=None,
                token_out=None,
                latency_ms=latency_ms,
            )
            return None, trace

    def _build_prompt(self, prompt_template: str, payload: dict[str, Any]) -> str:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            f"{prompt_template.strip()}\n\n"
            "请基于输入生成 JSON（不要输出 markdown），字段如下：\n"
            "{\n"
            '  "problem_statement": "string",\n'
            '  "candidate_change": "string",\n'
            '  "expected_benefit": "string",\n'
            '  "success_metrics": ["string"],\n'
            '  "risk_level": 1-5\n'
            "}\n\n"
            f"输入:\n{payload_text}\n"
        )

    def _parse_json_payload(self, content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return None
        return None

    def _extract_usage(self, response: Any) -> dict[str, int | None]:
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage_metadata", {}) if isinstance(meta, dict) else {}
        in_tokens = usage.get("input_tokens")
        out_tokens = usage.get("output_tokens")
        return {
            "input_tokens": int(in_tokens) if isinstance(in_tokens, int) else None,
            "output_tokens": int(out_tokens) if isinstance(out_tokens, int) else None,
        }

