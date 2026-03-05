"""LLM cognition extraction engine with strict JSON validation and fallback."""

from __future__ import annotations

import json
import logging
from datetime import timezone

from pydantic import BaseModel, Field

from ai_trading_coach.domain.contracts import CognitionExtractionInput, CognitionExtractionOutput
from ai_trading_coach.domain.enums import HypothesisStatus, HypothesisType, ModelCallPurpose
from ai_trading_coach.domain.models import (
    BehavioralSignal,
    CognitionState,
    DailyLogNormalized,
    EmotionSignal,
    Hypothesis,
    ModelCallTrace,
    UserIntentSignal,
)
from ai_trading_coach.llm.provider import LLMProvider
from ai_trading_coach.modules.cognition.service import HeuristicCognitionExtractionEngine

logger = logging.getLogger(__name__)


class LLMHypothesisPayload(BaseModel):
    statement: str = Field(..., min_length=3)
    hypothesis_type: HypothesisType = Field(default=HypothesisType.OTHER)
    related_tickers: list[str] = Field(default_factory=list)
    time_horizon: str = Field(..., min_length=1)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    falsifiable_signals: list[str] = Field(default_factory=list)
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING)
    confidence: float = Field(..., ge=0, le=1)


class LLMBehaviorSignalPayload(BaseModel):
    signal_type: str = Field(..., min_length=1)
    intensity: float = Field(default=0.5, ge=0, le=1)
    polarity: str = Field(default="neutral")
    evidence: str = Field(..., min_length=1)


class LLMEmotionSignalPayload(BaseModel):
    emotion: str = Field(..., min_length=1)
    intensity: float = Field(default=0.5, ge=0, le=1)
    evidence: str = Field(..., min_length=1)


class LLMIntentSignalPayload(BaseModel):
    question: str = Field(..., min_length=1)
    priority: int = Field(default=3, ge=1, le=5)


class LLMCognitionPayload(BaseModel):
    core_judgements: list[str] = Field(...)
    hypotheses: list[LLMHypothesisPayload] = Field(..., min_length=1)
    risk_concerns: list[str] = Field(...)
    outside_opportunities: list[str] = Field(...)
    deliberate_no_trade_decisions: list[str] = Field(...)
    explicit_rules: list[str] = Field(...)
    fuzzy_tendencies: list[str] = Field(...)
    fact_statements: list[str] = Field(...)
    subjective_statements: list[str] = Field(...)
    behavioral_signals: list[LLMBehaviorSignalPayload] = Field(...)
    emotion_signals: list[LLMEmotionSignalPayload] = Field(...)
    user_intent_signals: list[LLMIntentSignalPayload] = Field(...)


class LLMCognitionExtractionEngine:
    """Use LLM JSON extraction with strict schema validation and deterministic fallback."""

    def __init__(
        self,
        provider: LLMProvider | None,
        timeout_seconds: float,
        fallback_engine: HeuristicCognitionExtractionEngine | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.fallback_engine = fallback_engine or HeuristicCognitionExtractionEngine()

    def extract(self, data: CognitionExtractionInput) -> CognitionExtractionOutput:
        if self.provider is None:
            return self._fallback(data, reason="llm_provider_unavailable")

        input_summary = self._input_summary(data)
        try:
            messages = self._build_messages(data)
            payload = self.provider.chat_json(
                schema_name="cognition_state.v1",
                messages=messages,
                timeout=self.timeout_seconds,
            )
            contract = LLMCognitionPayload.model_validate(payload)
            cognition_state = self._to_cognition_state(data, contract)
            output = CognitionExtractionOutput(cognition_state=cognition_state)
            self._attach_trace(
                output=output,
                purpose=ModelCallPurpose.COGNITION_EXTRACTION,
                input_summary=input_summary,
                output_summary=f"ok; hypotheses={len(cognition_state.hypotheses)}",
            )
            output.extensions["llm_engine"] = "enabled"
            return output
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_cognition_fallback reason=%s", exc)
            return self._fallback(data, reason=str(exc), input_summary=input_summary)

    def _fallback(
        self,
        data: CognitionExtractionInput,
        reason: str,
        input_summary: str | None = None,
    ) -> CognitionExtractionOutput:
        output = self.fallback_engine.extract(data)
        output.extensions["llm_engine"] = "fallback"
        output.extensions["llm_fallback_reason"] = reason
        self._attach_trace(
            output=output,
            purpose=ModelCallPurpose.COGNITION_EXTRACTION,
            input_summary=input_summary or self._input_summary(data),
            output_summary=f"fallback; reason={reason}",
        )
        return output

    def _build_messages(self, data: CognitionExtractionInput) -> list[dict[str, str]]:
        log = data.normalized_log
        raw_payload = {
            "log_id": log.log_id,
            "user_id": log.user_id,
            "log_date": log.log_date.isoformat(),
            "traded_tickers": log.traded_tickers,
            "mentioned_tickers": log.mentioned_tickers,
            "user_state": log.user_state.model_dump(mode="json"),
            "market_context": log.market_context.model_dump(mode="json"),
            "trade_events": [event.model_dump(mode="json") for event in log.trade_events],
            "scan_signals": log.scan_signals.model_dump(mode="json"),
            "reflection": log.reflection.model_dump(mode="json"),
            "ai_directives": log.ai_directives,
            "raw_text": log.raw_text,
        }

        hypothesis_types = ", ".join(item.value for item in HypothesisType)
        hypothesis_statuses = ", ".join(item.value for item in HypothesisStatus)

        system_prompt = (
            "You are an extraction engine for trading cognition logs. "
            "Return JSON only. Do not return markdown or explanations. "
            "The JSON object MUST include these top-level fields: "
            "core_judgements, hypotheses, risk_concerns, outside_opportunities, "
            "deliberate_no_trade_decisions, explicit_rules, fuzzy_tendencies, "
            "fact_statements, subjective_statements, behavioral_signals, emotion_signals, "
            "user_intent_signals.\n"
            "For each hypotheses item, include: statement, hypothesis_type, related_tickers, "
            "time_horizon, evidence_for, evidence_against, falsifiable_signals, status, confidence.\n"
            f"Allowed hypothesis_type: {hypothesis_types}.\n"
            f"Allowed status: {hypothesis_statuses}.\n"
            "confidence must be a float in [0, 1], and time_horizon is required.\n"
            "Keep values concise and grounded in the input journal."
        )

        user_prompt = (
            "Extract cognition state from this normalized daily log payload:\n"
            f"{json.dumps(raw_payload, ensure_ascii=False, indent=2)}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _to_cognition_state(self, data: CognitionExtractionInput, payload: LLMCognitionPayload) -> CognitionState:
        log = data.normalized_log

        hypotheses: list[Hypothesis] = []
        for idx, hypothesis in enumerate(payload.hypotheses):
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_{log.log_id}_{idx}",
                    statement=hypothesis.statement,
                    hypothesis_type=hypothesis.hypothesis_type,
                    related_tickers=self._normalize_tickers(hypothesis.related_tickers, log),
                    timeframe_hint=hypothesis.time_horizon,
                    evidence_for=self._dedupe(hypothesis.evidence_for),
                    evidence_against=self._dedupe(hypothesis.evidence_against),
                    falsifiable_signals=self._dedupe(hypothesis.falsifiable_signals),
                    status=hypothesis.status,
                    confidence=hypothesis.confidence,
                )
            )

        cognition_payload = {
            "cognition_id": f"cog_{log.log_id}",
            "log_id": log.log_id,
            "user_id": log.user_id,
            "as_of_date": log.log_date,
            "core_judgements": self._dedupe(payload.core_judgements),
            "hypotheses": hypotheses,
            "risk_concerns": self._dedupe(payload.risk_concerns),
            "outside_opportunities": self._dedupe(payload.outside_opportunities),
            "deliberate_no_trade_decisions": self._dedupe(payload.deliberate_no_trade_decisions),
            "explicit_rules": self._dedupe(payload.explicit_rules),
            "fuzzy_tendencies": self._dedupe(payload.fuzzy_tendencies),
            "fact_statements": self._dedupe(payload.fact_statements),
            "subjective_statements": self._dedupe(payload.subjective_statements),
            "behavioral_signals": [
                BehavioralSignal(
                    signal_id=f"bs_{log.log_id}_{idx}",
                    signal_type=item.signal_type,
                    intensity=item.intensity,
                    polarity=item.polarity,
                    evidence=item.evidence,
                )
                for idx, item in enumerate(payload.behavioral_signals)
            ],
            "emotion_signals": [
                EmotionSignal(
                    signal_id=f"es_{log.log_id}_{idx}",
                    emotion=item.emotion,
                    intensity=item.intensity,
                    evidence=item.evidence,
                )
                for idx, item in enumerate(payload.emotion_signals)
            ],
            "user_intent_signals": [
                UserIntentSignal(
                    intent_id=f"intent_{idx}",
                    question=item.question,
                    priority=item.priority,
                )
                for idx, item in enumerate(payload.user_intent_signals)
            ],
        }
        return CognitionState.model_validate(cognition_payload)

    def _normalize_tickers(self, tickers: list[str], data: DailyLogNormalized) -> list[str]:
        known = {
            *data.traded_tickers,
            *data.mentioned_tickers,
            *(ticker.strip() for ticker in tickers if ticker.strip()),
        }
        return sorted(known)

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            text = item.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _input_summary(self, data: CognitionExtractionInput) -> str:
        log = data.normalized_log
        return (
            f"log_id={log.log_id}; "
            f"trade_events={len(log.trade_events)}; "
            f"ai_directives={len(log.ai_directives)}"
        )

    def _attach_trace(
        self,
        output: CognitionExtractionOutput,
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


__all__ = ["LLMCognitionExtractionEngine"]
