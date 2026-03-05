"""Cognition extraction implementation."""

from __future__ import annotations

import re
from typing import Iterable

from ai_trading_coach.domain.contracts import CognitionExtractionInput, CognitionExtractionOutput
from ai_trading_coach.domain.enums import HypothesisStatus, HypothesisType
from ai_trading_coach.domain.models import (
    BehavioralSignal,
    CognitionState,
    EmotionSignal,
    Hypothesis,
    UserIntentSignal,
)


class HeuristicCognitionExtractionEngine:
    """Deterministic extractor used as round-2 baseline before LLM extraction."""

    _short_keywords = ("财报", "公告", "催化", "政策", "情绪修复", "事件")
    _mid_keywords = ("行业", "估值", "盈利", "周期", "景气", "资金面")
    _long_keywords = ("长期", "护城河", "成长", "结构", "长期化", "十年")

    def extract(self, data: CognitionExtractionInput) -> CognitionExtractionOutput:
        log = data.normalized_log
        hypotheses = self._build_hypotheses(log)

        behavior_signals = self._build_behavior_signals(log)
        emotion_signals = self._build_emotion_signals(log)
        intent_signals = self._build_intent_signals(log.ai_directives)

        cognition = CognitionState(
            cognition_id=f"cog_{log.log_id}",
            log_id=log.log_id,
            user_id=log.user_id,
            as_of_date=log.log_date,
            core_judgements=self._collect_core_judgements(log),
            hypotheses=hypotheses,
            risk_concerns=log.scan_signals.anxiety + [e.risk_note for e in log.trade_events if e.risk_note],
            outside_opportunities=log.scan_signals.fomo,
            deliberate_no_trade_decisions=log.scan_signals.not_trade,
            explicit_rules=log.reflection.lessons,
            fuzzy_tendencies=self._build_fuzzy_tendencies(log),
            fact_statements=log.reflection.facts,
            subjective_statements=self._collect_subjective_statements(log),
            behavioral_signals=behavior_signals,
            emotion_signals=emotion_signals,
            user_intent_signals=intent_signals,
        )
        return CognitionExtractionOutput(cognition_state=cognition)

    def _build_hypotheses(self, log) -> list[Hypothesis]:
        raw_candidates: list[str] = []
        raw_candidates.extend([event.reason for event in log.trade_events if event.reason])
        raw_candidates.extend(log.scan_signals.fomo)
        raw_candidates.extend(log.scan_signals.not_trade)

        hypotheses: list[Hypothesis] = []
        for idx, statement in enumerate(self._unique_nonempty(raw_candidates)):
            hypothesis_type = self._classify_type(statement)
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_{log.log_id}_{idx}",
                    statement=statement,
                    hypothesis_type=hypothesis_type,
                    related_tickers=self._extract_related_tickers(statement, log),
                    timeframe_hint=self._timeframe_hint(hypothesis_type),
                    evidence_for=[statement],
                    falsifiable_signals=self._default_falsifiable_signals(hypothesis_type),
                    status=HypothesisStatus.PENDING,
                    confidence=0.6,
                )
            )

        if not hypotheses:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_{log.log_id}_0",
                    statement="No explicit thesis found; require clarification in next journal",
                    hypothesis_type=HypothesisType.OTHER,
                    related_tickers=log.traded_tickers or log.mentioned_tickers,
                    timeframe_hint="unknown",
                    status=HypothesisStatus.PENDING,
                    confidence=0.3,
                )
            )

        return hypotheses

    def _build_behavior_signals(self, log) -> list[BehavioralSignal]:
        out: list[BehavioralSignal] = []

        if log.scan_signals.fomo:
            out.append(
                BehavioralSignal(
                    signal_id=f"bs_{log.log_id}_fomo",
                    signal_type="fomo",
                    intensity=min(1.0, 0.4 + 0.2 * len(log.scan_signals.fomo)),
                    polarity="negative",
                    evidence="; ".join(log.scan_signals.fomo),
                )
            )

        if log.scan_signals.not_trade:
            out.append(
                BehavioralSignal(
                    signal_id=f"bs_{log.log_id}_not_trade",
                    signal_type="disciplined_not_trade",
                    intensity=0.7,
                    polarity="positive",
                    evidence="; ".join(log.scan_signals.not_trade),
                )
            )

        if (log.user_state.stress or 0) >= 7 and log.trade_events:
            out.append(
                BehavioralSignal(
                    signal_id=f"bs_{log.log_id}_stress_trade",
                    signal_type="stress_affected_execution",
                    intensity=0.8,
                    polarity="negative",
                    evidence="high stress with active trading",
                )
            )

        return out

    def _build_emotion_signals(self, log) -> list[EmotionSignal]:
        if not log.user_state.emotion:
            return []

        intensity_base = 0.5
        if log.user_state.stress is not None:
            intensity_base = min(1.0, 0.1 * log.user_state.stress)
        return [
            EmotionSignal(
                signal_id=f"es_{log.log_id}_0",
                emotion=log.user_state.emotion,
                intensity=max(0.2, intensity_base),
                evidence=f"emotion={log.user_state.emotion}, stress={log.user_state.stress}",
            )
        ]

    def _build_intent_signals(self, directives: list[str]) -> list[UserIntentSignal]:
        out: list[UserIntentSignal] = []
        for idx, directive in enumerate(directives):
            priority = 5 if "重点" in directive or "必须" in directive else 3
            out.append(
                UserIntentSignal(
                    intent_id=f"intent_{idx}",
                    question=directive,
                    priority=priority,
                )
            )
        return out

    def _collect_core_judgements(self, log) -> list[str]:
        judgements: list[str] = []
        judgements.extend([event.reason for event in log.trade_events if event.reason])
        judgements.extend(log.scan_signals.not_trade)
        judgements.extend(log.reflection.lessons)
        return self._unique_nonempty(judgements)

    def _collect_subjective_statements(self, log) -> list[str]:
        statements: list[str] = []
        statements.extend([event.moment_emotion for event in log.trade_events if event.moment_emotion])
        statements.extend(log.scan_signals.anxiety)
        statements.extend(log.scan_signals.fomo)
        return self._unique_nonempty(statements)

    def _build_fuzzy_tendencies(self, log) -> list[str]:
        tendencies: list[str] = []
        if (log.user_state.stress or 0) >= 6:
            tendencies.append("stress_sensitive_decision_pattern")
        if log.scan_signals.fomo:
            tendencies.append("outside_opportunity_attention")
        if log.reflection.lessons:
            tendencies.append("active_rule_updating")
        return tendencies

    def _classify_type(self, statement: str) -> HypothesisType:
        for keyword in self._short_keywords:
            if keyword in statement:
                return HypothesisType.SHORT_CATALYST
        for keyword in self._mid_keywords:
            if keyword in statement:
                return HypothesisType.MID_THESIS
        for keyword in self._long_keywords:
            if keyword in statement:
                return HypothesisType.LONG_THESIS
        return HypothesisType.OTHER

    def _timeframe_hint(self, hypothesis_type: HypothesisType) -> str:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return "1D-20D"
        if hypothesis_type == HypothesisType.MID_THESIS:
            return "20D-120D"
        if hypothesis_type == HypothesisType.LONG_THESIS:
            return "120D-252D+"
        return "unknown"

    def _default_falsifiable_signals(self, hypothesis_type: HypothesisType) -> list[str]:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return ["event reaction weaker than sector baseline"]
        if hypothesis_type == HypothesisType.MID_THESIS:
            return ["sector relative strength degrades for multiple weeks"]
        if hypothesis_type == HypothesisType.LONG_THESIS:
            return ["core thesis broken by fundamentals or regime shift"]
        return ["insufficient evidence to validate"]

    def _extract_related_tickers(self, statement: str, log) -> list[str]:
        discovered = set(log.traded_tickers + log.mentioned_tickers)
        for ticker in re.findall(r"\b(?:[0-9]{3,5}\.[A-Z]{1,4}|[A-Z]{1,5}\.[A-Z]{1,4})\b", statement):
            discovered.add(ticker)
        return sorted(discovered)

    def _unique_nonempty(self, values: Iterable[str | None]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value is None:
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out


# Backward-compatible alias
PlaceholderCognitionExtractionEngine = HeuristicCognitionExtractionEngine
