"""Rule-based dynamic analysis window selector (explainable baseline)."""

from __future__ import annotations

from datetime import date, timedelta

from ai_trading_coach.domain.contracts import WindowSelectorInput, WindowSelectorOutput
from ai_trading_coach.domain.enums import AnalysisWindowType, HypothesisType, JudgementType
from ai_trading_coach.domain.models import WindowChoice, WindowDecision, WindowRejected


class RuleBasedWindowSelector:
    """Deterministic selector with explicit rationale and follow-up policy."""

    def select(self, data: WindowSelectorInput) -> WindowSelectorOutput:
        run_date = data.trade_ledger.as_of_date
        selected: list[WindowChoice] = []
        selection_reason: list[str] = []

        hypothesis_types = {hyp.hypothesis_type for hyp in data.cognition_state.hypotheses}
        if data.thesis_type_hint:
            hint_lower = data.thesis_type_hint.lower()
            if "short" in hint_lower:
                hypothesis_types.add(HypothesisType.SHORT_CATALYST)
            if "mid" in hint_lower:
                hypothesis_types.add(HypothesisType.MID_THESIS)
            if "long" in hint_lower:
                hypothesis_types.add(HypothesisType.LONG_THESIS)

        selected.extend(self._select_by_hypothesis(hypothesis_types, run_date))

        if self._requires_event_window(data):
            selected.append(
                self._window(
                    AnalysisWindowType.EVENT_CENTERED,
                    run_date,
                    3,
                    "Event-centered check for catalyst reaction and expectation gap",
                    ["Need event-before/after comparison"],
                    0.75,
                )
            )
            selection_reason.append("Event-centered analysis required by evidence plan or event timestamps.")

        if data.plan.requires_analog_history:
            selected.append(
                self._window(
                    AnalysisWindowType.ANALOG_SEGMENT,
                    run_date,
                    252,
                    "Analog historical segment requested for regime comparison",
                    ["Need historical analog validation"],
                    0.68,
                )
            )
            selected.append(
                self._window(
                    AnalysisWindowType.MULTI_WINDOW,
                    run_date,
                    120,
                    "Cross-window comparison to avoid single-horizon bias",
                    ["Need consistency across horizons"],
                    0.7,
                )
            )
            selection_reason.append("Analog historical comparison explicitly requested in evidence plan.")

        if data.holding_period_days and data.holding_period_days >= 120:
            selected.append(
                self._window(
                    AnalysisWindowType.SINCE_ENTRY,
                    run_date,
                    None,
                    "Long holding period requires since-entry attribution",
                    ["Need decision quality since position inception"],
                    0.78,
                )
            )
            selection_reason.append("Holding period is long; include since-entry perspective.")

        if not selected:
            selected.extend(
                [
                    self._window(
                        AnalysisWindowType.D5,
                        run_date,
                        5,
                        "Fallback short horizon check",
                        ["Short reaction or execution noise?"],
                        0.6,
                    ),
                    self._window(
                        AnalysisWindowType.D20,
                        run_date,
                        20,
                        "Fallback medium horizon check",
                        ["Is the judgement supported in medium horizon?"],
                        0.65,
                    ),
                ]
            )
            selection_reason.append("No clear thesis horizon found; fallback to 5D+20D baseline.")

        selected = self._dedupe(selected)

        completeness = data.evidence_completeness if data.evidence_completeness is not None else 0.5
        judgement_type = self._judgement_type(completeness)
        follow_up_needed = judgement_type != JudgementType.FINAL
        next_review_date = self._next_review_date(run_date, judgement_type)

        if judgement_type == JudgementType.FOLLOW_UP_REQUIRED:
            selection_reason.append("Evidence insufficient for final judgement; follow-up required.")
        elif judgement_type == JudgementType.PRELIMINARY:
            selection_reason.append("Evidence partially sufficient; preliminary judgement with scheduled follow-up.")
        else:
            selection_reason.append("Evidence coverage is acceptable for final judgement in selected windows.")

        rejected = self._build_rejected(selected, data)

        decision = WindowDecision(
            decision_id=f"wd_{data.trade_ledger.ledger_id}",
            selected_windows=selected,
            rejected_windows=rejected,
            selection_reason=selection_reason,
            judgement_type=judgement_type,
            follow_up_needed=follow_up_needed,
            recommended_next_review_date=next_review_date,
            confidence=self._confidence(completeness, len(selected), judgement_type),
        )
        return WindowSelectorOutput(decision=decision)

    def _select_by_hypothesis(
        self,
        hypothesis_types: set[HypothesisType],
        run_date: date,
    ) -> list[WindowChoice]:
        selected: list[WindowChoice] = []

        if HypothesisType.SHORT_CATALYST in hypothesis_types:
            selected.extend(
                [
                    self._window(
                        AnalysisWindowType.D1,
                        run_date,
                        1,
                        "Short catalyst immediate reaction",
                        ["Is market response immediate and directional?"],
                        0.74,
                    ),
                    self._window(
                        AnalysisWindowType.D5,
                        run_date,
                        5,
                        "Short catalyst confirmation horizon",
                        ["Did day-1 reaction persist?"],
                        0.72,
                    ),
                    self._window(
                        AnalysisWindowType.D20,
                        run_date,
                        20,
                        "Avoid overfitting to one-day move",
                        ["Any reversal vs short-term narrative?"],
                        0.7,
                    ),
                ]
            )

        if HypothesisType.MID_THESIS in hypothesis_types:
            selected.extend(
                [
                    self._window(
                        AnalysisWindowType.D20,
                        run_date,
                        20,
                        "Mid-thesis initial validation window",
                        ["Is thesis entering verification stage?"],
                        0.72,
                    ),
                    self._window(
                        AnalysisWindowType.D60,
                        run_date,
                        60,
                        "Mid-thesis persistence check",
                        ["Is trend persistence supporting thesis?"],
                        0.73,
                    ),
                    self._window(
                        AnalysisWindowType.D120,
                        run_date,
                        120,
                        "Cross-quarter confirmation for medium horizon",
                        ["Does quarterly structure support thesis?"],
                        0.74,
                    ),
                ]
            )

        if HypothesisType.LONG_THESIS in hypothesis_types:
            selected.extend(
                [
                    self._window(
                        AnalysisWindowType.D120,
                        run_date,
                        120,
                        "Long-thesis medium anchor window",
                        ["Is long thesis degraded or intact?"],
                        0.72,
                    ),
                    self._window(
                        AnalysisWindowType.D252,
                        run_date,
                        252,
                        "Long-thesis annual structure check",
                        ["Does yearly structure support strategic view?"],
                        0.75,
                    ),
                    self._window(
                        AnalysisWindowType.SINCE_ENTRY,
                        run_date,
                        None,
                        "Long-thesis should be evaluated since entry",
                        ["Was position management coherent since entry?"],
                        0.78,
                    ),
                ]
            )

        return selected

    def _requires_event_window(self, data: WindowSelectorInput) -> bool:
        hypothesis_types = {hyp.hypothesis_type for hyp in data.cognition_state.hypotheses}
        if HypothesisType.SHORT_CATALYST in hypothesis_types:
            return True
        if data.event_timestamps:
            return True
        return any(need.event_centered for need in data.plan.needs)

    def _window(
        self,
        window_type: AnalysisWindowType,
        run_date: date,
        lookback_days: int | None,
        reason: str,
        target_questions: list[str],
        confidence: float,
    ) -> WindowChoice:
        if lookback_days is None:
            return WindowChoice(
                window_type=window_type,
                start_date=None,
                end_date=run_date,
                reason=reason,
                target_questions=target_questions,
                confidence=confidence,
            )

        return WindowChoice(
            window_type=window_type,
            start_date=run_date - timedelta(days=lookback_days),
            end_date=run_date,
            reason=reason,
            target_questions=target_questions,
            confidence=confidence,
        )

    def _dedupe(self, windows: list[WindowChoice]) -> list[WindowChoice]:
        seen: set[AnalysisWindowType] = set()
        deduped: list[WindowChoice] = []
        for window in windows:
            if window.window_type in seen:
                continue
            seen.add(window.window_type)
            deduped.append(window)
        return deduped

    def _judgement_type(self, completeness: float) -> JudgementType:
        if completeness < 0.45:
            return JudgementType.FOLLOW_UP_REQUIRED
        if completeness < 0.6:
            return JudgementType.PRELIMINARY
        return JudgementType.FINAL

    def _next_review_date(self, run_date: date, judgement_type: JudgementType) -> date:
        if judgement_type == JudgementType.FOLLOW_UP_REQUIRED:
            return run_date + timedelta(days=2)
        if judgement_type == JudgementType.PRELIMINARY:
            return run_date + timedelta(days=4)
        return run_date + timedelta(days=7)

    def _confidence(
        self,
        completeness: float,
        window_count: int,
        judgement_type: JudgementType,
    ) -> float:
        base = 0.35 + min(0.45, completeness)
        coverage_bonus = min(0.12, window_count * 0.02)
        if judgement_type == JudgementType.FOLLOW_UP_REQUIRED:
            penalty = 0.1
        elif judgement_type == JudgementType.PRELIMINARY:
            penalty = 0.05
        else:
            penalty = 0.0
        return max(0.2, min(0.95, base + coverage_bonus - penalty))

    def _build_rejected(
        self,
        selected: list[WindowChoice],
        data: WindowSelectorInput,
    ) -> list[WindowRejected]:
        selected_types = {window.window_type for window in selected}
        rejected: list[WindowRejected] = []
        for window_type in AnalysisWindowType:
            if window_type in selected_types:
                continue
            reason = "Lower incremental value for current thesis/evidence setting."
            if window_type == AnalysisWindowType.SINCE_ENTRY and not data.holding_period_days:
                reason = "Since-entry window skipped because holding period is unknown or short."
            if window_type == AnalysisWindowType.ANALOG_SEGMENT and not data.plan.requires_analog_history:
                reason = "Analog history not requested by current evidence plan."
            rejected.append(WindowRejected(window_type=window_type, reason=reason))
        return rejected
