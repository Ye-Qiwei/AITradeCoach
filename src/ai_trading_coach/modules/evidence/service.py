"""Claim-driven evidence planner implementation."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ai_trading_coach.domain.contracts import EvidencePlanningInput, EvidencePlanningOutput
from ai_trading_coach.domain.enums import EvidenceType, HypothesisType
from ai_trading_coach.domain.models import EvidenceNeed, EvidencePlan, QueryTimeRange


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ClaimDrivenEvidencePlanner:
    """Build evidence needs from hypotheses, risks, and task goals."""

    _macro_keywords = {
        "利率": "interest_rate",
        "通胀": "inflation",
        "地缘": "geopolitics",
        "冲突": "geopolitics",
        "美元": "usd_index",
        "流动性": "liquidity",
    }

    def plan(self, data: EvidencePlanningInput) -> EvidencePlanningOutput:
        cognition = data.cognition_state
        needs: list[EvidenceNeed] = []
        idx = 0

        for hypothesis in self._dedupe_hypotheses(cognition.hypotheses):
            evidence_types = self._evidence_types_for_hypothesis(hypothesis.hypothesis_type)
            query_range = self._range_for_hypothesis(hypothesis.hypothesis_type)
            event_centered = hypothesis.hypothesis_type == HypothesisType.SHORT_CATALYST
            analog_history = hypothesis.hypothesis_type == HypothesisType.LONG_THESIS
            inferred = self._infer_market_dimensions(hypothesis.statement)

            need = EvidenceNeed(
                need_id=f"need_hyp_{idx}",
                hypothesis_id=hypothesis.hypothesis_id,
                claim=hypothesis.statement,
                evidence_types=evidence_types,
                tickers=hypothesis.related_tickers,
                indexes=inferred["indexes"],
                sectors=inferred["sectors"],
                macro_variables=inferred["macro_variables"],
                query_range=query_range,
                priority=self._priority_for_hypothesis(hypothesis.hypothesis_type),
                event_centered=event_centered,
                analog_history=analog_history,
                questions=self._questions_for_hypothesis(hypothesis.hypothesis_type),
            )
            needs.append(need)
            idx += 1

        for risk in cognition.risk_concerns[:3]:
            inferred = self._infer_market_dimensions(risk)
            needs.append(
                EvidenceNeed(
                    need_id=f"need_risk_{idx}",
                    claim=f"验证风险担忧: {risk}",
                    evidence_types=[EvidenceType.PRICE_PATH, EvidenceType.MACRO, EvidenceType.SENTIMENT],
                    tickers=self._pick_tickers(cognition),
                    indexes=inferred["indexes"],
                    sectors=inferred["sectors"],
                    macro_variables=inferred["macro_variables"],
                    query_range=QueryTimeRange(relative_window="20D"),
                    priority=2,
                    event_centered=False,
                    analog_history=False,
                    questions=[
                        "风险是否已经被市场定价？",
                        "风险变量是否继续恶化？",
                    ],
                )
            )
            idx += 1

        if not needs:
            needs.append(
                EvidenceNeed(
                    need_id="need_fallback_0",
                    claim="缺少明确假设，先构建基础事实面",
                    evidence_types=[EvidenceType.PRICE_PATH, EvidenceType.NEWS, EvidenceType.SENTIMENT],
                    tickers=self._pick_tickers(cognition),
                    query_range=QueryTimeRange(relative_window="20D"),
                    priority=3,
                    questions=[
                        "市场是否存在显著偏离？",
                        "当前更像趋势还是噪声？",
                    ],
                )
            )

        active_memory_claims = {
            memory.document_text.strip()
            for memory in data.active_theses
            if memory.document_text.strip()
        }
        existing_claims = {need.claim.strip() for need in needs}
        for claim in sorted(active_memory_claims - existing_claims)[:2]:
            inferred = self._infer_market_dimensions(claim)
            needs.append(
                EvidenceNeed(
                    need_id=f"need_active_thesis_{idx}",
                    claim=f"验证活跃 thesis: {claim}",
                    evidence_types=[EvidenceType.PRICE_PATH, EvidenceType.FILING, EvidenceType.NEWS],
                    tickers=self._pick_tickers(cognition),
                    indexes=inferred["indexes"],
                    sectors=inferred["sectors"],
                    macro_variables=inferred["macro_variables"],
                    query_range=QueryTimeRange(relative_window="60D"),
                    priority=3,
                    questions=["历史 thesis 当前是否强化/削弱？"],
                )
            )
            idx += 1

        needs.sort(key=lambda item: (item.priority, item.need_id))
        plan = EvidencePlan(
            plan_id=f"plan_{cognition.cognition_id}",
            user_id=cognition.user_id,
            generated_at=utc_now(),
            needs=needs,
            priority_order=[need.need_id for need in needs],
            requires_event_centered_analysis=any(need.event_centered for need in needs),
            requires_analog_history=any(need.analog_history for need in needs),
            planner_notes=[
                f"hypotheses={len(cognition.hypotheses)}",
                f"risk_concerns={len(cognition.risk_concerns)}",
                f"active_thesis_memories={len(data.active_theses)}",
                f"deduped_needs={len(needs)}",
            ],
        )
        return EvidencePlanningOutput(plan=plan)

    def _dedupe_hypotheses(self, hypotheses):
        seen: set[str] = set()
        deduped = []
        for hypothesis in hypotheses:
            key = re.sub(r"\s+", " ", hypothesis.statement.strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(hypothesis)
        return deduped

    def _infer_market_dimensions(self, text: str) -> dict[str, list[str]]:
        macro_vars: set[str] = set()
        sectors: set[str] = set()
        indexes: set[str] = set()

        lower_text = text.lower()
        for keyword, macro in self._macro_keywords.items():
            if keyword in text:
                macro_vars.add(macro)

        if any(word in text for word in ["半导体", "芯片", "AI"]):
            sectors.add("semiconductor")
        if any(word in text for word in ["科技", "互联网", "软件"]):
            sectors.add("technology")
        if any(word in text for word in ["恒生", "港股", "hong kong"]):
            indexes.add("HSI")
        if any(word in lower_text for word in ["nasdaq", "sp500", "s&p"]):
            indexes.add("SPX")

        return {
            "indexes": sorted(indexes),
            "sectors": sorted(sectors),
            "macro_variables": sorted(macro_vars),
        }

    def _evidence_types_for_hypothesis(self, hypothesis_type: HypothesisType) -> list[EvidenceType]:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return [EvidenceType.PRICE_PATH, EvidenceType.NEWS, EvidenceType.FILING, EvidenceType.SENTIMENT]
        if hypothesis_type == HypothesisType.MID_THESIS:
            return [EvidenceType.PRICE_PATH, EvidenceType.SECTOR_LINKAGE, EvidenceType.FILING, EvidenceType.MACRO]
        if hypothesis_type == HypothesisType.LONG_THESIS:
            return [EvidenceType.PRICE_PATH, EvidenceType.FILING, EvidenceType.MACRO, EvidenceType.ANALOG_HISTORY]
        return [EvidenceType.PRICE_PATH, EvidenceType.NEWS]

    def _range_for_hypothesis(self, hypothesis_type: HypothesisType) -> QueryTimeRange:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return QueryTimeRange(relative_window="20D")
        if hypothesis_type == HypothesisType.MID_THESIS:
            return QueryTimeRange(relative_window="120D")
        if hypothesis_type == HypothesisType.LONG_THESIS:
            return QueryTimeRange(relative_window="252D")
        return QueryTimeRange(relative_window="60D")

    def _priority_for_hypothesis(self, hypothesis_type: HypothesisType) -> int:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return 1
        if hypothesis_type in {HypothesisType.MID_THESIS, HypothesisType.LONG_THESIS}:
            return 2
        return 3

    def _questions_for_hypothesis(self, hypothesis_type: HypothesisType) -> list[str]:
        if hypothesis_type == HypothesisType.SHORT_CATALYST:
            return [
                "事件后 1D/5D 反应是否符合预期？",
                "是否出现超预期反应或明显钝化？",
            ]
        if hypothesis_type == HypothesisType.MID_THESIS:
            return [
                "20D/60D 趋势是否持续支持 thesis？",
                "行业相对强弱是否强化原判断？",
            ]
        if hypothesis_type == HypothesisType.LONG_THESIS:
            return [
                "长期结构是否仍成立？",
                "当前是否仅为阶段性回撤而非 thesis 失效？",
            ]
        return ["该判断是否有事实证据支持？"]

    def _pick_tickers(self, cognition) -> list[str]:
        tickers: set[str] = set()
        for hypothesis in cognition.hypotheses:
            tickers.update(hypothesis.related_tickers)
        return sorted(tickers)[:6]


# Backward-compatible alias
PlaceholderEvidencePlanner = ClaimDrivenEvidencePlanner
