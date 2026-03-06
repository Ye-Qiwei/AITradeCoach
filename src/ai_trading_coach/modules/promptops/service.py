"""PromptOps implementation with controlled proposal -> offline eval -> AB gate."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.contracts import PromptOpsInput, PromptOpsOutput
from ai_trading_coach.domain.enums import ImprovementScope, ModuleName, ProposalStatus
from ai_trading_coach.domain.models import (
    ContextPolicyCandidate,
    EvaluationRubricCandidate,
    ImprovementBundle,
    ImprovementProposal,
    ModelCallTrace,
    PromptVersionCandidate,
)
from ai_trading_coach.modules.promptops.llm_advisor import GeminiPromptOpsAdvisor
from ai_trading_coach.modules.promptops.quality import ReportQualityScorer
from ai_trading_coach.modules.promptops.replay import ReplayEvaluator
from ai_trading_coach.prompts.registry import PromptRegistry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ControlledPromptOpsSelfImprovementEngine:
    """Generate offline-testable proposals without mutating production prompts."""

    def __init__(
        self,
        prompt_registry: PromptRegistry | None = None,
        quality_scorer: ReportQualityScorer | None = None,
        replay_evaluator: ReplayEvaluator | None = None,
        llm_advisor: GeminiPromptOpsAdvisor | None = None,
        enable_llm: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.prompt_registry = prompt_registry or PromptRegistry(settings.prompt_registry_path)
        self.quality_scorer = quality_scorer or ReportQualityScorer()
        self.replay_evaluator = replay_evaluator or ReplayEvaluator()
        llm_enabled = settings.use_gemini if enable_llm is None else enable_llm
        self.llm_advisor = llm_advisor
        if self.llm_advisor is None and llm_enabled and settings.gemini_api_key:
            self.llm_advisor = GeminiPromptOpsAdvisor(
                model_name=settings.model_for_module(ModuleName.PROMPTOPS),
                api_key=settings.gemini_api_key,
                timeout_seconds=settings.model_timeout_seconds,
                prompt_registry=self.prompt_registry,
            )

    def propose(self, data: PromptOpsInput) -> PromptOpsOutput:
        report_quality = self.quality_scorer.score(data.report, data.evaluation)
        replay_result = None
        if data.replay_cases:
            replay_result = self.replay_evaluator.evaluate(data.replay_cases, data.replay_predictions)

        scope = self._select_scope(data, report_quality, replay_result)
        proposal_id = self._proposal_id(data.report.report_id, scope.value)
        candidate_change = self._candidate_change(scope, report_quality, replay_result)
        expected_benefit = self._expected_benefit(scope, report_quality, replay_result)
        success_metrics = self._success_metrics(scope)
        status = self._status_from_offline_eval(report_quality.overall_score, replay_result)
        problem_statement = self._problem_statement(scope, report_quality, replay_result)

        model_traces: list[ModelCallTrace] = []
        llm_patch = self._suggest_with_llm(
            scope=scope,
            report_quality=report_quality.overall_score,
            replay_score=replay_result.average_score if replay_result else None,
        )
        if llm_patch is not None:
            patch, trace = llm_patch
            model_traces.append(trace)
            candidate_change = self._maybe_str(patch.get("candidate_change"), candidate_change)
            expected_benefit = self._maybe_str(patch.get("expected_benefit"), expected_benefit)
            problem_statement = self._maybe_str(patch.get("problem_statement"), problem_statement)
            success_metrics = self._maybe_list_of_str(patch.get("success_metrics"), success_metrics)
            llm_risk = self._maybe_int(patch.get("risk_level"))
        else:
            llm_risk = None

        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            generated_at=_utc_now(),
            scope=scope,
            problem_statement=problem_statement,
            candidate_change=candidate_change,
            expected_benefit=expected_benefit,
            risk_level=llm_risk if llm_risk is not None else self._risk_level(scope),
            offline_eval_plan=self._offline_eval_plan(scope),
            success_metrics=success_metrics,
            status=status,
        )
        proposal.extensions["offline_eval"] = {
            "report_quality_overall": report_quality.overall_score,
            "replay_average_score": replay_result.average_score if replay_result else None,
            "replay_hit_rate": replay_result.category_hit_rate if replay_result else None,
            "gate_ready_for_ab": status == ProposalStatus.AB_TESTING,
        }

        prompt_candidate = self._prompt_candidate(scope, proposal, data)
        context_policy_candidate = self._context_policy_candidate(scope)
        rubric_candidate = self._rubric_candidate(scope)

        bundle = ImprovementBundle(
            proposal=proposal,
            prompt_candidate=prompt_candidate,
            context_policy_candidate=context_policy_candidate,
            rubric_candidate=rubric_candidate,
        )
        return PromptOpsOutput(
            bundle=bundle,
            report_quality=report_quality,
            replay_result=replay_result,
            extensions={
                "model_call_traces": [trace.model_dump() for trace in model_traces],
                "llm_advisor_used": bool(model_traces),
            },
        )

    def _select_scope(
        self,
        data: PromptOpsInput,
        report_quality,
        replay_result,
    ) -> ImprovementScope:
        tool_failure_rate = float(data.run_metrics.get("tool_failure_rate", 0.0))
        completeness = float(data.run_metrics.get("evidence_completeness", 1.0))

        if tool_failure_rate >= 0.3:
            return ImprovementScope.TOOL_SEQUENCE
        if completeness < 0.55:
            return ImprovementScope.CONTEXT_POLICY
        if report_quality.actionability_score < 0.7:
            return ImprovementScope.REPORT_STYLE
        if any(bias.bias_type.value == "time_scale_bias" for bias in data.evaluation.bias_findings):
            return ImprovementScope.WINDOW_SELECTION
        if replay_result is not None and replay_result.ahead_of_market_recall is not None:
            if replay_result.ahead_of_market_recall < 0.6:
                return ImprovementScope.BIAS_RULE
        return ImprovementScope.PROMPT

    def _problem_statement(self, scope: ImprovementScope, report_quality, replay_result) -> str:
        if scope == ImprovementScope.CONTEXT_POLICY:
            return "证据完整度偏低导致结论稳定性不足。"
        if scope == ImprovementScope.TOOL_SEQUENCE:
            return "工具调用失败率偏高，影响证据覆盖和交叉验证。"
        if scope == ImprovementScope.REPORT_STYLE:
            return "报告可执行性不足，难以直接指导下次交易观察。"
        if scope == ImprovementScope.WINDOW_SELECTION:
            return "时间尺度偏差出现，窗口选择解释力需要提升。"
        if scope == ImprovementScope.BIAS_RULE:
            return "回放中 ahead_of_market 识别召回偏低，存在误伤风险。"
        if replay_result is not None and replay_result.average_score < 0.75:
            return "离线回放表现未达晋升阈值，需优化核心 prompt。"
        return (
            f"当前质量得分={report_quality.overall_score:.2f}，"
            "需持续优化提示词稳定性与一致性。"
        )

    def _candidate_change(self, scope: ImprovementScope, report_quality, replay_result) -> str:
        if scope == ImprovementScope.CONTEXT_POLICY:
            return "收紧上下文裁剪：优先 ticker+regime+emotion 交集召回，并限制低价值历史片段。"
        if scope == ImprovementScope.TOOL_SEQUENCE:
            return "对高优先级证据类型启用主备 provider 顺序与失败回退策略。"
        if scope == ImprovementScope.REPORT_STYLE:
            return "强化报告模板中的‘下一步观察清单’与‘策略动作-触发条件’表达。"
        if scope == ImprovementScope.WINDOW_SELECTION:
            return "在窗口选择提示中加入 since_entry 与 multi_window 比较优先级规则。"
        if scope == ImprovementScope.BIAS_RULE:
            return "更新评估 rubric：增加‘正确但超前’保护阈值与延迟判定条件。"
        if replay_result is not None:
            return (
                "重构核心评估 prompt 的输出约束，"
                f"目标提升离线回放命中率（当前 {replay_result.category_hit_rate:.2f}）。"
            )
        return (
            "优化核心 prompt 约束与输出 schema，"
            f"目标提升报告质量（当前 {report_quality.overall_score:.2f}）。"
        )

    def _expected_benefit(self, scope: ImprovementScope, report_quality, replay_result) -> str:
        if scope in {ImprovementScope.CONTEXT_POLICY, ImprovementScope.TOOL_SEQUENCE}:
            return "提升证据覆盖率与结论稳定性，减少因证据缺失导致的误判。"
        if scope == ImprovementScope.REPORT_STYLE:
            return "提升复盘可执行性，缩短用户从洞察到行动的路径。"
        if scope in {ImprovementScope.WINDOW_SELECTION, ImprovementScope.BIAS_RULE}:
            return "降低时间尺度误判与超前洞察误伤率。"
        if replay_result is not None:
            return f"提升离线回放平均分（当前 {replay_result.average_score:.2f}）。"
        return f"提升报告整体质量（当前 {report_quality.overall_score:.2f}）。"

    def _success_metrics(self, scope: ImprovementScope) -> list[str]:
        if scope == ImprovementScope.CONTEXT_POLICY:
            return [
                "evidence_completeness >= 0.75",
                "missing_requirements count 下降 >= 30%",
                "token 使用量不过度增长",
            ]
        if scope == ImprovementScope.TOOL_SEQUENCE:
            return [
                "tool_failure_rate <= 0.10",
                "high-priority evidence latency p95 <= 3s",
                "cross-check source count >= 2",
            ]
        if scope == ImprovementScope.REPORT_STYLE:
            return [
                "report actionability score >= 0.80",
                "next_watchlist 至少 3 条高质量信号",
                "risk alert section 覆盖率 100%",
            ]
        if scope == ImprovementScope.WINDOW_SELECTION:
            return [
                "time_scale_bias 检出后误伤率下降 >= 20%",
                "follow-up 推荐日期合理率 >= 0.75",
            ]
        if scope == ImprovementScope.BIAS_RULE:
            return [
                "ahead_of_market recall >= 0.70",
                "wrong false-positive rate <= 0.20",
            ]
        return [
            "offline replay average_score >= 0.80",
            "category_hit_rate >= 0.80",
            "unexpected_category_rate <= 0.20",
        ]

    def _status_from_offline_eval(self, quality_score: float, replay_result) -> ProposalStatus:
        if replay_result is None:
            return ProposalStatus.OFFLINE_EVALUATING
        if (
            quality_score >= 0.8
            and replay_result.average_score >= 0.8
            and replay_result.category_hit_rate >= 0.78
            and replay_result.unexpected_category_rate <= 0.2
        ):
            return ProposalStatus.AB_TESTING
        return ProposalStatus.OFFLINE_EVALUATING

    def _risk_level(self, scope: ImprovementScope) -> int:
        if scope in {ImprovementScope.PROMPT, ImprovementScope.BIAS_RULE}:
            return 4
        if scope in {ImprovementScope.CONTEXT_POLICY, ImprovementScope.WINDOW_SELECTION}:
            return 3
        return 2

    def _offline_eval_plan(self, scope: ImprovementScope) -> str:
        if scope == ImprovementScope.TOOL_SEQUENCE:
            return "离线回放 30 个样本 + 工具故障注入测试 + A/B 验证后再晋升。"
        if scope == ImprovementScope.REPORT_STYLE:
            return "离线评分器评估 30 份报告，再用双盲审阅抽检可执行性。"
        return "先运行离线回放评估，再进入 A/B 小流量验证，达标后方可晋升。"

    def _prompt_candidate(
        self,
        scope: ImprovementScope,
        proposal: ImprovementProposal,
        data: PromptOpsInput,
    ) -> PromptVersionCandidate | None:
        prompt_name = self._scope_to_prompt_name(scope)
        if prompt_name is None:
            return None

        active_version = data.active_prompt_versions.get(prompt_name)
        if not active_version:
            active_version = self.prompt_registry.get_active_version(prompt_name)
        if not active_version:
            return None

        try:
            base_text = self.prompt_registry.load_version(prompt_name, active_version)
        except FileNotFoundError:
            return None

        candidate_hash = hashlib.sha1(
            f"{proposal.proposal_id}:{prompt_name}:{active_version}".encode("utf-8")
        ).hexdigest()[:10]
        candidate_version_id = f"{prompt_name}.{active_version}.cand_{candidate_hash}"
        candidate_content = (
            f"{base_text.rstrip()}\n\n"
            "## Candidate Patch Notes\n"
            f"- proposal_id: {proposal.proposal_id}\n"
            f"- scope: {proposal.scope.value}\n"
            f"- change: {proposal.candidate_change}\n"
        )
        return PromptVersionCandidate(
            version_id=candidate_version_id,
            prompt_name=prompt_name,
            content=candidate_content,
            rationale=proposal.expected_benefit,
        )

    def _context_policy_candidate(self, scope: ImprovementScope) -> ContextPolicyCandidate | None:
        if scope not in {
            ImprovementScope.CONTEXT_POLICY,
            ImprovementScope.TOOL_SEQUENCE,
            ImprovementScope.WINDOW_SELECTION,
        }:
            return None
        return ContextPolicyCandidate(
            policy_id=f"ctx_policy_{scope.value}_{int(_utc_now().timestamp())}",
            description="Controlled candidate policy; requires offline replay and AB before promotion.",
            retrieval_rules=[
                "优先召回 ticker + regime + emotion 三重匹配记忆。",
                "上下文 budget 超限时，先保留当前窗口所需事实证据。",
                "工具失败时按 evidence priority 触发主备 provider 回退。",
            ],
        )

    def _rubric_candidate(self, scope: ImprovementScope) -> EvaluationRubricCandidate | None:
        if scope not in {ImprovementScope.BIAS_RULE, ImprovementScope.WINDOW_SELECTION, ImprovementScope.PROMPT}:
            return None
        return EvaluationRubricCandidate(
            rubric_id=f"rubric_{scope.value}_{int(_utc_now().timestamp())}",
            title="Cognition-vs-Reality Rubric Candidate",
            scoring_rules=[
                "短期未兑现但 thesis 未证伪 => 优先判定为 ahead_of_market 或 follow_up_required。",
                "证据不足时禁止输出终局结论。",
                "执行偏差与方向判断偏差必须分离评分。",
            ],
        )

    def _scope_to_prompt_name(self, scope: ImprovementScope) -> str | None:
        mapping = {
            ImprovementScope.PROMPT: "cognition_evaluation",
            ImprovementScope.REPORT_STYLE: "report_generation",
            ImprovementScope.WINDOW_SELECTION: "window_selection",
            ImprovementScope.BIAS_RULE: "cognition_evaluation",
        }
        return mapping.get(scope)

    def _proposal_id(self, report_id: str, scope: str) -> str:
        base = f"{report_id}:{scope}:{int(_utc_now().timestamp())}"
        token = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
        return f"imp_{scope}_{token}"

    def _suggest_with_llm(
        self,
        scope: ImprovementScope,
        report_quality: float,
        replay_score: float | None,
    ) -> tuple[dict[str, object], ModelCallTrace] | None:
        if self.llm_advisor is None:
            return None
        payload = {
            "scope_hint": scope.value,
            "report_quality_score": report_quality,
            "replay_average_score": replay_score,
        }
        patch, trace = self.llm_advisor.suggest(payload)
        if patch is None:
            return {"_failed": True}, trace
        return patch, trace

    def _maybe_str(self, value: object, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback

    def _maybe_list_of_str(self, value: object, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            out = [str(item).strip() for item in value if str(item).strip()]
            if out:
                return out
        return fallback

    def _maybe_int(self, value: object) -> int | None:
        try:
            ivalue = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if 1 <= ivalue <= 5:
            return ivalue
        return None


# Backward-compatible alias
PlaceholderPromptOpsSelfImprovementEngine = ControlledPromptOpsSelfImprovementEngine
