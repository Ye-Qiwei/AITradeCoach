from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import PromptOpsInput
from ai_trading_coach.domain.enums import EvaluationCategory, ModelCallPurpose
from ai_trading_coach.domain.models import (
    DailyReviewReport,
    EvaluationResult,
    ExecutionAssessment,
    HypothesisAssessment,
    ModelCallTrace,
    ReportSection,
    ReplayCase,
    ReplayPrediction,
)
from ai_trading_coach.modules.promptops.service import ControlledPromptOpsSelfImprovementEngine


def _mock_report() -> DailyReviewReport:
    sections = [
        ReportSection(title="今日结论", content="- 结论"),
        ReportSection(title="关键事实", content="- 证据完整度: 0.50\n- 证据来源数: 1"),
        ReportSection(title="你看对了什么", content="- 优势"),
        ReportSection(title="你看错了什么", content="- 错误"),
        ReportSection(title="哪些判断可能是超前洞察", content="- 暂无"),
        ReportSection(title="执行/仓位/风控评估", content="- 纪律性评分: 0.70"),
        ReportSection(title="今天最值得强化的一条规则", content="- 规则"),
        ReportSection(title="今天最需要警惕的一条风险", content="- 风险"),
        ReportSection(title="下一步观察清单", content="- 信号1\n- 信号2"),
        ReportSection(title="策略修正建议 / 预警", content="- 建议\n- 证据来源交叉检查: mock_news"),
    ]
    return DailyReviewReport(
        report_id="r1",
        user_id="u1",
        report_date=date(2026, 3, 5),
        title="demo",
        sections=sections,
        key_takeaways=["k1"],
        next_watchlist=["n1", "n2"],
        strategy_adjustments=["s1"],
        risk_alerts=["risk"],
        generated_prompt_version="report_generation.v1",
        markdown_body="# report\n证据来源交叉检查: mock_news\n",
    )


def _mock_evaluation() -> EvaluationResult:
    return EvaluationResult(
        evaluation_id="e1",
        user_id="u1",
        as_of_date=date(2026, 3, 5),
        summary="demo",
        hypothesis_assessments=[
            HypothesisAssessment(
                hypothesis_id="h1",
                category=EvaluationCategory.CORRECT,
                thesis_still_valid=True,
                market_in_verification_phase=True,
                commentary="ok",
            )
        ],
        strengths=["good"],
        mistakes=["bad"],
        execution_assessment=ExecutionAssessment(),
        follow_up_signals=["watch"],
        warning_flags=["warn"],
        extensions={"layers": {"facts": ["price_evidence=1"]}},
    )


def test_promptops_engine_outputs_controlled_proposal_bundle() -> None:
    engine = ControlledPromptOpsSelfImprovementEngine()
    output = engine.propose(
        PromptOpsInput(
            evaluation=_mock_evaluation(),
            report=_mock_report(),
            run_metrics={"evidence_completeness": 0.5, "tool_failure_rate": 0.05},
            replay_cases=[
                ReplayCase(
                    case_id="c1",
                    user_id="u1",
                    run_date=date(2026, 3, 4),
                    raw_log_text="x",
                    expected_categories=[EvaluationCategory.CORRECT],
                    expected_follow_up_needed=False,
                )
            ],
            replay_predictions=[
                ReplayPrediction(
                    case_id="c1",
                    predicted_categories=[EvaluationCategory.CORRECT],
                    predicted_follow_up_needed=False,
                )
            ],
        )
    )

    assert output.bundle.proposal.status.value in {"offline_evaluating", "ab_testing"}
    assert output.bundle.proposal.scope.value in {
        "context_policy",
        "report_style",
        "prompt",
        "window_selection",
        "bias_rule",
        "tool_sequence",
    }
    assert output.report_quality is not None
    assert output.replay_result is not None


class _FakeAdvisor:
    def suggest(self, payload):
        return (
            {
                "problem_statement": "LLM refined problem",
                "candidate_change": "LLM refined change",
                "expected_benefit": "LLM refined benefit",
                "success_metrics": ["m1", "m2"],
                "risk_level": 3,
            },
            ModelCallTrace(
                call_id="m1",
                purpose=ModelCallPurpose.IMPROVEMENT_PROPOSAL,
                model_name="gemini-test",
                input_summary="in",
                output_summary="out",
                latency_ms=10,
            ),
        )


def test_promptops_engine_accepts_llm_advisor_and_exports_model_trace() -> None:
    engine = ControlledPromptOpsSelfImprovementEngine(llm_advisor=_FakeAdvisor())
    output = engine.propose(
        PromptOpsInput(
            evaluation=_mock_evaluation(),
            report=_mock_report(),
            run_metrics={"evidence_completeness": 0.9},
        )
    )
    assert output.bundle.proposal.problem_statement == "LLM refined problem"
    traces = output.extensions.get("model_call_traces")
    assert isinstance(traces, list)
    assert traces
