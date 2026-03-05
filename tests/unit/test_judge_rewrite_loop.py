from __future__ import annotations

from datetime import date, datetime, timezone

from ai_trading_coach.config import Settings
from ai_trading_coach.domain.enums import ModelCallPurpose, RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.llm.provider import LLMCallRecord
from ai_trading_coach.modules.agent import (
    CombinedParserAgent,
    ContextBuilderV2,
    ExecutorEngine,
    PlannerAgent,
    ReportJudge,
    ReporterAgent,
)
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator import OrchestratorModules, PipelineOrchestrator


def _combined_parse_payload() -> dict[str, object]:
    return {
        "parse_id": "parse_r1",
        "normalized_log": {
            "log_id": "log_r1",
            "user_id": "u1",
            "log_date": "2026-03-05",
            "traded_tickers": ["AAPL.US"],
            "mentioned_tickers": ["AAPL.US"],
            "user_state": {"emotion": "calm", "stress": 2, "focus": 8},
            "market_context": {"regime": "risk_on", "key_variables": ["rates"]},
            "trade_events": [],
            "trade_narratives": [],
            "scan_signals": {"anxiety": [], "fomo": [], "not_trade": []},
            "reflection": {"facts": ["price stabilized"], "gaps": [], "lessons": ["wait for confirmation"]},
            "ai_directives": ["@AI 关注我是否过度自信"],
            "raw_text": "sample",
            "field_errors": [],
        },
        "cognition_state": {
            "cognition_id": "cog_r1",
            "log_id": "log_r1",
            "user_id": "u1",
            "as_of_date": "2026-03-05",
            "core_judgements": ["trend improving"],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "AAPL re-rating continues",
                    "hypothesis_type": "short_catalyst",
                    "related_tickers": ["AAPL.US"],
                    "timeframe_hint": "5D",
                    "evidence_for": ["momentum positive"],
                    "evidence_against": [],
                    "falsifiable_signals": ["breakdown below support"],
                    "status": "pending",
                    "confidence": 0.63,
                }
            ],
            "risk_concerns": ["macro surprise"],
            "outside_opportunities": [],
            "deliberate_no_trade_decisions": [],
            "explicit_rules": ["keep size moderate"],
            "fuzzy_tendencies": [],
            "fact_statements": ["price stabilized"],
            "subjective_statements": [],
            "behavioral_signals": [],
            "emotion_signals": [],
            "user_intent_signals": [{"intent_id": "i1", "question": "是否过度自信", "priority": 4}],
        },
    }


def _plan_payload() -> dict[str, object]:
    return {
        "plan_id": "plan_r1",
        "subtasks": [
            {
                "subtask_id": "s1",
                "objective": "Get price evidence",
                "tool_category": "market_data",
                "evidence_type": "price_path",
                "query": {"window": "5d"},
                "success_criteria": ["return series available"],
                "stop_conditions": [{"condition": "enough_data", "should_stop_when": ">=1 item"}],
                "tickers": ["AAPL.US"],
                "time_window": "5D",
            }
        ],
        "risk_uncertainties": ["macro surprise"],
        "follow_up_triggers": ["watch volume follow-through"],
    }


class QueueProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_call: LLMCallRecord | None = None

    def __init__(self, responses: dict[str, list[dict[str, object]]]) -> None:
        self.responses = responses

    def chat_json(
        self,
        schema_name: str,
        messages: list[dict[str, str]],
        timeout: float,
        prompt_version: str | None = None,
    ) -> dict[str, object]:
        del messages, timeout
        now = datetime.now(timezone.utc)
        payload = self.responses[schema_name].pop(0)
        self.last_call = LLMCallRecord(
            provider_name=self.provider_name,
            model_name=self.model_name,
            schema_name=schema_name,
            prompt_version=prompt_version,
            started_at=now,
            ended_at=now,
            latency_ms=3,
            response_size=len(str(payload)),
            token_in=11,
            token_out=17,
            error=None,
        )
        return payload

    def chat_text(self, messages: list[dict[str, str]], prompt_version: str | None = None) -> str:
        del messages, prompt_version
        return ""


def test_judge_failure_triggers_rewrite_and_passes_within_n_rounds() -> None:
    provider = QueueProvider(
        responses={
            "combined_parse_result.v1": [_combined_parse_payload()],
            "plan.v1": [_plan_payload()],
            "reporter_draft.v1": [
                {
                    "markdown": "# Daily Review Report\n## Summary\n- 初稿没有引用\n## Evidence\n- 事实缺引用\n## Key Risks\n- 风险\n## Actions\n- 动作"
                },
                {
                    "markdown": "# Daily Review Report\n## Summary\n- 结论 [source:src_srv_price_history_s1_0]\n## Evidence\n- 价格稳定 [source:src_srv_price_history_s1_0]\n## Key Risks\n- 宏观风险 [source:src_srv_price_history_s1_0]\n## Actions\n- 控制仓位 [source:src_srv_price_history_s1_0]"
                },
            ],
            "judge_verdict.v1": [
                {
                    "passed": False,
                    "reasons": ["citation missing"],
                    "rewrite_instruction": "补全引用并对齐意图",
                    "contradiction_flags": [],
                    "citation_coverage": 0.0,
                },
                {
                    "passed": True,
                    "reasons": [],
                    "rewrite_instruction": None,
                    "contradiction_flags": [],
                    "citation_coverage": 1.0,
                },
            ],
        }
    )

    settings = Settings(
        _env_file=None,
        atc_llm_provider="openai",
        openai_api_key="test-key",
        atc_agent_max_rewrite_rounds=2,
        atc_mcp_servers='[{"server_id":"srv","transport":"stdio","command":"noop"}]',
        atc_evidence_tool_map='{"price_path":"srv:price_history"}',
        atc_mcp_tool_allowlist="srv:price_history",
    )

    async def invoker(server_id: str, tool_name: str, arguments: dict[str, object]):
        del arguments
        return [{"title": f"{server_id}:{tool_name}", "ticker": "AAPL.US", "price": 101.2}]

    modules = OrchestratorModules(
        parser_agent=CombinedParserAgent(provider=provider, timeout_seconds=5),
        planner_agent=PlannerAgent(provider=provider, timeout_seconds=5),
        executor_engine=ExecutorEngine(mcp_manager=MCPClientManager(settings=settings, invoker=invoker)),
        reporter_agent=ReporterAgent(provider=provider, timeout_seconds=5),
        report_judge=ReportJudge(provider=provider, timeout_seconds=5),
        context_builder=ContextBuilderV2(settings=settings),
    )
    orchestrator = PipelineOrchestrator(modules=modules, settings=settings)
    request = ReviewRunRequest(
        run_id="r1",
        user_id="u1",
        run_date=date(2026, 3, 5),
        trigger_type=TriggerType.MANUAL,
        raw_log_text="sample",
    )
    result = orchestrator.run(request)

    assert result.status == RunStatus.SUCCESS
    assert result.trace is not None
    assert result.trace.rewrite_rounds == 2
    report_calls = [item for item in result.trace.model_calls if item.purpose == ModelCallPurpose.REPORT_GENERATION]
    assert len(report_calls) == 2
    assert "[source:src_srv_price_history_s1_0]" in (result.report.markdown_body if result.report else "")

