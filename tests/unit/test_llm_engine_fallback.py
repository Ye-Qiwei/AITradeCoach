from __future__ import annotations

from datetime import date

from ai_trading_coach.app import factory
from ai_trading_coach.config import Settings
from ai_trading_coach.domain.contracts import CognitionExtractionInput, LogIntakeInput
from ai_trading_coach.domain.enums import RunStatus, TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.modules.cognition.llm_engine import LLMCognitionExtractionEngine
from ai_trading_coach.modules.cognition.service import HeuristicCognitionExtractionEngine
from ai_trading_coach.modules.intake.service import MarkdownLogIntakeCanonicalizer
from ai_trading_coach.modules.report.service import StructuredReviewReportGenerator
from ai_trading_coach.orchestrator import PipelineOrchestrator

SAMPLE = """---
date: 2026-03-04
traded_tickers: [9660.HK]
mentioned_tickers: [4901.T]
---

## 状态
- emotion: 焦虑
- stress: 6

## 交易记录
- 9660.HK SELL 600股 7.39HKD | reason=公告后反应弱，先控制风险

## 扫描
- fomo: 4901.T 抗跌，想继续跟踪
- not_trade: 波动加剧，今天不追高

## 复盘
- fact: 板块资金明显分化
- lesson: 先看资金面再做判断

@AI 请重点看我的执行纪律
"""


class InvalidJSONProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_call = None

    def chat_json(self, schema_name: str, messages: list[dict[str, str]], timeout: float) -> dict[str, str]:
        del schema_name, messages, timeout
        return {"invalid": "payload"}

    def chat_text(self, messages: list[dict[str, str]]) -> str:
        del messages
        return ""


def test_flags_off_do_not_instantiate_provider(monkeypatch) -> None:
    called = {"count": 0}

    def _should_not_run(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("build_llm_provider should not be called when LLM flags are off")

    monkeypatch.setattr(factory, "build_llm_provider", _should_not_run)

    settings = Settings(
        atc_enable_llm_cognition=False,
        atc_enable_llm_report=False,
    )

    cognition_engine = factory.build_cognition_engine(settings)
    report_generator = factory.build_report_generator(settings)

    assert isinstance(cognition_engine, HeuristicCognitionExtractionEngine)
    assert isinstance(report_generator, StructuredReviewReportGenerator)
    assert called["count"] == 0


def test_flags_on_missing_key_falls_back_and_pipeline_succeeds(tmp_path) -> None:
    settings = Settings(
        atc_enable_llm_cognition=True,
        atc_enable_llm_report=True,
        atc_llm_provider="gemini",
        gemini_api_key="",
        atc_use_gemini=False,
    )

    modules = factory.build_orchestrator_modules(settings, memory_persist_dir=str(tmp_path / ".chroma"))
    orchestrator = PipelineOrchestrator(modules=modules)

    request = ReviewRunRequest(
        run_id="llm_missing_key",
        user_id="u1",
        run_date=date(2026, 3, 4),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=SAMPLE,
        options={"dry_run": True},
    )

    result = orchestrator.run(request)

    assert result.status == RunStatus.SUCCESS
    assert result.report is not None
    assert result.evaluation is not None


def test_invalid_json_from_llm_falls_back_to_heuristic_engine() -> None:
    intake = MarkdownLogIntakeCanonicalizer()
    normalized = intake.ingest(
        LogIntakeInput(
            user_id="u1",
            run_date=date(2026, 3, 4),
            raw_log_text=SAMPLE,
        )
    ).normalized
    data = CognitionExtractionInput(normalized_log=normalized)

    fallback_engine = HeuristicCognitionExtractionEngine()
    expected = fallback_engine.extract(data)

    engine = LLMCognitionExtractionEngine(
        provider=InvalidJSONProvider(),
        timeout_seconds=3.0,
        fallback_engine=fallback_engine,
    )
    out = engine.extract(data)

    assert out.cognition_state == expected.cognition_state
    assert out.extensions.get("llm_engine") == "fallback"
