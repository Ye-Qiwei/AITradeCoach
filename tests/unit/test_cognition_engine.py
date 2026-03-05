from __future__ import annotations

from datetime import date

from ai_trading_coach.domain.contracts import CognitionExtractionInput, LogIntakeInput
from ai_trading_coach.modules.cognition.service import HeuristicCognitionExtractionEngine
from ai_trading_coach.modules.intake.service import MarkdownLogIntakeCanonicalizer


SAMPLE = """---
date: 2026-03-04
traded_tickers: [9660.HK]
mentioned_tickers: [4901.T]
---

## 状态
- emotion: 焦虑
- stress: 7

## 交易记录
- 9660.HK SELL 600股 7.39HKD | reason=公告后反应弱，先控制风险 | source=新闻/宏观 | trig=亏损20%

## 扫描
- fomo: 4901.T 抗跌，想继续跟踪
- not_trade: 波动加剧，今天不追高

## 复盘
- lesson: 短期事件交易必须看板块资金面

@AI 请重点看我是否受到情绪影响
"""


def test_cognition_extraction_produces_hypothesis_and_signals() -> None:
    intake = MarkdownLogIntakeCanonicalizer()
    normalized = intake.ingest(LogIntakeInput(user_id="u1", run_date=date(2026, 3, 4), raw_log_text=SAMPLE)).normalized

    engine = HeuristicCognitionExtractionEngine()
    out = engine.extract(CognitionExtractionInput(normalized_log=normalized))

    assert out.cognition_state.hypotheses
    assert out.cognition_state.outside_opportunities
    assert out.cognition_state.behavioral_signals
    assert out.cognition_state.user_intent_signals
