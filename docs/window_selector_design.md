# Dynamic Analysis Window Selector 设计草图

## 目标
将“分析窗口选择”从 prompt 隐含规则中剥离为独立模块，输出可解释、可测试、可回放的 `WindowDecision`。

## 输入
- `EvidencePlan`
- `CognitionState`
- `TradeLedger` / `PositionSnapshot`
- `market_volatility_state`
- `event_timestamps`
- `holding_period_days`
- `thesis_type_hint`
- `evidence_completeness`

## 输出（必须字段）
- `selected_windows`
- `rejected_windows`
- `selection_reason`
- `judgement_type`
- `follow_up_needed`
- `recommended_next_review_date`
- `confidence`

## 可判定问题模板
1. 这是短期反应问题还是 thesis 验证问题？
2. 这是市场定价问题还是用户执行问题？
3. 当前是否进入可判定阶段？
4. 需要事件前后对照还是长期相对表现？
5. 是否需要历史相似片段类比？

## 策略规则（v1 基线）

### 情况 A：短线事件交易
触发：`HypothesisType.SHORT_CATALYST`
优先窗口：`event_centered_window`, `1D`, `5D`, `20D`

### 情况 B：中期 thesis
触发：`HypothesisType.MID_THESIS`
优先窗口：`20D`, `60D`, `120D`

### 情况 C：长期认知
触发：`HypothesisType.LONG_THESIS`
优先窗口：`since_entry`, `120D`, `252D`

### 情况 D：认知可能超前
触发：短期未兑现 + 证据不充分 + 无明显证伪
策略：`judgement_type=preliminary`，`follow_up_needed=true`

### 情况 E：历史相似片段
触发：`EvidencePlan.requires_analog_history=true`
优先窗口：`analog_historical_segment` + `multi_window_comparison`

## 置信度建议
- 基础分：0.35
- + 证据完整度（0~0.5）
- + 时间尺度匹配度（0~0.15）
- 低于阈值（默认 0.55）时触发 follow-up

## 代码位置
- 实现：`src/ai_trading_coach/modules/window/rule_based_selector.py`
- 接口：`src/ai_trading_coach/interfaces/modules.py`
- 契约：`src/ai_trading_coach/domain/contracts.py`
