# 用户日志书写规范

本文档用于说明：智能体如何理解你的日志，以及每个部分建议输入什么内容。

## 1. 总体原则

- 每篇日志聚焦“当天关键判断 + 执行依据 + 风险反思”。
- 能量化就量化，能明确就明确。
- 你不需要写很长，但要尽量让判断可被未来验证。

## 2. 推荐结构

```markdown
---
date: 2026-03-05
tags: [daily_log]
traded_tickers: [0700.HK, AAPL.US]
mentioned_tickers: [4063.T, 0331418A.JP]
---

# 交易日报

## 状态
- emotion: 焦虑
- stress: 4
- focus: 6

## 市场
- regime: 避险情绪
- key_var: 美元走强, 地缘冲突

## 交易记录
- 0700.HK BUY 100 380 HKD | reason=港股仓位回补 | source=资金面 | trig=回撤到支撑位

## 扫描
- anxiety: 某基金重仓板块连续下跌
- fomo: 4063.T 抗跌，担心错过
- not_trade: 高波动日不追高

## 复盘
- fact: 今天减仓后回撤压力下降
- gap: 预期反弹强度高于实际
- lesson: 先确认资金回流再放大仓位

@AI 请重点检查：我的减仓是纪律执行还是恐慌驱动？
```

## 3. 各部分的预期输入

### 3.1 Frontmatter
- `date`: 日志业务日期（`YYYY-MM-DD`）
- `traded_tickers`: 当天真实交易过的标的
- `mentioned_tickers`: 关注但未必交易的标的

建议：
- `traded_tickers` 只放真实成交标的
- `mentioned_tickers` 放观察名单与潜在机会

### 3.2 `## 状态`
- `emotion`: 当天主要情绪（如：焦虑、冷静、亢奋）
- `stress`: 压力等级（0-10）
- `focus`: 专注等级（0-10）

### 3.3 `## 市场`
- `regime`: 你感知到的市场状态（如：风险偏好上行、避险情绪）
- `key_var`: 驱动市场的关键变量（宏观、政策、流动性、地缘）

### 3.4 `## 交易记录`
每行建议包含：
- 标的代码
- `BUY`/`SELL`
- 数量
- 价格（可选）
- 币种（可选）
- 通过 `| key=value` 补充理由与触发条件

示例：
- `AAPL.US BUY 10 180 USD | reason=业绩兑现后回调结束`
- `4063.T SELL 50 2750 JPY | reason=短线目标达成 | trig=+15%`
- `0331418A.JP BUY 200 1250 JPY | asset_type=fund | reason=定投`

可选属性：
- `reason`
- `source`
- `trig` 或 `trigger`
- `moment_emotion`
- `risk`
- `asset_type`（`stock/etf/fund/index/option/future`）

### 3.5 `## 扫描`
- `anxiety`: 你担心的风险
- `fomo`: 你担心错过的机会
- `not_trade`: 你明确决定“不做什么”

### 3.6 `## 复盘`
- `fact`: 已发生的事实
- `gap`: 预期与现实的差距
- `lesson`: 可复用规则

### 3.7 `@AI` 指令
每行以 `@AI` 开头，表示你希望系统重点回答的问题。

作用：
- 进入 `ai_directives`
- 转为 `UserIntentSignal`
- 影响本次证据规划与报告重点

优先级规则：
- 若指令中包含“重点/必须”，系统会提升优先级

## 4. 交易内容是如何被识别的

系统主要通过两条路径识别交易相关内容：

1. `traded_tickers`（frontmatter）
- 你显式声明的交易标的，直接进入结构化字段

2. `## 交易记录`（正文）
- 解析每一行交易语句，生成 `TradeEvent`
- 同步更新 `traded_tickers`

## 5. 多市场标的支持说明

当前解析器支持常见格式：
- 港股：`0700.HK`, `9660.HK`
- 日股：`4063.T`, `4901.T`
- 美股（带市场后缀）：`AAPL.US`, `MSFT.US`
- 日本基金/基金代码：`0331418A.JP`（建议使用代码形式）

说明：
- 若是基金，建议显式加 `asset_type=fund`
- 若标的不带后缀，建议在 frontmatter 中补充 `mentioned_tickers` 或在交易行用 `asset_type` 标注

## 6. 常见失败场景与建议

- 交易行缺少 `BUY/SELL`：无法转成 `TradeEvent`
- 价格/数量格式异常：会保留原文并给出字段级 warning
- 日期不合法：回退到运行日期并告警

建议：
- 每条交易一行
- 用英文 `BUY/SELL`
- 数量与价格尽量用数字

