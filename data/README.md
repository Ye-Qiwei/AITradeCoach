# 用户交易日志目录

本目录用于存放用户真实的每日交易日志（长期累积数据）。

## 命名建议

`UserSummary_YYMMDD.md`，例如：`UserSummary_260317.md`

## 写作频率

- 每个交易日 1 篇（推荐收盘后写）
- 若出现重大交易决策变化，可追加补充日志

## 文件格式

每篇日志以 YAML frontmatter 开头，后接 Markdown 正文：

```markdown
---
date: 2026-03-17
tags: [daily_log]
traded_tickers: [9660.HK]
mentioned_tickers: [4901.T, 4063.T]
---

# 交易日报

## 状态
- emotion: 平静 | 焦虑 | 兴奋 | 疲惫
- stress: 1–5
- focus: 1–5

## 市场
- regime: 风险偏好 | 避险情绪 | 震荡
- key_var: 当日最重要的市场变量（如：美联储声明、非农数据）

## 交易记录
- {TICKER} {BUY|SELL|ADD|REDUCE} {数量} {价格} | reason=... | source=... | trig=... | moment_emotion=...

## 扫描
- anxiety: 令你感到不安的市场动向
- fomo: 你注意到但未操作的机会
- not_trade: 主动决定不交易的原因

## 复盘
- fact: 今日关键事实（盈亏、价格变动等）
- gap: 预期与现实的差距
- lesson: 本次交易的经验总结

@AI 请重点检查：<你希望 AI 重点分析的具体问题>
```

## 内容建议

- **`@AI` 指令**（可选）：在日志末尾写 `@AI 请重点检查：...`，告诉系统你最想得到反馈的问题，例如"这次止损是情绪驱动还是合理决策？"
- **交易记录字段说明**：
  - `reason`：交易理由
  - `source`：信息来源（宏观/技术/新闻）
  - `trig`：触发条件（如：突破压力位、止损 -15%）
  - `moment_emotion`：执行时的情绪状态

## 完整示例

参见 [`examples/logs/daily_log_sample.md`](../../examples/logs/daily_log_sample.md)。
