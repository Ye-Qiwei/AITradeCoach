# 提示词文件说明

本目录存放各 Agent 节点的系统提示词，均为 Markdown 格式。每个提示词文件对应流水线中的一个特定角色。

## 文件清单

| 文件 | 对应节点 | 角色 |
|---|---|---|
| `log_understanding.md` | `parse_log` | 日志解析器 |
| `research_plan.md` | `plan_research` | 研究规划者 |
| `research_agent.md` | `execute_collection` | 证据收集 Agent |
| `report_generation.md` | `generate_report` / `judge_report` | 报告撰写者 + 评审者 |

---

## 各文件职责详解

### `log_understanding.md` — 日志解析器

**用途**：将用户撰写的自由格式交易日志解析为结构化数据。

**输出格式**：Markdown，含两个一级标题：
- `# Trade Actions`：每条交易行为（买/卖/加仓/减仓等）及标的资产
- `# Judgements`：每条待验证的判断观点，含 `category`、`target`、`thesis`、`evaluation_window`

**`evaluation_window` 允许值**：`1 day`、`1 week`、`1 month`、`3 months`、`1 year`

**`category` 允许值**：`market_view`、`asset_view`、`macro_view`、`risk_view`、`opportunity_view`、`non_action`、`reflection`

---

### `research_plan.md` — 研究规划者

**用途**：针对单条 Judgement 生成详细的研究计划。

**输入上下文**：单条 Judgement 的完整信息 + 当前可用工具列表 + （重试时）上一轮证据不足的反馈。

**输出格式**：Markdown，含以下字段：
- `thesis`：判断论点复述
- `what_to_verify`：需要验证的具体问题
- `evidence_needed`：所需证据类型
- `suggested_search_queries`：建议的搜索关键词
- `suggested_tools`：建议使用的工具（仅可从 `available_tools` 中选取）
- `done_when`：判定证据充分的标准

**注意**：输入只包含一条 Judgement，提示词明确要求不得拆分或新增 Judgement。

---

### `research_agent.md` — 证据收集 Agent

**用途**：驱动 ReAct Agent 使用工具收集证据，并输出结构化的研究结果。

**工具使用边界**（提示词中明确规定）：

| 工具 | 用途边界 |
|---|---|
| `yfinance_search` | 仅用于 Ticker 发现，**不是**价格工具 |
| `yfinance_get_ticker_info` | 基本面/简介，**不是**新闻工具 |
| `yfinance_get_ticker_news` | 仅 Ticker 相关新闻，**不是**宏观搜索 |
| `yfinance_get_price_history` | 仅历史 OHLC，**不用**于 Ticker 发现 |
| `yahoo_japan_fund_history` | 仅日本雅虎财经基金净值，**不用**于普通股票 |
| `brave_search` | 仅用于发现 URL，**不用**作最终证据 |
| `firecrawl_extract` | 从 URL 提取正文，**不用**于搜索 |
| `playwright_fetch` | 仅当 firecrawl 失败时使用 |

**强制输出规则**：
- 若只有搜索摘要、没有原始数据或文章正文，必须输出 `evidence_quality: insufficient`
- 最终答案必须是纯 Markdown，不得输出 JSON 或代码块

**输出格式**：
```markdown
# Judgement Evidence

## Judgement 1
- support_signal: support | oppose | uncertain
- evidence_quality: sufficient | insufficient | conflicting | stale | indirect
- cited_sources:
  - <来源标识>
- rationale: <基于证据的简短推理>
```

---

### `report_generation.md` — 报告撰写者 + 评审者

**用途**：双重职责——既用于 `generate_report` 节点撰写复盘报告，也用于 `judge_report` 节点对报告进行独立评审。

**撰写模式输入**：
- 解析结果（TradeAction + Judgement 列表）
- 研究证据（每条 Judgement 的 support_signal、evidence_quality、来源列表）
- 可选：重写指令（当 Judge 认为报告不合格时）

**评审模式输入**：
- 报告草稿
- 证据包（用于核实报告内容是否有据可查）

---

## 如何定制提示词

直接编辑对应的 `.md` 文件即可。系统通过 `PromptManager` 在运行时读取文件内容，每次运行使用最新版本，无需重启或重新安装。

修改时的注意事项：
1. **保持输出格式结构不变**：解析器（`text_output_parsing.py`）依赖特定的 Markdown 结构（标题层级、字段名称）。若修改输出格式，需同步修改对应的解析逻辑。
2. **工具边界规则**：`research_agent.md` 中的工具边界规则对 Agent 行为影响很大，修改前请充分测试。
3. **版本追踪**：每次运行的提示词版本（文件名）会记录在 `trace_logs/` 中，便于复现和回滚。
