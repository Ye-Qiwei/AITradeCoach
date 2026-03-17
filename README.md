# AI Trading Coach

AI Trading Coach 是一款自动化的交易日志分析系统。它读取用户每日撰写的 Markdown 交易日志，通过大型语言模型（LLM）和一系列研究工具，模拟专业教练的工作流程，帮助用户识别决策偏差、验证交易判断，并生成结构化的每日复盘报告。

## 核心功能

- **日志解析**：从非结构化文本中自动提取交易行为、个人反思和待验证的判断（Judgements）。
- **自动化研究**：针对每条判断，自主规划研究任务并通过多种工具收集市场数据、新闻和文章证据。
- **报告生成**：将原始日志、研究证据和 LLM 分析洞察结合，生成包含可行动反馈的每日复盘报告。
- **质量自评**：对报告草稿进行独立评审，质量不达标则触发重写，确保最终输出的可靠性。
- **长期追踪**：将每日的判断及初步反馈写入长期记忆，支持未来对判断准确性的跟踪评估。

## 整体架构

本系统基于 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建了一个图（Graph）驱动的 Agent 工作流。整个流程是一个状态机，数据在各节点间流动，并根据每步结果动态路由。

```
parse_log → plan_research → execute_collection → verify_information
                                  ↑ (retry)             ↓
                        build_report_context ←──────────┘
                                  ↓
                          generate_report → judge_report
                                               ↓        ↓
                                       finalize_result  finalize_failure
```

详细架构说明见 [`docs/architecture.md`](docs/architecture.md)。

---

## 安装指南

### 前置条件

- **Python ≥ 3.11**

### 步骤

1. **克隆仓库**：
   ```bash
   git clone <your-repo-url>
   cd AITradeCoach
   ```

2. **创建并激活 Python 虚拟环境**：
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **安装依赖**：
   ```bash
   pip install -e .
   ```
   如需开发（运行测试、类型检查等）：
   ```bash
   pip install -e .[dev]
   ```

---

## 环境配置

### 1. 创建 `.env` 文件

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

#### LLM 配置（必填）

| 变量 | 说明 |
|---|---|
| `LLM_PROVIDER` | 必填。当前支持 `openai` 和 `gemini` |
| `OPENAI_API_KEY` | 使用 OpenAI 时填写 |
| `GEMINI_API_KEY` | 使用 Gemini 时填写 |
| `LLM_MODEL` | 可选。不填则使用默认值：`gpt-4o-mini`（OpenAI）或 `gemini-2.5-pro`（Gemini） |

#### 研究工具配置（可选）

系统至少需要启用一个工具才能运行。工具按类型分为两类：

**API 密钥类工具（HTTP）**

| 变量 | 工具 | 说明 |
|---|---|---|
| `BRAVE_API_KEY` | `brave_search` | 网页搜索。获取：[brave.com/search/api](https://brave.com/search/api/) |
| `FIRECRAWL_API_KEY` | `firecrawl_extract` | 从 URL 提取文章正文。获取：[firecrawl.dev](https://firecrawl.dev/) |
| `AGENT_BROWSER_ENDPOINT` | `playwright_fetch` | JS 渲染页面抓取。需要一个运行中的浏览器代理服务，例如 [browserless/chrome](https://github.com/browserless/chrome) |

**MCP（Model Context Protocol）类工具**

| 变量 | 说明 |
|---|---|
| `MCP_SERVERS` | JSON 数组，定义如何启动和连接 MCP 服务。默认值通过 `uvx` 启动 `yfmcp`（yfinance 封装），提供股票市场数据，通常无需修改 |

> MCP（Model Context Protocol）是由 Anthropic 发布的开放标准，用于统一 AI 模型与外部工具/数据源的交互方式。

#### 输出路径配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEFAULT_USER_ID` | `demo_user` | 未指定 `--user-id` 时使用的用户 ID，影响报告和日志的文件命名 |
| `REPORT_OUTPUT_DIR` | `./reports` | 最终 Markdown 报告的保存目录 |
| `TRACE_OUTPUT_DIR` | `./trace_logs` | 运行追踪日志（JSON）的保存目录，包含每次运行的完整调用链 |

#### 运行时调参（可选）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `REACT_MAX_ITERATIONS` | `6` | 单个 Judgement 的 Agent 最大工具调用轮次 |
| `REACT_MAX_TOOL_FAILURES` | `2` | 允许的工具调用失败次数上限 |
| `REACT_REQUIRE_MIN_SOURCES` | `2` | 证据充分性所需的最少来源数 |
| `AGENT_MAX_REWRITE_ROUNDS` | `2` | 报告评审失败后的最大重写次数 |
| `MCP_TIMEOUT_SECONDS` | `60` | MCP 工具调用的超时秒数 |
| `MCP_MAX_RETRIES` | `1` | MCP 调用失败后的重试次数 |

---

## 运行方式

### 手动运行（本地开发）

```bash
python -m ai_trading_coach.app.run_manual [OPTIONS]
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--log-file TEXT` | `examples/logs/daily_log_sample.md` | 交易日志文件路径 |
| `--user-id TEXT` | `demo_user` | 用户 ID |
| `--run-date TEXT` | 今日日期（`YYYY-MM-DD`） | 本次运行的日期 |
| `--dry-run` | `False` | 开启后跳过报告写入和长期记忆写入，仅在终端输出结果，适合调试 |

**示例：**

```bash
# 使用内置示例日志快速体验
python -m ai_trading_coach.app.run_manual

# 指定日志文件和用户
python -m ai_trading_coach.app.run_manual \
  --log-file ./data/user_logs/my_log_260317.md \
  --user-id alice

# 调试模式（不写入文件）
python -m ai_trading_coach.app.run_manual \
  --log-file ./examples/logs/daily_log_sample.md \
  --dry-run
```

### 定时运行（生产/调度）

`run_daily.py` 提供了 `build_scheduled_request` 函数，用于构建定时触发的请求对象，可集成到 cron job 或任务调度系统中：

```python
from ai_trading_coach.app.run_daily import build_scheduled_request
from ai_trading_coach.app.factory import build_pipeline_orchestrator
from ai_trading_coach.config import get_settings

request = build_scheduled_request(log_path="./data/user_logs/today.md")
orchestrator = build_pipeline_orchestrator(get_settings())
result = orchestrator.run(request)
```

### 输出文件

每次成功运行会生成两类文件：

| 类型 | 路径格式 | 内容 |
|---|---|---|
| 复盘报告 | `reports/{run_id}.md` | Markdown 格式的每日交易复盘反馈 |
| 运行追踪 | `trace_logs/{run_id}.json` | 完整运行记录，含所有工具调用、LLM 调用、ReAct 步骤和证据列表 |

`run_id` 格式为 `manual_{user_id}_{date}` 或 `scheduled_{user_id}_{date}`。

---

## 核心组件详解

### LangGraph 节点

| 节点 | 功能 |
|---|---|
| `parse_log` | 使用 LLM 从原始文本中提取结构化的 `TradeAction` 列表和 `JudgementItem` 列表 |
| `plan_research` | 针对每个 Judgement 调用 LLM 生成研究计划，包含建议使用的工具和搜索查询 |
| `execute_collection` | 为每个 Judgement 创建独立的 ReAct Agent，执行工具调用并收集证据 |
| `verify_information` | 质量检查：若证据不足（无来源、仅有结论等）则触发重试，最多重试 `REACT_MAX_ITERATIONS` 次 |
| `build_report_context` | 整合解析结果、研究证据和收集的数据，构建报告生成所需的完整上下文 |
| `generate_report` | 基于上下文调用 LLM 生成包含反馈的 Markdown 复盘报告草稿 |
| `judge_report` | 独立的"评委"LLM 对报告草稿评审；不通过则生成重写指令，通过则进入终态 |
| `finalize_result` | 成功终态：保存报告文件、写入追踪日志、将 Judgements 持久化到长期记忆 |
| `finalize_failure` | 失败终态：超出重写次数后触发，记录失败状态 |

### Agent 研究工具

| 工具 | 后端 | 用途 | 可用条件 |
|---|---|---|---|
| `yfinance_search` | MCP (yfinance) | Ticker 发现，当公司名/代码不确定时首先调用 | `MCP_SERVERS` 配置有效 |
| `yfinance_get_ticker_info` | MCP (yfinance) | 获取 Ticker 基本面、公司简介 | 同上 |
| `yfinance_get_ticker_news` | MCP (yfinance) | 获取 Ticker 相关新闻标题 | 同上 |
| `yfinance_get_price_history` | MCP (yfinance) | 获取历史 OHLC 价格数据 | 同上 |
| `yfinance_get_top` | MCP (yfinance) | 获取板块涨跌幅/活跃股排行 | 同上 |
| `yahoo_japan_fund_history` | Python 本地 | 专用于日本雅虎财经基金历史净值（NAV）抓取 | 始终可用 |
| `brave_search` | HTTP (Brave API) | 互联网搜索，发现相关文章 URL | `BRAVE_API_KEY` 已填写 |
| `firecrawl_extract` | HTTP (Firecrawl) | 从指定 URL 提取文章正文（转为 Markdown） | `FIRECRAWL_API_KEY` 已填写 |
| `playwright_fetch` | HTTP (Browser) | 抓取需要 JS 渲染的动态页面 | `AGENT_BROWSER_ENDPOINT` 已填写 |

---

## 项目结构

```
AITradeCoach/
├── config/
│   └── prompts/            # 各 Agent 的系统提示词（Markdown 格式）
├── data/
│   └── user_logs/          # 用户每日交易日志存放目录
├── docs/
│   └── architecture.md     # 详细架构文档
├── examples/
│   └── logs/               # 示例日志文件
├── reports/                # 生成的复盘报告输出目录
├── trace_logs/             # 运行追踪 JSON 输出目录
└── src/
    └── ai_trading_coach/
        ├── app/            # 入口脚本（run_manual, run_daily）
        ├── config.py       # 配置管理（Settings + .env 读取）
        ├── domain/         # 数据模型（Pydantic schemas）
        ├── llm/            # LLM Gateway（LangChain 模型封装）
        ├── modules/
        │   ├── agent/      # Agent、工具定义、提示词构建
        │   ├── data_sources/ # 直连数据源（Yahoo Japan）
        │   ├── evaluation/ # 长期记忆存储
        │   ├── mcp/        # MCP 客户端管理
        │   └── report/     # 报告生成模块
        ├── observability/  # 追踪与日志
        └── orchestrator/   # LangGraph 图定义、节点、状态
```

---

## TODO
- [ ] 当前系统对用户judgement给出的feedback倾向于是泛泛而谈的comment，不够具体。且缺少正确与否的判断。针对这点需要改进。
- [ ] 追加长期记忆定期读取，重新评测对用户judgement初次给出的feedback是否正确，并以之为依据优化agent的prompt的功能。
- [ ] playwright_fetch没被测试过。
- [ ] 给agent更高级的网页浏览能力（引入agent-browser等）。
- [ ] 增加更多的tools。
- [ ] 引入skills结构。尝试看能否强化工具调用的能力。 
