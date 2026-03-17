# 架构文档

## 总览

AI Trading Coach 使用 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建了一个有向状态图（Directed State Graph）作为核心工作流引擎。所有节点共享一个类型化的状态对象 `OrchestratorGraphState`，每个节点读取状态、执行逻辑，然后返回需要更新的状态字段子集，由 LangGraph 自动合并。

## 流程图

```
START
  │
  ▼
parse_log
  │
  ▼
plan_research
  │
  ▼
execute_collection ◄─────────────┐
  │                              │ (retry，最多 REACT_MAX_ITERATIONS 次)
  ▼                              │
verify_information ──────────────┘
  │ (sufficient)
  ▼
build_report_context
  │
  ▼
generate_report ◄────────────────┐
  │                              │ (rewrite，最多 AGENT_MAX_REWRITE_ROUNDS 次)
  ▼                              │
judge_report ────────────────────┘
  │ (pass)          │ (fail，超出重写次数)
  ▼                 ▼
finalize_result   finalize_failure
  │                 │
  ▼                 ▼
 END               END
```

## 各节点详解

### `parse_log`

**输入**：`ReviewRunRequest`（含原始日志文本）
**输出**：`ParserOutput`（`TradeAction` 列表 + `JudgementItem` 列表）

调用 `CombinedParserAgent`，使用 `log_understanding.md` 提示词，通过 LLM 将非结构化日志文本解析为两类结构化数据：
- `TradeAction`：交易行为（买/卖/加/减/观望等）及标的资产
- `JudgementItem`：可验证的判断观点，含 `category`、`target`、`thesis`、`evaluation_window`

### `plan_research`

**输入**：`ParserOutput`
**输出**：`per_judgement_plans`（每个 Judgement 对应一份研究计划字符串列表）

针对每个 Judgement 独立调用 LLM（使用 `research_plan.md` 提示词），生成包含建议工具、搜索查询、验证标准的研究计划。可用工具列表由 `get_tool_availability()` 动态查询，仅将已配置可用的工具传给 LLM。

若本节点因 `verify_information` 触发重试而再次执行，上一轮的 `verify_suggestions`（证据不足原因）会注入到提示词上下文中，引导 LLM 改善计划。

### `execute_collection`

**输入**：`per_judgement_plans`、`ParserOutput`
**输出**：`research_output`、`evidence_packet`、`tool_calls`、`react_steps`

这是最核心的节点。对每个 Judgement 独立执行以下流程：

1. 实例化 `ToolRuntime`（副作用收集器）
2. 调用 `build_runtime_tools()` 构建 LangChain `StructuredTool` 列表
3. 调用 `create_agent()` 创建 ReAct Agent（`system_prompt` 来自 `research_agent.md`）
4. 将 Judgement 信息和研究计划封装为任务 Markdown，作为 `HumanMessage` 传入 Agent
5. Agent 通过 LLM function calling 机制自主决策工具调用顺序，直至输出最终答案
6. 解析 Agent 输出的 `# Judgement Evidence` Markdown，提取 `support_signal` 和 `evidence_quality`
7. 将工具调用产出的 `EvidenceItem` 与 LLM 文本输出合并为 `ResearchedJudgementItem`

每次工具调用的执行通过 `_record_call()` 统一处理，该函数负责：
- 验证输入参数（Pydantic）
- 执行工具（异步）
- 将 `EvidenceItem` 追加到 `ToolRuntime`
- 记录 `ToolCallTrace` 和 `ReActStep` 用于可观测性

### `verify_information`

**输入**：`research_output`、`tool_calls`
**输出**：`is_sufficient`、`continue_collection`、`verify_suggestions`

对 `research_output` 进行质量审核，触发重试的条件包括：
- 存在无证据项却标记为 `sufficient` 的 Judgement
- 存在有方向性结论（`support`/`oppose`）却无来源的 Judgement
- 所有工具调用均失败

若不满足质量要求，且 `research_retry_count < REACT_MAX_ITERATIONS`，则设置 `continue_collection=True`，图路由器将流程打回 `execute_collection`。

### `build_report_context`

**输入**：`ParserOutput`、`ResearchOutput`、`EvidencePacket`
**输出**：`report_context`（dict）

调用 `ContextBuilderV2.for_reporter()`，将前面所有节点的产出整合为报告生成所需的结构化上下文。

### `generate_report`

**输入**：`report_context`、`rewrite_instruction`（重写时才有）
**输出**：`report_draft`（Markdown 字符串）、`judgement_feedback`

调用 `ReporterAgent`，使用 `report_generation.md` 提示词。若为重写轮次，`rewrite_instruction` 会注入提示词，指导 LLM 针对性修改。

### `judge_report`

**输入**：`report_draft`、`evidence_packet`、`parse_result`、`research_output`
**输出**：`judge_verdict`、`rewrite_instruction`、`rewrite_count`

调用 `ReportJudge`，独立 LLM 实例对报告草稿进行评审，返回 `JudgeVerdict`：
- `passed=True`：报告通过，进入 `finalize_result`
- `passed=False` 且 `rewrite_count <= AGENT_MAX_REWRITE_ROUNDS`：生成 `rewrite_instruction`，路由回 `generate_report`
- `passed=False` 且超出重写次数：路由到 `finalize_failure`

### `finalize_result` / `finalize_failure`

**`finalize_result`** 负责：
- 构建 `DailyReviewReport`（含 Markdown 报告体）
- 构建 `RunTrace`（完整运行追踪记录）
- 调用 `LongTermMemoryStore.upsert_records()` 将 `LongTermJudgementRecord` 写入长期记忆（`dry_run=True` 时跳过）
- 返回 `TaskResult`（`status=SUCCESS`）

**`finalize_failure`** 仅返回 `TaskResult`（`status=FAILED`），不写入任何文件。

---

## 工具层架构

### 工具路由

Agent 只能看到**经过筛选的工具（curated tools）**，原始 MCP 工具名不直接暴露。工具调用有两条路径：

```
Agent 调用工具名
    │
    ├── yfinance_* / yahoo_japan_* 等
    │       │
    │       ├── (yfinance_*) → MCPClientManager.call_tool()
    │       │                    │
    │       │                    └── MCP SDK → yfinance 进程（stdio/sse/http）
    │       │
    │       └── (yahoo_japan_fund_history) → Python 函数（本地 HTTP 爬取）
    │
    ├── brave_search → HTTP → Brave Search API
    ├── firecrawl_extract → HTTP → Firecrawl API
    └── playwright_fetch → HTTP → 浏览器代理端点
```

### 工具可用性检测

`get_tool_availability()` 在每次运行前动态检测工具是否可用（基于 API 密钥和 MCP 连接状态），不可用的工具不会被创建为 `StructuredTool`，也不会出现在 `plan_research` 的可用工具列表中。

### MCP 客户端

`MCPClientManager` 支持三种 MCP 传输协议：
- `stdio`：启动子进程，通过标准输入输出通信（默认，用于本地 `uvx yfmcp`）
- `sse`：Server-Sent Events（HTTP 长连接）
- `http`：Streamable HTTP

工具 Schema 在首次使用时通过 `session.list_tools()` 从 MCP 服务端动态获取并缓存，之后复用。

---

## 数据模型关键类型

| 类型 | 所在模块 | 说明 |
|---|---|---|
| `ReviewRunRequest` | `domain.models` | 单次运行的入口请求，含日志文本、user_id、run_date |
| `ParserOutput` | `domain.judgement_models` | 解析结果：TradeAction 列表 + JudgementItem 列表 |
| `JudgementItem` | `domain.judgement_models` | 单条判断观点：category/target/thesis/evaluation_window |
| `ResearchedJudgementItem` | `domain.judgement_models` | 研究后的判断：在 JudgementItem 基础上附加 JudgementEvidence |
| `EvidenceItem` | `domain.models` | 单条证据：evidence_type/summary/sources/related_tickers |
| `EvidencePacket` | `domain.models` | 所有证据的汇总包，含 source_registry |
| `ToolCallTrace` | `domain.models` | 单次工具调用的完整记录：输入/输出/耗时/是否成功 |
| `ReActStep` | `domain.react_models` | 单个 ReAct 步骤：thought/action/action_input/observation |
| `RunTrace` | `domain.models` | 完整运行追踪，写入 `trace_logs/` |
| `TaskResult` | `domain.models` | 最终运行结果，含报告和追踪 |
| `LongTermJudgementRecord` | `domain.judgement_models` | 长期记忆条目，含 due_date 用于后续跟踪评估 |

---

## 可观测性

每次运行产生两类持久化记录：

**复盘报告** (`reports/{run_id}.md`)：面向用户的最终输出，Markdown 格式。

**运行追踪** (`trace_logs/{run_id}.json`)：完整的技术记录，包含：
- 所有 LLM 调用（purpose、prompt_version、耗时）
- 所有工具调用（tool_name、输入输出、耗时、成功/失败）
- 所有 ReAct 步骤（thought → action → observation 链路）
- 所有收集到的证据来源
- 重写轮次计数

追踪日志由 `observability/tracing.py` 的 `save_run_trace()` 写入，可用于调试和行为审计。
