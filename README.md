# AITradeCoach (LLM-First + MCP Tool Calling)

本仓库已重构为 **LLM 必需** 的可控代理架构：

1. `CombinedParser`（LLM JSON）：一次调用把 `raw_log_text` 解析为 `DailyLogNormalized + CognitionState`
2. `Planner`（LLM JSON）：生成 `Plan.subtasks`
3. `Executor`（代码并行）：按 Plan 走 MCP 工具调用并归一化为 `EvidencePacket`
4. `Reporter`（LLM JSON）：输出带引用的 markdown
5. `Judge`（规则+LLM JSON）：检查引用覆盖/意图覆盖/冲突；失败触发 Reporter 重写（最多 N 轮）

## 1. 强约束行为

- **无 LLM 配置不允许启动主流程**
  - `MissingLLMProviderError`: `ATC_LLM_PROVIDER` 缺失或非法
  - `MissingAPIKeyError`: 对应 API key 缺失
- **所有 LLM 输出必须是严格 JSON + Pydantic 校验**
  - 失败抛 `LLMOutputValidationError`
  - 错误包含 schema 名称与校验摘要
- **MCP 调用走 allowlist**
  - 不在 `ATC_MCP_TOOL_ALLOWLIST` 的 tool 会被拦截并写入 trace
- **trace 完整**
  - LLM trace: provider/model/prompt_version/latency/error/response_size
  - Tool trace: server_id/tool/latency/success/error/payload_hash
  - 并行 subtask trace: 每个 subtask 独立记录

## 2. 目录（新增/重构）

```text
src/ai_trading_coach/
├── llm/
│   ├── provider.py
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   └── registry.py
├── modules/
│   ├── agent/
│   │   ├── combined_parser_agent.py
│   │   ├── planner_agent.py
│   │   ├── executor_engine.py
│   │   ├── reporter_agent.py
│   │   ├── report_judge.py
│   │   └── context_builder_v2.py
│   └── mcp/
│       ├── mcp_client_manager.py
│       ├── adapters.py
│       └── rss_server_example.py
├── domain/
│   ├── models.py
│   └── agent_models.py
└── orchestrator/system_orchestrator.py
```

## 3. 环境变量（关键）

复制模板：

```bash
cp .env.example .env
```

必须配置：

- `ATC_LLM_PROVIDER`: `openai` 或 `gemini`
- `ATC_LLM_MODEL`
- `ATC_LLM_TIMEOUT_SECONDS`
- `OPENAI_API_KEY` 或 `GEMINI_API_KEY`
- `ATC_MCP_SERVERS`: MCP server 定义 JSON 数组（stdio/http/sse）
- `ATC_MCP_TOOL_ALLOWLIST`
- `ATC_EVIDENCE_TOOL_MAP`
- `ATC_AGENT_MAX_REWRITE_ROUNDS`
- `ATC_CONTEXT_BUDGET_PLANNER`
- `ATC_CONTEXT_BUDGET_REPORTER`
- `ATC_CONTEXT_BUDGET_JUDGE`

## 4. MCP server（stdio）示例

`ATC_MCP_SERVERS` 示例：

```json
[
  {"server_id":"yahoo_finance","transport":"stdio","command":"python","args":["-m","your_mcp_yahoo_server"]},
  {"server_id":"sec_edgar","transport":"stdio","command":"python","args":["-m","your_mcp_sec_server"]},
  {"server_id":"fred","transport":"stdio","command":"python","args":["-m","your_mcp_fred_server"]},
  {"server_id":"rss_search","transport":"stdio","command":"python","args":["-m","ai_trading_coach.modules.mcp.rss_server_example"]}
]
```

映射表示例（`ATC_EVIDENCE_TOOL_MAP`）：

```json
{
  "price_path": "yahoo_finance:price_history",
  "news": "rss_search:rss_search",
  "filing": "sec_edgar:list_filings",
  "macro": "fred:series_observations"
}
```

allowlist 示例（`ATC_MCP_TOOL_ALLOWLIST`）：

```text
yahoo_finance:price_history,sec_edgar:list_filings,fred:series_observations,rss_search:rss_search
```

## 5. 运行

安装：

```bash
pip install -e ".[dev,openai]"
```

手动复盘（保留原入口）：

```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_manual \
  --user-id demo_user \
  --log-file examples/logs/daily_log_sample.md \
  --run-date 2026-03-05 \
  --dry-run
```

回放（保留原入口）：

```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_replay \
  --cases-file examples/replay/replay_cases.sample.json
```

最小 demo（确保报告里有 `source_id` 引用）：

```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_manual --dry-run
```

输出：

- 报告：`reports/<run_id>.md`
- trace：`trace_logs/<run_id>.json`

## 6. 测试

```bash
python -m pytest -q
```

新增关键测试覆盖：

- 未配置 LLM key 时立即失败
- LLM 非法 JSON/schema 不通过时失败（无 fallback）
- MCP allowlist 拦截并留 trace
- Executor 并行 subtasks trace 完整
- Judge 失败触发 Reporter 重写并在 N 轮内通过

