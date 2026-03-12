# AITradeCoach

一个围绕 **LangGraph 主链** 的交易日志智能体系统：

`run_manual -> build_pipeline_orchestrator -> PipelineOrchestrator.run -> build_review_graph -> parse_log -> plan_research -> execute_collection -> verify_information(循环最多3次) -> build_report_context -> generate_report -> judge_report -> finalize_result/finalize_failure`

## 系统目标
- 每日从用户日志抽取交易行动与显式/隐含判断。
- 基于 ReAct + MCP 工具进行外部证据研究。
- 给出当天初步反馈，并为每个 judgement 记录评测周期（1 day / 1 week / 1 month / 3 months / 1 year）。
- 将 judgement 写入长期记忆，等待到期后复评。
- 到期复评后沉淀 prompt 改进 overlay（不直接改 Python 源码）。

## 每日流程
1. `parse_log`：输出严格 schema 化 judgement JSON。
2. `plan_research`：按原子判断生成分析框架、分析方向和信息需求清单。
3. `execute_collection`：结合 MCP 与通用网页工具执行信息收集。
4. `verify_information`：校验信息充分性，不足则回到 execute_collection，最多重试 3 次。
5. `build_report_context`：组装可审阅上下文。
6. `generate_report`：输出日报 + judgement_feedback（含 evaluation_window）。
7. `judge_report`：硬校验 citation、source id、feedback 完整性。
8. `finalize_result`：写报告、trace、长期记忆。

## 长期评测流程
- 入口：`python3 -m ai_trading_coach.app.run_due_evaluations run --as-of 2026-03-06`
- 扫描已到期 judgement。
- 基于周期内沉淀证据输出 final_score / final_commentary。
- 记录 prompt overlay（`config/prompts/learned_overlays.json`）。

## Trace 与长期记忆区别
- Trace：运行过程回放（`trace_logs/*.json`），可用 `clear_traces` 清理。
- 长期记忆：judgement 生命周期数据（`data/long_term_memory.json`），不会被清 trace 删除。

## 目录结构（精简后）
- `src/ai_trading_coach/app/`：CLI 入口（run_manual / run_due_evaluations / clear_traces）
- `src/ai_trading_coach/orchestrator/`：LangGraph 主链
- `src/ai_trading_coach/modules/agent/`：parser/reporter/judge/context
- `src/ai_trading_coach/modules/evaluation/`：长期记忆与到期评测
- `src/ai_trading_coach/prompts/`：prompt registry + overlay store
- `docs/architecture.md`：最新架构文档

## 安装
1. 创建虚拟环境：
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. 安装项目：
   - `python3 -m pip install --upgrade pip`
   - `python3 -m pip install -e ".[dev]"`
3. 复制配置：
   - `cp .env.example .env`
4. 至少配置一组 LLM 凭证：
   - `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...`
   - 或 `LLM_PROVIDER=gemini` + `GEMINI_API_KEY=...`
5. 至少配置一类 research 工具：
   - 方案 A：配置 `BRAVE_API_KEY` / `FIRECRAWL_API_KEY` / `AGENT_BROWSER_ENDPOINT`
   - 方案 B：配置 `MCP_SERVERS`、`MCP_TOOL_ALLOWLIST`、`EVIDENCE_TOOL_MAP`
6. 运行环境检查：
   - `python3 -m ai_trading_coach.app.run_manual doctor`

## 配置
- LLM：`LLM_PROVIDER`, `OPENAI_API_KEY` / `GEMINI_API_KEY`, `LLM_MODEL`
- MCP：`MCP_SERVERS`, `MCP_TOOL_ALLOWLIST`, `EVIDENCE_TOOL_MAP`
- 泛用研究工具：`BRAVE_API_KEY`, `FIRECRAWL_API_KEY`, `AGENT_BROWSER_ENDPOINT`（用于 Playwright/Agent-Browser 网页抓取桥接）
- 若 `MCP_SERVERS=[]`，系统会自动跳过未配置的 MCP 工具，只保留已启用的网页研究工具。
- 若未配置 `BRAVE_API_KEY` / `FIRECRAWL_API_KEY` / `AGENT_BROWSER_ENDPOINT`，对应网页工具不会注入到 research agent。
- `doctor` 会直接列出 `agent_tools`，用于确认 Brave / Firecrawl / 浏览器抓取和 MCP 动作是否真的暴露给 research agent。

## yfinance MCP
推荐用 [narumiruna/yfinance-mcp](https://github.com/narumiruna/yfinance-mcp) 作为 `price_path` 的 MCP 来源。项目当前已内置对以下 tool 的参数适配：
- `yfinance:yfinance_get_price_history`
- `yfinance:yfinance_get_ticker_news`

一个常见配置如下：

```dotenv
MCP_SERVERS=[{"server_id":"yfinance","transport":"stdio","command":"uvx","args":["yfmcp@latest"]}]
MCP_TOOL_ALLOWLIST=yfinance:yfinance_get_price_history,yfinance:yfinance_get_ticker_news
EVIDENCE_TOOL_MAP={"price_path":"yfinance:yfinance_get_price_history","news":"yfinance:yfinance_get_ticker_news"}
```

如果你想把股票新闻也切到 yfinance，可把 `news` 映射改成 `yfinance:yfinance_get_ticker_news`。但宏观、政策、非股票主题仍更适合 `rss_search` 或 Brave/Firecrawl。

## 运行
- 环境诊断：
  - `python3 -m ai_trading_coach.app.run_manual doctor`
- 每日入口：
  - `python3 -m ai_trading_coach.app.run_manual run --log-file examples/logs/daily_log_sample.md --dry-run false`
- 到期评测入口：
  - `python3 -m ai_trading_coach.app.run_due_evaluations run --as-of 2026-03-06`
- 清理 trace：
  - `python3 -m ai_trading_coach.app.clear_traces run`

## Migration Note
- 已删除历史 replay 入口与 replay 相关产物。
- 已删除旧 pipeline/service 分支与未被主链调用的历史模块。
- 当前唯一保留主链：`run_manual -> ... -> finalize_result/finalize_failure`。


## Curated tool architecture

- The research agent only receives curated tools (stable canonical names).
- Internal raw MCP discovery remains available for diagnostics via `MCPClientManager.diagnostics()`.
- `yahoo_japan_fund_history` now runs on direct local Python implementation by default.
- `japan_fund_mcp_server` has been removed.
