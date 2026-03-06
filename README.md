# AITradeCoach

一个围绕 **LangGraph 主链** 的交易日志智能体系统：

`run_manual -> build_pipeline_orchestrator -> PipelineOrchestrator.run -> build_review_graph -> parse_log -> react_research -> build_report_context -> generate_report -> judge_report -> finalize_result/finalize_failure`

## 系统目标
- 每日从用户日志抽取交易行动与显式/隐含判断。
- 基于 ReAct + MCP 工具进行外部证据研究。
- 给出当天初步反馈，并为每个 judgement 记录评测周期（1 day / 1 week / 1 month / 3 months / 1 year）。
- 将 judgement 写入长期记忆，等待到期后复评。
- 到期复评后沉淀 prompt 改进 overlay（不直接改 Python 源码）。

## 每日流程
1. `parse_log`：输出严格 schema 化 judgement JSON。
2. `react_research`：按 judgement 做最小充分研究并记录工具调用。
3. `build_report_context`：组装可审阅上下文。
4. `generate_report`：输出日报 + judgement_feedback（含 evaluation_window）。
5. `judge_report`：硬校验 citation、source id、feedback 完整性。
6. `finalize_result`：写报告、trace、长期记忆。

## 长期评测流程
- 入口：`python -m ai_trading_coach.app.run_due_evaluations run --as-of 2026-03-06`
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

## 配置
- LLM：`LLM_PROVIDER`, `OPENAI_API_KEY` / `GEMINI_API_KEY`, `LLM_MODEL`
- MCP：`MCP_SERVERS`, `MCP_TOOL_ALLOWLIST`

## 运行
- 每日入口：
  - `python -m ai_trading_coach.app.run_manual run --log-file examples/logs/daily_log_sample.md --dry-run false`
- 到期评测入口：
  - `python -m ai_trading_coach.app.run_due_evaluations run --as-of 2026-03-06`
- 清理 trace：
  - `python -m ai_trading_coach.app.clear_traces run`

## Migration Note
- 已删除历史 replay 入口与 replay 相关产物。
- 已删除旧 pipeline/service 分支与未被主链调用的历史模块。
- 当前唯一保留主链：`run_manual -> ... -> finalize_result/finalize_failure`。
