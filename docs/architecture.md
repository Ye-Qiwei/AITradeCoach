# Architecture (Current Mainline)

## Daily Graph

`parse_log -> react_research -> build_report_context -> generate_report -> judge_report -> finalize_result/finalize_failure`

- **parse_log**: LLM-only structured extraction to `ParserOutput`。
- **react_research**: ReAct agent on MCP tools, judgement-oriented research, traceable tool calls。
- **build_report_context**: judgement + evidence 合并。
- **generate_report**: 产出 markdown 与 `judgement_feedback`。
- **judge_report**: rule check + LLM check。
- **finalize_result**: 产出 `TaskResult`，写长期记忆记录。

## Long-Term Evaluation Pipeline

独立于 daily graph：

- `run_due_evaluations` 扫描 `data/long_term_memory.json` 中到期记录。
- 执行最终评测，输出 final_score / commentary。
- 产生 prompt 改进 overlay 到 `config/prompts/learned_overlays.json`。

## Storage Separation

- Trace storage: `trace_logs/*.json`（流程回看）
- Long-term memory: `data/long_term_memory.json`（judgement 生命周期）

`clear_traces` 仅清理 trace，不触碰长期记忆。


## Daily Review LLM Architecture (Refactor)
- Unified model invocation via `LangChainLLMGateway` (parser/research synthesis/reporter/judge all share one path).
- ReAct research now has two explicit phases: evidence gathering and structured synthesis (`ResearchOutput`).
- Research output enforces judgement-to-evidence ID validity and complete judgement coverage.
- `dry_run=true` disables long-term memory writes and run artifact persistence.
