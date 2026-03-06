# AITradeCoach

AITradeCoach 是一个面向交易复盘场景的智能体系统：输入交易日志，自动完成结构化解析、证据检索、报告生成与质量审查，输出可追溯的每日复盘报告。

## 核心能力

- **日志理解**：把原始 markdown 交易日志解析为结构化认知状态。
- **证据规划与执行**：基于认知假设自动生成证据采集计划，并通过 MCP 工具执行。
- **报告生成**：生成包含要点、风险与后续动作的复盘报告。
- **质量审查**：对报告进行引用覆盖、意图覆盖与一致性检查，不通过时自动重写。
- **全链路追踪**：记录模型调用、工具调用、模块耗时与错误信息。

## 架构概览

主流程由五个阶段组成：

1. Parser：解析日志并抽取认知状态
2. Planner：生成 evidence plan
3. Executor：调用 MCP 工具获取证据
4. Reporter：生成 markdown 报告
5. Judge：审查并驱动重写闭环

代码入口主要位于：

- `src/ai_trading_coach/orchestrator/system_orchestrator.py`
- `src/ai_trading_coach/modules/agent/`
- `src/ai_trading_coach/modules/mcp/`
- `src/ai_trading_coach/llm/`

## 环境变量

关键配置如下：

- `LLM_PROVIDER`: `openai` 或 `gemini`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `MCP_SERVERS`
- `MCP_TOOL_ALLOWLIST`
- `EVIDENCE_TOOL_MAP`
- `AGENT_MAX_REWRITE_ROUNDS`
- `CONTEXT_BUDGET_PLANNER`
- `CONTEXT_BUDGET_REPORTER`
- `CONTEXT_BUDGET_JUDGE`

复制模板并修改：

```bash
cp .env.example .env
```

## 安装

```bash
pip install -e ".[dev]"
```

## 运行

### 手动复盘

```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_manual \
  --user-id demo_user \
  --log-file examples/logs/daily_log_sample.md \
  --run-date 2026-03-05 \
  --dry-run
```

### 批量回放

```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_replay \
  --cases-file examples/replay/replay_cases.sample.json
```

输出目录：

- 报告：`reports/<run_id>.md`
- trace：`trace_logs/<run_id>.json`

## 测试

```bash
python -m pytest -q
```
