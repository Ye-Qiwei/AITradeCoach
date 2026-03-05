# 每日任务流转（实现级）

## 流程图
```mermaid
flowchart TD
    A[1. 读取当天日志] --> B[2. 校验并规范化日志]
    B --> C[3. 更新交易流水候选]
    C --> D[4. 抽取用户认知对象]
    D --> E[5. 召回长期记忆]
    E --> F[6. 生成证据计划]
    F --> G[7. MCP 拉取外部证据]
    G --> H[8. 动态选择分析窗口]
    H --> I[9. 认知-现实对照评估]
    I --> J[10. 生成复盘报告]
    J --> K[11. 写回长期记忆]
    K --> L[12. 生成自我改进提案]
    L --> M[13. 记录 trace 与观测指标]
```

## 编排伪代码
```python
request = ReviewRunRequest(...)

normalized = intake.ingest(request)
ledger = ledger_engine.rebuild(normalized.trade_events, historical_events)
cognition = cognition_engine.extract(normalized)
memories = memory_service.recall(query_from(cognition, normalized))
context = context_builder.build(normalized, cognition, memories)
plan = evidence_planner.plan(cognition, memories, context.task_goals)
packet = mcp_gateway.collect(plan)
window = window_selector.select(plan, cognition, ledger, packet)
evaluation = evaluator.evaluate(cognition, packet, window, memories, ledger.position_snapshot)
report = report_generator.generate(evaluation, ledger.position_snapshot, ledger.pnl_snapshot, packet, window)
memory_write = memory_service.write(high_value_records(report, evaluation, cognition))
proposal = promptops.propose(report, evaluation, run_metrics)
trace = persist_trace(...)

return TaskResult(...)
```

## 失败恢复与幂等
- 使用 `run_id + user_id + run_date` 作为幂等键。
- 对 MCP 调用支持重试/超时/降级。
- `dry_run=true` 时禁用写回（记忆写入、报告持久化）。
- 单模块失败时可返回 `partial`，保留可诊断 trace。

## 并行建议
- 可并行：
  - 不同证据源 MCP 查询
  - 同一类证据多 provider 交叉验证
- 串行：
  - intake -> cognition -> planner
  - window_selector 在 evidence packet 后执行
  - evaluator 在 window decision 后执行
