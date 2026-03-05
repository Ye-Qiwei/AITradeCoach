# PromptOps & Self-Improvement 机制

## 核心原则
禁止在线直接修改生产 prompt。所有优化必须走受控生命周期：
1. 提案（Proposal）
2. 离线评估（Offline Evaluation）
3. A/B 或回放验证
4. 晋升（Promotion）

## 输入信号
- 历史复盘质量反馈
- 用户后续行为与市场验证结果
- 失败案例 / 误判案例
- token 消耗与上下文利用率
- 工具调用成功率和覆盖度

## 输出对象
- `ImprovementProposal`
- `PromptVersionCandidate`
- `ContextPolicyCandidate`
- `EvaluationRubricCandidate`
- `ReportQualityScore`
- `ReplayEvaluationResult`

## 评估门槛建议
- 准确性不下降
- “超前洞察误伤率”下降
- 报告可执行性评分提升
- token 成本可控

## 存储
- 提案与评估日志进入 `agent_improvement_notes` collection
- 提案状态：`proposed -> offline_evaluating -> ab_testing -> promoted/rejected`

## 实现要点
- `ControlledPromptOpsSelfImprovementEngine`
  - 从运行指标 + 报告质量评分 + 回放评分生成提案
  - 仅输出候选，不直接修改生产 prompt
  - 当离线门槛满足时，状态进入 `ab_testing`（仍不自动 `promoted`）
- 可选 `GeminiPromptOpsAdvisor`（`ATC_USE_GEMINI=true`）
- 模型调用写入 `RunTrace.model_calls`
- 提案结果以 `improvement_note` 写回长期记忆
