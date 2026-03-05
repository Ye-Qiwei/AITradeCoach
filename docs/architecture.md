# 架构设计说明

## 1. 系统定位
AI Trading Cognitive Coach 是“认知教练系统”，不是自动交易系统。核心工作流是：
1. 从用户日志抽取认知对象
2. 规划证据并获取外部事实
3. 在动态时间窗口内进行认知-现实对照
4. 输出结构化复盘报告
5. 写回高价值长期记忆
6. 生成可控自我改进提案

## 2. 架构原则
- 认知优先：先解析用户判断，再找证据
- 三层分离：事实层 / 解释层 / 评价层
- 奖惩对称：识别错误，也识别超前洞察
- 动态窗口：拒绝固定 5 天评估
- 可控进化：提案 -> 离线评估 -> 晋升

## 3. 模块边界（12 核心模块）

| # | 模块 | 接口方法 | 输入契约 | 输出契约 |
|---|---|---|---|---|
| 1 | System Orchestrator | `run(request)` | `ReviewRunRequest` | `TaskResult` |
| 2 | Daily Log Intake & Canonicalizer | `ingest(data)` | `LogIntakeInput` | `LogIntakeOutput` |
| 3 | Trade Ledger & Position Engine | `rebuild(data)` | `LedgerInput` | `LedgerOutput` |
| 4 | Cognition Extraction Engine | `extract(data)` | `CognitionExtractionInput` | `CognitionExtractionOutput` |
| 5 | Long-Term Memory Service | `recall(query)`, `write(data)` | `MemoryRecallQuery`, `MemoryWriteInput` | `MemoryRecallOutput`, `MemoryWriteOutput` |
| 6 | Short-Term Context Builder | `build(data)` | `ContextBuildInput` | `ContextBuildOutput` |
| 7 | Evidence Planner | `plan(data)` | `EvidencePlanningInput` | `EvidencePlanningOutput` |
| 8 | MCP Tool Gateway | `collect(data)` | `MCPGatewayInput` | `MCPGatewayOutput` |
| 9 | Dynamic Analysis Window Selector | `select(data)` | `WindowSelectorInput` | `WindowSelectorOutput` |
| 10 | Cognition vs Reality Evaluator | `evaluate(data)` | `EvaluatorInput` | `EvaluatorOutput` |
| 11 | Review Report Generator | `generate(data)` | `ReportGeneratorInput` | `ReportGeneratorOutput` |
| 12 | PromptOps & Self-Improvement Engine | `propose(data)` | `PromptOpsInput` | `PromptOpsOutput` |

接口定义文件：`src/ai_trading_coach/interfaces/modules.py`

## 4. 分层设计
- `domain/`: 领域模型与数据契约（可测试、稳定）
- `interfaces/`: 协议层，隔离实现
- `modules/`: 各模块实现（日志、认知、证据、评估、报告、PromptOps 等）
- `orchestrator/`: 主链路编排、状态流转与错误边界
- `observability/`: trace 与指标
- `config/`: 运行配置与 prompt registry

## 5. Prompt 分层
每类 Prompt 单独模板，不允许一个超长 prompt 覆盖全流程：
1. 日志理解
2. 认知抽取
3. 证据规划
4. 窗口选择
5. 认知评估
6. 报告生成
7. 自我改进提案

模板位置：`config/prompts/*.v1.md`

## 6. 模型路由策略
- 支持模块级模型配置：每个模型任务可指定独立模型
- 未指定模块模型时，自动回退 `MODEL_DEFAULT`（再回退 `GEMINI_MODEL`）
- 当前配置入口：`.env`（如 `MODEL_COGNITION_EXTRACTION`, `MODEL_PROMPTOPS`）

## 7. 长期记忆设计
Chroma collections 规划：
- `raw_logs`
- `cognitive_cases`
- `user_profile`
- `active_theses`
- `agent_improvement_notes`

记忆记录统一模型：`MemoryRecord`。
支持字段：`status(active/archived/invalidated)`、`quality_score`、`importance`、`confidence`、`structured_payload`。

## 8. 可观测性
`RunTrace` 覆盖：
- 任务 ID、触发方式、起止时间
- 模块执行状态与耗时
- 模型调用摘要
- 工具调用摘要
- 窗口选择结果
- 证据来源
- 报告版本
- 调试上下文

## 9. 当前能力
- 每日复盘主链路可运行：日志解析 -> 认知抽取 -> 证据规划 -> MCP 取证 -> 动态窗口 -> 评估 -> 报告 -> 记忆沉淀 -> PromptOps 提案
- PromptOps 采用受控策略：提案、离线评估、A/B 准入，不在线直接改生产 prompt
- 支持回放评估与报告质量评分，便于迭代评估规则与提示词
