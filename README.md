# AI Trading Cognitive Coach

面向交易者的长期陪伴式认知教练系统。

系统目标不是给出“买/卖结论”，而是持续提升你的交易认知质量：
- 从日志中抽取可验证判断
- 用真实市场证据对照验证
- 输出结构化复盘报告
- 沉淀高价值长期记忆
- 通过受控 PromptOps 机制持续优化系统本身

## 1. 设计思路

### 1.1 认知优先，而不是行情优先
系统先分析“你在想什么、你在担心什么、你依据什么做决定”，再去找市场证据验证。

### 1.2 三层输出，避免事后诸葛亮
每次评估都分层：
- 事实层：价格、新闻、公告、情绪、宏观变量
- 解释层：这些事实如何支持/削弱你的判断
- 评价层：判断、执行、仓位、风控、情绪是否合理

### 1.3 奖惩对称
系统既指出偏差，也识别洞察。尤其对“方向对但市场尚未兑现”的情况，支持延迟判定与 follow-up 机制。

### 1.4 动态分析窗口
系统不固定看过去 5 天，而是按问题自动选择 1D/5D/20D/120D/252D/since_entry/event-centered/analog 等窗口。

### 1.5 可控自改进
PromptOps 采用受控生命周期：
1. 提案
2. 离线评估
3. A/B 准入
4. 晋升或拒绝

不允许在线直接改生产 prompt。

## 2. 系统功能模块

系统由 12 个核心模块组成：

| 模块 | 作用 | 关键输入 | 关键输出 |
|---|---|---|---|
| System Orchestrator | 串联每日主链路、异常恢复、trace 汇总 | `ReviewRunRequest` | `TaskResult` |
| Daily Log Intake | 日志接入、解析、规范化 | markdown 日志 | `DailyLogRaw`/`DailyLogNormalized` |
| Trade Ledger Engine | 自动推导交易流水、持仓、PnL | `TradeEvent` 历史+当日 | `TradeLedger`/`PositionSnapshot`/`PnLSnapshot` |
| Cognition Extraction | 抽取判断、假设、情绪、行为信号 | `DailyLogNormalized` | `CognitionState` |
| Long-Term Memory | Chroma 记忆召回与写回 | 查询条件/写入批次 | `RelevantMemorySet`/`MemoryWriteOutput` |
| Short-Term Context Builder | 构造本次执行所需上下文 | 今日输入+相关历史 | `ExecutionContext` |
| Evidence Planner | 证据规划（先规划再调用工具） | 认知对象+历史 | `EvidencePlan` |
| MCP Tool Gateway | 多数据源拉取与标准化 | `EvidencePlan` | `EvidencePacket` |
| Dynamic Window Selector | 动态选择时间窗口 | 证据计划+持仓+认知 | `WindowDecision` |
| Cognition vs Reality Evaluator | 认知-现实对照评估 | 认知+证据+窗口 | `EvaluationResult` |
| Review Report Generator | 生成用户可读复盘报告 | 评估+仓位+证据 | `DailyReviewReport` |
| PromptOps Engine | 生成改进提案与候选版本 | 运行指标+质量评分+回放 | `ImprovementProposal` 等 |

### 2.1 主任务流（简版）
对应 `docs/task_flow.md` 的完整流程，README 中简要概括如下：
1. 读取并规范化当天日志
2. 推导交易流水、持仓和 PnL
3. 抽取认知对象（判断/情绪/规则/@AI 意图）
4. 召回长期记忆并生成证据计划
5. 通过 MCP 拉取外部证据
6. 动态选择分析窗口并做认知-现实评估
7. 生成复盘报告
8. 写回高价值记忆与改进提案
9. 记录 trace（模块耗时/工具调用/窗口决策/模型调用）

## 3. 代码结构

```text
.
├── config/
│   ├── settings.example.yaml
│   └── prompts/
│       ├── *.v1.md
│       └── manifest.json
├── data/
│   └── user_logs/                  # 你的真实历史日志
├── docs/
│   ├── architecture.md
│   ├── data_contracts.md
│   ├── task_flow.md
│   ├── window_selector_design.md
│   ├── promptops_design.md
│   └── testing_strategy.md
├── examples/
│   ├── logs/
│   ├── evidence/
│   ├── outputs/
│   └── replay/
├── src/ai_trading_coach/
│   ├── app/                        # CLI 入口
│   ├── domain/                     # 枚举、模型、契约
│   ├── interfaces/                 # 模块协议
│   ├── modules/                    # 12 模块实现
│   ├── orchestrator/               # 主链路编排
│   ├── observability/              # trace/metrics
│   ├── prompts/                    # prompt registry loader
│   └── replay/                     # 回放框架
├── tests/
│   ├── unit/
│   ├── integration/
│   └── replay/
├── .env.example
└── pyproject.toml
```

## 4. 环境安装（Conda + pyproject）

### 4.1 创建并激活 Conda 环境
```bash
conda create -n AITradeCoach python=3.12 -y
conda activate AITradeCoach
```

### 4.2 安装项目依赖（基于 pyproject）
```bash
pip install -e ".[dev]"
```

### 4.3 校验安装
```bash
python -m pytest -q
```

## 5. 环境变量配置

先复制模板：
```bash
cp .env.example .env
```

重点变量说明：

| 变量 | 说明 | 示例 |
|---|---|---|
| `ATC_ENV` | 运行环境标识 | `local` |
| `ATC_DEBUG` | debug 开关 | `true` |
| `GEMINI_API_KEY` | Gemini key（可空，关闭 LLM 时不必填） | `xxx` |
| `GEMINI_MODEL` | Gemini 模型名 | `gemini-2.5-pro` |
| `ATC_USE_GEMINI` | 是否启用 Gemini 顾问增强 PromptOps | `false` |
| `ATC_MODEL_TIMEOUT_SECONDS` | 模型调用超时 | `20` |
| `MODEL_DEFAULT` | 默认模型（未指定模块模型时使用） | `gemini-2.5-pro` |
| `MODEL_LOG_UNDERSTANDING` | 日志理解模块模型覆盖 | `` |
| `MODEL_COGNITION_EXTRACTION` | 认知抽取模块模型覆盖 | `` |
| `MODEL_EVIDENCE_PLANNING` | 证据规划模块模型覆盖 | `` |
| `MODEL_WINDOW_SELECTION` | 窗口选择模块模型覆盖 | `` |
| `MODEL_COGNITION_EVALUATION` | 认知评估模块模型覆盖 | `` |
| `MODEL_REPORT_GENERATION` | 报告生成模块模型覆盖 | `` |
| `MODEL_PROMPTOPS` | PromptOps 模块模型覆盖 | `` |
| `CHROMA_PERSIST_DIR` | Chroma 存储目录 | `./.chroma` |
| `MCP_TIMEOUT_SECONDS` | MCP 调用超时 | `12` |
| `MCP_MAX_RETRIES` | MCP 重试次数 | `2` |
| `MCP_SERVERS` | MCP server 列表 | `search,price,filing,news,...` |
| `TRACE_OUTPUT_DIR` | trace 输出目录 | `./trace_logs` |
| `REPORT_OUTPUT_DIR` | 报告输出目录 | `./reports` |
| `PROMPT_REGISTRY_PATH` | prompt 目录 | `./config/prompts` |

## 6. 用户日志怎么写

### 6.1 什么时候写
建议每个交易日写 1 次，时间点任选其一：
- 收盘后（推荐）
- 当天所有交易结束后
- 遇到重大情绪波动或策略变更时补写

### 6.2 写作目标
日志不是流水账，重点是“可验证的判断”和“执行依据”：
- 我今天的核心判断是什么
- 我做/不做交易的原因是什么
- 我担心什么风险
- 我复盘得到什么规则
- 我希望 AI 下次重点检查什么

### 6.3 `@AI` 指令的作用
日志中每一行 `@AI ...` 会被系统提取为用户意图信号，用于：
- 提升本次证据规划与报告中的关注重点
- 驱动“你想优先验证的问题”进入评估流程
- 在复盘报告中形成针对性反馈，而不是泛化建议

### 6.4 交易内容解析说明（港股/日股/美股/基金）
系统主要从两个位置识别交易相关信息：
- frontmatter：`traded_tickers`
- `## 交易记录`：每一行交易语句

支持的常见标的格式（示例）：
- 港股：`0700.HK`, `9660.HK`
- 日股：`4063.T`, `4901.T`
- 美股（带市场后缀）：`AAPL.US`, `MSFT.US`
- 基金代码：`0331418A.JP`（建议配合 `asset_type=fund`）

完整书写规范、字段期望、模板和常见错误处理请看：
- `docs/user_log_format.md`

### 6.5 日志文件位置
你的历史日志建议统一放在：
- `data/user_logs/`

示例：
- `data/user_logs/UserSummary_260304.md`

## 7. 如何运行系统

### 7.1 正式运行（每日复盘）
步骤 1：准备当天日志  
作用：给智能体提供当天认知输入与交易事实。  
建议：先放在 `data/user_logs/`，再执行命令。

步骤 2：先 dry-run 验证  
作用：不写入长期记忆，先检查解析、评估、报告和 trace 是否符合预期。
```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_manual \
  --user-id demo_user \
  --log-file data/user_logs/UserSummary_260304.md \
  --run-date 2026-03-04 \
  --dry-run
```

步骤 3：正式运行  
作用：写回长期记忆（包括认知案例、活跃 thesis、PromptOps 提案）。
```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_manual \
  --user-id demo_user \
  --log-file data/user_logs/UserSummary_260304.md \
  --run-date 2026-03-04
```

步骤 4：检查输出  
作用：确认报告质量、窗口选择合理性、工具/模型调用稳定性。  
重点查看：`reports/<run_id>.md` 与 `trace_logs/<run_id>.json`。

### 7.2 回放评估（历史样本验证）
步骤 1：准备 replay case 文件  
作用：定义“预期类别/预期 follow-up”的历史样本，用于离线验证质量。

步骤 2：执行回放
```bash
PYTHONPATH=src python -m ai_trading_coach.app.run_replay \
  --cases-file examples/replay/replay_cases.sample.json
```

步骤 3：解读回放结果  
作用：观察 `average_score`、`category_hit_rate`、`ahead_of_market_recall` 等指标，作为 PromptOps 晋升依据。

## 8. 输出结果说明

每次运行后主要产物：
- 报告：`reports/<run_id>.md`
- 运行追踪：`trace_logs/<run_id>.json`
- 长期记忆：`CHROMA_PERSIST_DIR` 指向目录（默认 `./.chroma`）

`trace` 中重点关注：
- `module_spans`: 每个模块耗时和状态
- `tool_calls`: MCP 工具调用详情
- `model_calls`: 模型调用详情（启用 LLM 时）
- `window_decisions`: 时间窗口选择依据
- `debug_context`: 关键调试摘要

## 9. 日常维护建议

### 9.1 每周维护
- 抽查 3-5 份报告，看“事实-解释-评价”分层是否清晰
- 检查 `trace_logs/` 中失败率高的 MCP provider
- 检查 `agent_improvement_notes` 中提案是否堆积未评估

### 9.2 每月维护
- 归档低价值或失效的记忆记录（`archived/invalidated`）
- 更新 prompt 候选并运行回放评估
- 复核动态窗口策略是否误伤长期 thesis

### 9.3 Prompt 版本维护
- 所有生产 prompt 在 `config/prompts/`
- 活跃版本由 `config/prompts/manifest.json` 管理
- 不要直接覆盖旧版本，新增版本文件并通过回放验证后再切换 active version

## 10. 常见问题

### Q1: 为什么我日志里没有交易也能跑？
系统支持 `not_trade` 场景，会重点评估你的决策纪律与观察框架。

### Q2: 为什么有时给我“暂判”？
当证据不足或未进入有效验证窗口时，系统会给出 follow-up，而不是强行终判。

### Q3: 一定要开 Gemini 吗？
不用。默认关闭也可运行完整链路。开启后仅用于增强 PromptOps 提案文案，失败会自动回退。

## 11. 开发与测试

运行全部测试：
```bash
python -m pytest -q
```

查看可观测输出建议：
- 先看 `trace_logs/<run_id>.json`
- 再定位到对应模块的 `service.py`
- 最后结合 `docs/data_contracts.md` 检查输入输出契约

## 12. 相关文档

- [架构设计](docs/architecture.md)
- [数据契约](docs/data_contracts.md)
- [任务流转](docs/task_flow.md)
- [用户日志规范](docs/user_log_format.md)
- [动态窗口设计](docs/window_selector_design.md)
- [PromptOps 设计](docs/promptops_design.md)
- [测试策略](docs/testing_strategy.md)
