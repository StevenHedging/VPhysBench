# 系统架构

## 1. 领域对象与所有权

| 对象 | 所有者 | 负责内容 | 不负责内容 |
| --- | --- | --- | --- |
| DatasetSnapshot | Dataset | case、资产、物理标注、View、哈希 | prompt、tensor、模型 cache |
| TaskSpec | Benchmark | case 选择、任务族、conditioning、seed、评估协议 | 模型执行参数 |
| CanonicalTaskPlan | Benchmark | train/eval case 与 job 主表 | 模型输入 |
| Baseline Bundle | Baseline | 实现、能力、TaskBuilder、DataAdapter、trainer、predictor | 修改 Dataset |
| BaselineTaskInstance | Baseline TaskBuilder | 原生输入、operation DAG、artifact 绑定、seal | 改写 canonical plan |
| AtomicRun | Orchestrator | 执行、prediction、evaluation、状态 | 混合不同任务条件 |

依赖方向：

```text
DatasetSnapshot ─┐
                  ├─> Benchmark CanonicalTaskPlan ─┐
TaskSpec ─────────┘                                │
                                                   ├─> BaselineTaskInstance
Baseline Bundle ──> command host ─> TaskBuilder ───┘
                                                        │
                                                        ▼
                                                    AtomicRun
                                                        │
                                           predictions.jsonl
                                                        │
                                                        ▼
                                            Benchmark TaskEvaluator
```

Dataset 不认识 Baseline；Task 不携带模型输入；Baseline 不定义正式 evaluator。

## 2. Dataset 边界

正式 Dataset：

```text
datasets/physics_video/releases/3.0.0/dataset.json
```

Dataset loader 冻结 descriptor、cases、scene catalog、View A/B、asset lock 和所有引用
资产。Case 只包含模型无关事实：scene、物理量、环境、OOD、时间语义、资产和来源。
Prompt、模型尺寸、embedding、训练 metadata 和派生媒体都不能写入 Dataset。

## 3. Task 与 canonical plan

TaskSpec 使用 schema `2.0`。Planner 根据 Dataset View 产生不可由 Baseline 修改的
`CanonicalTaskPlan`。四类正式任务：

| family | conditioning | View | 训练 |
| --- | --- | --- | --- |
| `finetune_eval` | `generic` | A | 是 |
| `finetune_eval` | `physics` | A | 是 |
| `direct_eval` | `generic` | B | 否 |
| `direct_eval` | `physics` | B | 否 |

同一 family 的 generic/physics 共享 case、partition、seed 和媒体约束，只能在
Baseline 私有条件适配阶段产生差异。

## 4. Baseline Bundle v3

Baseline 集成边界是目录，而不是核心源码分支：

```text
baselines/<name>/
├── baseline.json                    # portable contract
├── baseline.local.json              # optional local deployment
├── plugin/main.py                   # command endpoint
└── implementation/config/provenance # 按模型需要
```

Registry 扫描 `baselines/*/baseline.json`，仅解析 manifest，不 import 插件。manifest
声明 `implementation.kind=command` 和 `protocol=physbench-baseline-v1`。具体模型名
永远不进入核心 Registry。

两个独立身份防止复现实验时混淆：

- bundle digest：manifest + 声明的实现文件和配置文件 SHA-256；
- deployment digest：应用本地 `runtime/model` 覆盖后的配置 SHA-256。

Bundle 内代码和配置进入 bundle digest；本机 checkpoint、Python 与外部模型路径进入
deployment digest。Bundle 复用的仓库内模型族实现、公共 profile 和 runner 逐文件进入
TaskBuilder 的 `runtime_dependency_fingerprints`。大 checkpoint 不要求每次全量扫描，但
必须在 portable manifest 中声明稳定 revision 或 digest，并在 build 前验证轻量身份文件
或 checkpoint digest。三层身份都进入 TaskInstance，避免共享代码成为指纹盲区。

当前结构：

```text
baselines/wan22_lora/                 # View A 可微调 WAN Bundle
baselines/wan22_g15_sparse_motion/    # 冻结 G15 诊断 Bundle
baselines/cosmos3_nano_i2v/           # 自包含 Cosmos 插件
src/physbench/baseline_plugins/wan22.py
                                       # 两个 WAN Bundle 唯一共享实现
src/physbench/baseline_plugins/resources/five_scene_i2v_v1/
                                       # 公平复用的五场景 prompt profiles
```

共享实现是减少同模型族复制的扩展点，不是 Registry 分支。Registry 仍只认识
`command`。

## 5. 进程隔离与 Command Host

通用宿主通过 request/response JSON 文件调用 Bundle command。支持：

```text
describe
adapt_case
build_task_instance
run_task
```

这一边界有三个目的：

1. 新 Baseline 只需新增目录，不修改或 import 核心 Registry；
2. Baseline 依赖和全局状态不会进入 Benchmark planner/evaluator 进程；
3. 输入输出可序列化、可审计，canonical plan 与 identity 可在核心侧复验。

Bundle entrypoint 必须位于自身目录，工作目录固定为 Bundle root。发现 manifest 不会
执行代码；只有显式验证、构建或运行时才启动 command。

当前 `{python}` 使用 Benchmark 的 `phybench` Python 启动协议 endpoint。模型需要不同
环境时，由 Bundle 的执行器使用 `runtime.python` 启动外部训练或推理进程。

## 6. TaskBuilder 与封印验证

核心 Planner 只回答“评哪些 case”；Baseline-owned TaskBuilder 回答“模型如何执行”。

```text
DatasetSnapshot + TaskSpec + CanonicalTaskPlan + Baseline Bundle
→ Baseline DataAdapter
→ training/inference native inputs
→ operation DAG
→ sealed BaselineTaskInstance
```

通用宿主收到 instance 后验证：

- Dataset、Task、Baseline 与 deployment identity；
- TaskBuilder/DataAdapter fingerprints；
- canonical plan 内容与 digest；
- instance canonical SHA-256。

任何一项不匹配都会在创建 run 目录之前失败。

## 7. DataAdapter 与 cache

DataAdapter 属于 Baseline，负责：

1. 空间适配；
2. 时间适配；
3. 输入范式映射；
4. 文本条件适配；
5. 物理信息注入。

每个阶段都产生结构化审计。媒体派生物写入内容寻址、不可变 cache；同一 source、
实现和配置产生同一 key。generic 分支不能观察 `case.physics`。physics 分支可读取
白名单字段，但必须记录值、单位、模板和原生注入位置。

## 8. Operation DAG

`finetune_eval`：

```text
materialize training inputs
→ fine-tune
→ bind artifact://train/model
→ materialize inference inputs
→ generate
→ evaluate
```

`direct_eval`：

```text
bind baseline://frozen_model
→ materialize inference inputs
→ generate
→ evaluate
```

操作引用符号 artifact，而不是运行前猜测的“最新文件”。执行器必须使用与实例身份相同
的 deployment。

## 9. AtomicRun

AtomicRun 冻结：

```text
DatasetSnapshot × AtomicTask × Baseline Bundle × Deployment × Seeds
```

标准输出：

```text
runs_v2/<run_id>/
├── run.json
├── frozen/
├── task_instance/
├── component_fingerprints.json
├── adaptations/
├── training/
├── jobs/
├── predictions.jsonl
├── artifacts/
└── evaluation/
```

`predictions.jsonl` 是 Baseline 与 Benchmark evaluator 的硬边界。run 同时记录 portable
bundle digest 和 deployment digest。

## 10. Scene-aware evaluation

TaskEvaluator 以冻结 `canonical_plan.jobs` 为主表，按 `scene_id` 调度
CaseEvaluator：

```text
prediction video
→ common media timeline
→ scene observation/masks
→ scene-local physical state
→ reference similarity score
→ strict Task aggregation
```

公共层负责媒体采样、SAM2、mask 质量、跟踪、几何拟合和曲线；scene 模块负责主体
prompt、状态提取、质量阈值和评分。正式分数满足：

```text
S(reference, reference) = 1
```

无同 case GT 的 OOD case 只能使用 Dataset 明确登记、物理标注逐项相同的 parent
reference；没有可信 parent 时返回 `no_trustworthy_physics_reference`，不伪造
reference，也不产生正式 case score。

## 11. 预训练污染与可比性

模型部署身份与评测资格是两件事。若冻结模型的历史训练数据与 Dataset 重叠：

1. Bundle 仍可注册并复现实验；
2. manifest 必须标为 diagnostic/non-comparable；
3. source-aware audit 必须处理“同源素材、不同 case ID”；
4. 全量结果不能进入无泄漏排行榜；
5. clean subset 必须显式列出，不能把部分结果称为正式 Task score。

G15 的审计结果是 176/214 个源 case 重叠；精确 case ID 只能发现其中 45 个，因此不能
作为污染判据。

## 12. 系统不变量

1. `datasets/` 是唯一权威数据根。
2. Dataset 不保存模型条件、模型缓存或生成结果。
3. TaskSpec 不包含 Baseline 私有执行细节。
4. Registry 不包含具体模型名或 import 分支。
5. Baseline 不能修改 canonical plan。
6. generic 适配不能观察结构化物理标注。
7. TaskInstance 执行前后使用同一 bundle 和 deployment identity。
8. 缺失 prediction 必须产生显式 case 状态。
9. 部分 coverage 不能产生正式 Task score。
10. GT-dependent 诊断不能在无 GT 时伪造。
11. 相同参考输入的 case score 必须精确为 1。
12. 外部数据或模型变换必须留下可重放 provenance。
13. 训练源重叠必须按 source identity 审计，不能只比较 case ID。
