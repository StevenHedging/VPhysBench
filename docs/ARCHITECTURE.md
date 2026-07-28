# 系统架构

## 1. 领域边界

| 对象 | 所有者 | 负责 | 禁止 |
| --- | --- | --- | --- |
| DatasetSnapshot | Dataset | case、资产、物理/环境标注、View、哈希 | prompt、模型尺寸、生成结果 |
| TaskSpec | Benchmark | family、conditioning、选择、seed、评估协议 | 模型路径和执行参数 |
| CanonicalTaskPlan | Benchmark | train/eval case 与 job 主表 | 模型输入 |
| Baseline Bundle | Baseline | 能力、模型身份、adapter recipe、runner/driver | 修改 Dataset 或选例 |
| BaselineTaskInstance | TaskBuilder | native inputs、DAG、cache/artifact 绑定、seal | 改写 canonical plan |
| AtomicRun | Orchestrator | 快照、执行、prediction、evaluation、状态 | 混合不同任务条件 |

依赖方向：

```text
DatasetSnapshot ─┐
                  ├─> CanonicalTaskPlan ────────────────┐
TaskSpec ─────────┘                                      │
                                                         ├─> sealed
Baseline Bundle ─> Registry ─> TaskBuilder/DataAdapter ──┘   TaskInstance
                                                               │
                                                               ▼
                                                           AtomicRun
                                                               │
                                                     predictions.jsonl
                                                               │
                                                               ▼
                                                     TaskEvaluator
```

Dataset 不认识模型；Task 不携带模型输入；Baseline 不定义正式 evaluator。

## 2. Dataset 和 Task

当前正式 Dataset 是：

```text
datasets/physics_video/releases/3.0.0/dataset.json
```

loader 冻结 descriptor、cases、scene catalog、View A/B、asset lock 和引用资产。Case
只保存模型无关事实：scene、物理量、环境、OOD、时间语义、资产和来源。

TaskSpec schema 为 `2.0`。四个正式原子任务是：

| family | conditioning | View |
| --- | --- | --- |
| `finetune_eval` | `generic` | A |
| `finetune_eval` | `physics` | A |
| `direct_eval` | `generic` | B |
| `direct_eval` | `physics` | B |

同一 family 的 generic/physics 共享 case、partition、seed 和媒体资产，只能在
DataAdapter 的条件阶段产生差异。

## 3. Baseline Registry 与三档执行接口

Registry 只认识通用 kind，不认识模型名称。发现阶段仅扫描：

```text
baselines/*/baseline.json
```

当前接口：

```text
submission v4 ─┐
               ├─> adapter loader ─> DataAdapter interface ─> ManagedTaskBuilder
managed v4 ────┘                     │
                                     └─> TaskInstance ─> importer / driver

command v3 ──────> isolated command host ─> Baseline-owned TaskBuilder/executor
```

### Submission v4

用于已有预测。Benchmark 编译标准 TaskInstance，校验 submission 对 canonical jobs 的
完整覆盖和身份，再把视频复制进 run。没有 Bundle driver。

### Managed v4

是 text-conditioned T2V/I2V/V2V 与结构化控制的默认接口。Bundle 可选内置
`standard` adapter，也可提供 Bundle-local Python adapter。核心统一处理 capability、
条件隔离、input contract、计划展开、seal 和 identity；`native_inputs` 与模型执行仍
由 Baseline 持有。

### Command v3

保留给训练、复杂多进程控制或非标准模型原生协议。它实现 `describe`、`adapt_case`、
`build_task_instance` 和 `run_task` 四操作，通过 JSON request/response 进程边界运行。
核心仍复验 canonical plan、identity 和 seal。

这三档都输出同一种 TaskInstance 和 prediction contract，因此 evaluator 不需要知道
Baseline 的 kind。

扩展点遵循开闭原则与依赖倒置：新增物理注入 representation 时实现 `DataAdapter`
接口，新增模型执行方式时实现薄 `Driver`；Dataset、Task planner、canonical plan 和
Evaluator 依赖公共 contract，不依赖具体模型名称。只有出现新的任务语义或正式评分
协议时，才应修改核心层。

## 4. 身份与依赖指纹

每次运行冻结三层身份：

1. `bundle digest`：portable manifest 和 Bundle-local 实现/provenance 的 SHA-256；
2. `deployment digest`：应用本机 `model/runtime` 覆盖后的 manifest；
3. `TaskBuilder fingerprint`：adapter、runner/trainer、Bundle/deployment 以及共享
   runtime 和外部关键执行文件的指纹。

schema v4 Bundle 中所有 Python 文件自动纳入 bundle digest；JSON、shell、扩展模块等
非 Python 文件由 `fingerprint_paths` 显式登记。Bundle 外共享 prompt、solver、
生成脚本和 inference entry 由 `dependency_paths()` 纳入 TaskBuilder fingerprint。
checkpoint 路径本身不代表内容身份，必须另外声明稳定 revision、identity file 或完整
SHA-256。

`baseline.local.json` 只允许覆盖 `model` 与 `runtime`。portable capability、adapter、
implementation、ID 和版本不能由本机配置改变。

## 5. Managed compiler

`ManagedTaskBuilder` 是无副作用、确定性的公共 compiler：

```text
DatasetSnapshot + TaskSpec
→ Benchmark plan_atomic_task
→ capability/scene/conditioning validation
→ injected DataAdapter.adapt_case
→ input_contract validation
→ adaptations + inference jobs
→ training/infer/evaluate operation DAG
→ cache bindings
→ BaselineTaskInstance.seal
```

内置 `StandardDataAdapter` 提供 `standard_t2v_v1`、`standard_i2v_v1` 和
`standard_v2v_v1`。自定义 adapter 通过 `adapter.kind=python` 和
`create_adapter(bundle)` 接入；entrypoint 自动进入 Bundle digest。所有 adapter 都输出
轻量 `input_contract`：

```text
generation_mode + required text binding
+ media channels + physics channels
+ declared asset access
→ opaque native_inputs
```

`conditioning=generic|physics` 只表示 Benchmark 信息访问臂，不表示物理注入载体。
`generation_mode=t2v|i2v|v2v|hybrid` 与
`physics representation=structured_text|trajectory|mask|flow|...` 是两个独立维度。
所有 Baseline 必须有非空语言文本；physics 可以走文本、token、轨迹、mask、flow、
force field 或代理视频等模型原生通道。

compiler 不向 generic adapter 提供结构化 `case.physics`。physics adapter 只能登记
`annotated=true` 的字段，且必须实际使用至少一个参数；每个 representation 必须由
capability 声明。生成范式的媒体集合是严格的：T2V 无媒体、I2V 只有图像、V2V 只有
视频，hybrid 同时含图像和视频。

大型 control 只允许
`artifact://sha256/<digest>` / `cache://sha256/<digest>` 引用，不能把数组复制进
TaskInstance。核心校验 URI/content digest 一致以及 producer 属于当前 adapter、
materializer 或 TaskBuilder；`source_digest` 是 Baseline 声明的上游身份。当前没有
公共 artifact store，实际解析和字节 SHA-256 复验由 custom driver 负责。

### 受信任扩展边界

Bundle-local Python adapter/driver 与 command endpoint 都是受信任代码，并在 Benchmark
进程或其授权子进程中执行。当前隔离保证是“公共 compiler 不提供结构化 physics、
GT/reference/provenance，且运行前重验 contract”，不是针对恶意扩展的严格信息流
安全：case ID 和资产路径仍可能编码物理值。若未来接受盲测第三方代码，应增加
run-local opaque case handle、无语义媒体别名和进程级文件系统隔离。

## 6. Driver 与执行层

`DirectManagedDriver` 固定 direct-eval 生命周期：

```text
prepare_job
→ validate output is under run/predictions
→ write run/jobs/<job_id>.json
→ execute_job / execute_jobs
→ assemble immutable prediction identity
→ verify output file and status
```

子类不能覆盖 job、case、conditioning、partition、seed、status 和 output path。需要
常驻多 GPU worker 时可覆盖 `execute_jobs`，需要完全不同的训练生命周期时可实现
`ManagedDriver.run_task` 或使用 command。

普通 `DirectManagedDriver` 只收到 contract 声明的资产和非物理元数据；direct-eval
TaskInstance 不携带 GT/reference/source-video 或 raw physics。V2V 必须使用显式
`assets.input_video`（或同类独立 conditioning key），不得回退到
`reference_video`、`physics_reference_video` 或 `source_video`。高级
`ManagedDriver.run_task` 是更宽的生命周期扩展点；所有 Bundle Python 代码均属于上述
受信任边界。

当前正式 3.0.0 release 的 214 个 case 都没有 `assets.input_video`。因此
`managed-v2v` 目前是协议脚手架，只有 Dataset 增加独立条件视频（或 custom driver
解析经审计的 derived artifact）后才能编译正式任务。

当前复用关系：

```text
Cosmos Bundle driver
└── Cosmos payload + torchrun + checkpoint identity

G15 Bundle driver (one-line alias)
└── Wan22ManagedDriver
    └── Wan22ExecutionEngine
        ├── Wan22LoraAdapter
        └── Wan22MediaAdapter

WAN finetune command plugin
└── Wan22ExecutionEngine             # 与 G15 共用执行逻辑
```

因此 G15 与 WAN 微调 Bundle 不复制媒体和模型执行实现；Cosmos 不再复制 canonical
compiler、prompt adapter 或 prediction 组装。

## 7. BaselineTaskInstance

公共 schema 为 `2.1`：

```text
identity
├── dataset / task
├── baseline: id, version, bundle digest, deployment digest
├── task_builder / data_adapter
└── canonical_plan_digest

semantics
canonical_plan
source
adaptations
training
inference.jobs
execution_graph
cache_bindings
baseline_payload
instance_digest
```

seal 是 canonical JSON SHA-256 完整性校验，不是外部签名。TaskInstance 返回 fresh
object，执行器不能通过内存引用修改冻结实例。运行时再次对齐 canonical plan/job、
train/eval adaptation、input contract、runner/trainer/cache recipe、当前部署、
TaskBuilder、DataAdapter 和 instance digest。`source.asset_root` 仍来自受信任的本次
compiler/orchestrator；跨信任域导入实例时应重新绑定 active Dataset，而不能只重新
计算 seal。

## 8. AtomicRun 与写入边界

除模型代码、模型权重和可重建 cache 外，产物必须由 run 持有：

```text
runs_v2/<run_id>/
├── frozen/                    # Dataset、Task、Baseline 快照
├── task_instance/             # seal、jobs、adaptations、DAG
├── jobs/                      # 模型 payload/spec
├── predictions/               # 生成或导入的视频
├── predictions.jsonl          # Baseline/Evaluator 边界
├── logs/                      # 模型和评估日志
├── artifacts/                 # checkpoint、导入和 prediction digest
├── evaluation/               # case 曲线、分数和 Task 汇总
├── state.json
└── run.json
```

公共 artifact validator 对所有非空 `video_path` 执行：

- 路径必须位于当前 run；
- 文件必须存在；
- `status=complete` 必须有视频；
- 记录相对路径、大小和 SHA-256。

submission 和历史预测通过原子复制进入 run，并保留 source provenance。软链接不能绕过
run-local 约束。

## 9. Scene evaluation

TaskEvaluator 以 canonical jobs 为主表，按 `scene_id` 调度 CaseEvaluator：

```text
prediction
→ common timeline
→ masks / observations
→ scene physical state
→ reference similarity
→ strict Task aggregation
```

正式 evaluator 满足 `S(reference, reference) = 1`。无同 case GT 的 OOD 数据只能使用
Dataset 明确登记且物理标注对应的 parent reference；没有可信 parent 时返回
`no_trustworthy_physics_reference`，不伪造 GT。缺失 prediction 或部分 coverage 不会
产生正式 Task score。

## 10. 数据污染与可比性

模型部署可复现不等于结果可比较。历史训练语料与 Dataset 重叠时：

1. Bundle 必须声明 diagnostic/non-comparable；
2. provenance 要按 source identity 审计，而不只比较 case ID；
3. 全量结果不能进入无泄漏排名；
4. clean subset 必须明确列出，不能代替正式 Task score。

G15 的 source-aware audit 发现 176/214 个源 case 重叠，其中精确 case ID 只能识别
45 个。

## 11. 不变量

1. `datasets/` 是唯一权威数据根。
2. Dataset 不保存 prompt、模型 cache 或 prediction。
3. TaskSpec 不保存 Baseline 私有执行参数。
4. Registry 不包含具体模型分支。
5. Baseline 不能修改 canonical plan。
6. 公共 compiler 不向 generic 适配提供结构化物理标注。
7. 构建与执行使用同一 Bundle/deployment identity。
8. checkpoint 和所有输出相关依赖必须可追踪。
9. prediction 视频必须 run-local。
10. 缺失 case 和部分 coverage 必须显式暴露。
11. GT-dependent 诊断不能在无 GT 时伪造。
12. reference 自比的正式 case score 必须精确为 1。
13. 外部数据变换必须有可重放 provenance。
14. 训练重叠必须按 source identity 审计。
15. 所有生成模式必须声明非空语言文本 binding。
16. direct-eval 的模型输入不得绑定 evaluator reference；V2V 只能使用独立条件视频。
