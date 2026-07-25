# 系统架构

## 1. 领域对象

Benchmark 由五个不可混淆的领域对象构成。

| 对象 | 所有者 | 负责内容 | 不负责内容 |
| --- | --- | --- | --- |
| DatasetSnapshot | Dataset | case、资产、物理标注、View、哈希 | 模型 prompt、tensor、缓存 |
| TaskSpec | Benchmark | case 选择、任务族、conditioning、seed、评估协议 | 模型执行参数 |
| Baseline bundle | Baseline | 能力声明、TaskBuilder、DataAdapter、trainer、predictor | 修改 Dataset |
| BaselineTaskInstance | TaskBuilder | 冻结 jobs、模型原生输入、DAG、指纹 | 动态重新选 case |
| AtomicRun | Orchestrator | 执行、predictions、evaluation、状态 | 合并不同任务条件 |

它们的依赖方向是单向的：

```text
DatasetSnapshot ─┐
TaskSpec ─────────┼─> CanonicalTaskPlan ─┐
Baseline bundle ─┘                       ├─> BaselineTaskInstance
                                        │
                                        └─> AtomicRun
                                              ├─> predictions
                                              └─> evaluation
```

Dataset 不认识 Baseline；Task 不携带模型输入；Evaluator 不由 Baseline 实现。

## 2. Dataset 边界

正式 Dataset 是 `physics_video_five_scene_v3`：

```text
datasets/physics_video/releases/3.0.0/dataset.json
```

Descriptor 只引用版本内 metadata，并通过 `asset_root` 指向统一资产根。加载器会冻结：

- descriptor 和 release 指纹；
- `cases.jsonl`；
- scene catalog；
- View A / View B；
- `assets.lock.json`；
- 所有被引用资产的相对路径和 SHA-256。

Case 只包含模型无关事实：scene、物理量、环境、OOD 信息、时间语义、资产和来源。
Prompt、模型尺寸、latent、embedding、训练 metadata 均不属于 Dataset。

## 3. Task 规划

TaskSpec 使用 schema `2.0`。Planner 根据 Dataset View 产生
`CanonicalTaskPlan`，其 `jobs` 是后续推理和评估的主表。

四类正式任务：

| family | conditioning | 数据视图 | 是否训练 |
| --- | --- | --- | --- |
| `finetune_eval` | `generic` | View A | 是 |
| `finetune_eval` | `physics` | View A | 是 |
| `direct_eval` | `generic` | View B | 否 |
| `direct_eval` | `physics` | View B | 否 |

同一 family 下 generic 与 physics 必须共享 case、partition、seed 和生成规格。
它们只允许在 Baseline 私有条件适配阶段发生差异。

## 4. Baseline-owned TaskBuilder

核心 Planner 只回答“评哪些 case”。TaskBuilder 回答“该 Baseline 如何执行”。

TaskBuilder 输入：

```text
DatasetSnapshot + TaskSpec + CanonicalTaskPlan + Baseline bundle
```

输出是不可变 `BaselineTaskInstance`：

```text
envelope
├── dataset_snapshot
├── task_snapshot
├── baseline_snapshot
├── canonical_plan
├── native_jobs
├── operation_dag
├── adaptations
└── fingerprint
```

`native_jobs` 可携带任意 Baseline 私有输入，但必须位于
`baseline_payload`/`native_inputs` 边界内。公共 orchestration 不解释模型 tensor。

## 5. Operation DAG

`finetune_eval` 的典型 DAG：

```text
materialize_training_inputs
→ fine_tune
→ materialize_inference_inputs
→ generate
→ evaluate
```

`direct_eval`：

```text
resolve_frozen_checkpoint
→ materialize_inference_inputs
→ generate
→ evaluate
```

操作引用的是符号 artifact，而不是运行前猜测的文件路径。执行器必须先验证
TaskInstance 指纹，防止 jobs、conditioning 或 checkpoint 被静默替换。

## 6. DataAdapter

DataAdapter 是 Baseline 内部组件，当前统一分成：

1. 空间适配；
2. 时间适配；
3. 输入范式映射；
4. 文本条件适配；
5. 物理信息注入。

每个阶段都产生结构化审计。媒体派生物写入内容寻址、不可变 cache；同一输入和配置
得到相同 key。generic 分支在适配期间不得读取 case 的 `physics`。

## 7. AtomicRun

AtomicRun 冻结一个不可混合组合：

```text
DatasetSnapshot × AtomicTask × BaselineSnapshot × Seeds
```

标准目录：

```text
runs_v2/<run_id>/
├── run.json
├── task_instance.json
├── plan.json
├── predictions.jsonl
├── artifacts/
├── logs/
└── evaluation/
    ├── manifest.json
    ├── case_results.jsonl
    ├── task_result.json
    └── cases/<job_id>/
```

`predictions.jsonl` 是 Baseline 与 Benchmark evaluator 的硬边界。Baseline 完成生成后
不得自行定义正式分数。

## 8. Scene-aware evaluation

TaskEvaluator 以冻结 `plan.jobs` 为主表，按 `scene_id` 调度 CaseEvaluator。

```text
prediction video
→ common media timeline
→ scene observation/masks
→ scene-local state
→ reference similarity score
→ Task strict aggregation
```

公共层负责媒体采样、SAM2、mask 质量、质心/实例跟踪、轴/圆拟合和曲线；
scene 模块负责主体提示、状态提取、质量阈值和评分。

正式分数满足自反性：

```text
S(reference, reference) = 1
```

参考视频自身的噪声或物理残差只作为诊断；相似度只能由 reference 与 prediction 的
差值产生。详细定义见 [EVALUATION.md](EVALUATION.md)。

## 9. 不变量

实现和扩展必须持续满足：

1. `datasets/` 是唯一权威数据根。
2. Dataset 不保存模型条件或模型缓存。
3. TaskSpec 不包含 Baseline 私有执行细节。
4. generic 适配不能观察结构化物理标注。
5. TaskInstance 在执行前后使用同一指纹。
6. 缺失 prediction 必须产生显式 case 状态。
7. 部分 coverage 不能产生正式 Task score。
8. GT-dependent 诊断不能在无 GT 时伪造。
9. 相同参考输入的 case score 必须精确为 1。
10. 每次外部数据或模型变换必须留下可重放 provenance。
