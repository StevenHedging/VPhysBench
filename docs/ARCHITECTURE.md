# 系统架构

## 1. 领域边界

当前架构把“数据是什么”“评什么”“模型如何使用输入”分成三个独立决策：

| 对象 | 所有者 | 负责 | 不负责 |
| --- | --- | --- | --- |
| DatasetSnapshot / Case | Dataset | 原始 prompt、媒体、物理/环境标注与View | 模型输入格式、推理参数 |
| TaskSpec | Benchmark | family、选择、seed、报告策略、评估协议 | 是否使用物理信息、prompt拼接、模型路径 |
| CanonicalTaskPlan | Benchmark | 训练 case、评测 job、partition 与 seed 主表 | native model input |
| Baseline Bundle | Baseline | 模型身份、能力、input policy、adapter、trainer、runner/driver | 改写数据划分和正式评分 |
| BaselineTaskInstance | Compiler | 适配结果、执行图、cache binding、身份 seal | 重新抽样或换 seed |
| AtomicRun | Orchestrator | 冻结输入、执行、预测、评估、状态与制品 | 混合多个 Baseline identity |

依赖方向：

```text
DatasetSnapshot ─┐
                  ├─> CanonicalTaskPlan ────────────────┐
TaskSpec ─────────┘                                      │
                                                         ├─> sealed
Baseline Bundle ─> DataAdapter ─> ManagedTaskBuilder ────┘   TaskInstance
                                                               │
                                                               ▼
                                                           AtomicRun
                                                               │
                                                               ▼
                                                         TaskEvaluator
```

Dataset 不认识具体模型；Task 不携带模型输入策略；Baseline 不定义正式 evaluator。

## 2. Dataset 与 Case

当前入口：

```text
datasets/releases/13.0.0/dataset.json
```

Dataset和Loader物化后的Case使用schema 5.0。每个Case同时拥有：

- `text.prompt`：从Case-local `caption.json`读取的模型无关原始文本描述；
- `assets.first_frame` 等媒体；
- `physics`：从Case-local `physics.json`读取的结构化物理量；
- `assets.caption`：Case-local `caption.json`；
- `assets.physics_annotation`：最小Case-local `physics.json`；
- `appearance`、`temporal`；
- evaluator-only `assets.reference_video`。

划分不属于Case字段；当前View A只包含train和ID test。原始caption与物理标注均只在
Case资产目录保存一份，不由Task或Baseline临时生成。Release 13.0.0的`cases.jsonl`是
轻量索引，仅含身份、运行时资产、appearance与temporal；Loader严格读取两个Case-local
JSON并物化兼容的`case.text`和`case.physics`。逐Case来源、采集时序说明与alignment在
`datasets/provenance/releases/13.0.0/cases.jsonl`中，不进入运行时Case。
正式标量只含`value/unit/symbol`；正式时序量只含
`samples/time_unit/unit/symbol`，sample只含`time/value`。quantity的symbol必须进入
无数值英文prompt；辅助和审计量不进入运行时`physics`。所有正式数值为非负大小，方向
由prompt表达。推水瓶的完整`F(t)`绑定到`objects.object_1.applied_force`；旧四字段
quantity不再属于活动运行契约。

## 3. Task 是模型无关的评测定义

当前Task schema 4.0只允许以下字段，并显式兼容当前Dataset schema 5.0：

```text
schema_version
task_id
family
dataset_id / dataset_view
selection
seeds
evaluation
```

两份官方 Task：

| 文件 | family | View |
| --- | --- | --- |
| `tasks/official/five_scene_finetune_eval.json` | `finetune_eval` | A |
| `tasks/official/five_scene_direct_eval.json` | `direct_eval` | B |

Task中没有物理使用开关。Planner只依赖Dataset与Task，生成schema 4.0
`CanonicalTaskPlan`：

```text
task_id / family / dataset identity
scene_ids
train_case_ids / training_seed
jobs[]
  ├── job_id
  ├── case_id / scene_id
  ├── evaluation_partition
  └── seed
evaluation_annotations[job_id]
reporting_policy
```

finetune job的主partition统一为`test`。当前View A中的test全部是ID；兼容字段
`evaluation_annotations`仍随job封存，但不再构造OOD/mixed分支。总体Test仍是官方主分。

Job ID 也不包含模型输入策略。不同 Baseline 编译同一 Dataset + Task 时，上述 plan
必须完全一致。

## 4. Baseline 拥有输入策略

新任务只接受 schema 5.0 Baseline Bundle。一个 Bundle 固定声明：

```json
{
  "input_policy": {
    "schema_version": "1.0",
    "case_view": "conditionable_case_v1",
    "text": {
      "source": "case.text.prompt",
      "usage": "required"
    },
    "physics": {
      "source": "case.physics",
      "usage": "ignored",
      "representations": []
    }
  }
}
```

`physics.usage` 的含义：

| 值 | 契约 |
| --- | --- |
| `ignored` | 不得登记 used parameter 或 physics channel |
| `optional` | 可按 case 使用；使用时必须同时登记参数和 channel |
| `required` | 每条 adaptation 必须消费至少一个 formal quantity并登记 channel |

`representations` 描述模型侧表示，如 `structured_text`、`numeric_tokens`、
`trajectory`、`mask`、`optical_flow`、`force_field` 或 `control_video`。标准 adapter
当前实现 `structured_text`；其它表示使用 Bundle-local Python adapter 扩展。

generic/physics 只是当前 Baseline ID 的命名约定，不是 Task 的两个分支。例如：

```text
cosmos3_nano_i2v_generic  → physics.usage=ignored
cosmos3_nano_i2v_physics  → physics.usage=required
```

二者可以共享模型 checkpoint、driver、本机配置和媒体 materialization，但必须拥有不同
Baseline ID、Bundle digest、DataAdapter fingerprint 与 TaskInstance。

## 5. Conditionable Case 与信任边界

Compiler 给每个 adapter 同一种 `conditionable_case_v1`：

```text
case_id / scene_id
text
appearance / temporal / ood
允许作为生成输入的 assets
physics 中的全部正式quantity（由Compiler投影为稳定语义名）
```

这些quantity保留其完整标量或时序结构，使adapter能以符号为绑定键选择独立物理量；
Dataset中的数值不会因进入conditionable Case而自动拼接到原始prompt。当前内置
structured-text和quantity-embedding adapter是标量型消费者：它们只选择含`value`的
quantity，不会把时序量静默压缩为均值或峰值。使用完整`F(t)`的Baseline必须显式实现并
声明trajectory、force-field或其它时序representation。

它不会提供：

- evaluator reference或source video；
- provenance、原始定位和 alignment 证据；
- provenance中的派生、辅助和审计量；
- training target。Compiler 只在 sealed runtime source 的训练 case 中另加
  `supervised_targets`，供 trainer 使用；adapter 与 eval predictor 均看不到。

相同 Case 投影让 `ignored` 与 `required` Baseline 可在同一接口上实现。`ignored` 是
Baseline manifest、adapter 输出和 contract 验证共同保证的审计承诺，而不是对恶意
Bundle 代码的机密性沙箱。Bundle-local Python adapter/driver 是受信任代码；若未来
执行不受信任第三方代码，还需要 opaque case handle、无语义资产别名和进程级隔离。

## 6. DataAdapter 与输入 contract

接口固定为：

```python
adapt_case(case, *, role)
```

其中 `role` 为 `train` 或 `eval`。Adapter 返回：

```text
native_inputs
used_parameters
input_contract
  ├── generation_mode
  ├── required text binding
  ├── media_channels
  ├── physics_channels
  └── asset_access
adapter / materialization fingerprints
```

公共 compiler 验证文本非空、媒体与物理 channel、参数值和单位、资产白名单、producer
fingerprint 及 Baseline capability，但不解释模型专有 `native_inputs`。

输入范式与物理使用是两个正交维度：

| generation mode | 媒体要求 |
| --- | --- |
| T2V | 无媒体 channel |
| I2V | 图像 channel；标准 adapter 使用 `assets.first_frame` |
| V2V | 独立视频输入 channel |
| hybrid | 同时有图像和视频 channel |

V2V 中的 `conditioning_video` 是媒体 channel 的角色名，表示“作为模型输入的视频”，
不是 Task 层的物理注入实验臂。它必须来自明确的独立输入资产或经审计的派生 artifact，
禁止使用`reference_video`或`source_video`冒充输入。

大型控制表示必须使用
`artifact://sha256/<digest>` 或 `cache://sha256/<digest>`，并登记内容、producer 与
source digest；TaskInstance 不内嵌大数组。

## 7. Compiler 与 TaskInstance

`ManagedTaskBuilder` 是无副作用的确定性 compiler：

```text
Dataset + Task
→ canonical plan
→ Baseline capability check
→ adapt_case(..., role=train|eval)
→ input contract / asset / physics audit
→ training + inference jobs
→ execution graph + cache bindings
→ BaselineTaskInstance.seal
```

TaskInstance schema 3.0 冻结：

```text
identity
  ├── dataset / task
  ├── baseline bundle / deployment
  ├── TaskBuilder / DataAdapter
  └── canonical plan digest
semantics
canonical_plan
source.cases / asset_root
adaptations
training
inference.jobs
execution_graph
cache_bindings
baseline_payload
instance_digest
```

Seal 是 canonical JSON SHA-256 完整性校验，不是第三方数字签名。执行前会重新验证
instance、当前 deployment 和 canonical plan。

## 8. 身份与 cache

运行冻结三层身份：

1. `bundle digest`：portable manifest 与登记的 Bundle 文件；
2. `deployment digest`：应用 `baseline.local.json` 后的 model/runtime；
3. `TaskBuilder fingerprint`：adapter、runner/trainer、Bundle/deployment 与外部依赖。

`baseline.local.json` 只能覆盖 `model` 和 `runtime`，不能修改 ID、capability、
input policy 或 adapter。Checkpoint 还应声明稳定 revision、identity file 或完整
SHA-256；路径本身不构成模型身份。

完整 adapter fingerprint 包含文本和物理阶段；materialization fingerprint 只包含
空间、时间和输入范式。这样同模型的 generic/physics Baseline 可以复用相同媒体 cache，
同时保留不同输入语义和可审计身份。

## 9. AtomicRun 与矩阵

一个 AtomicRun 恰好对应：

```text
DatasetSnapshot × TaskSpec × Baseline identity × seeds
```

`matrix-run` 在同一 Task 上构建多个 AtomicRun，并先校验所有 Baseline 的 data、split、
seed 和 job 签名一致。矩阵索引位于：

```text
run/<matrix_id>.matrix.json
```

各元素位于：

```text
run/<matrix_id>__<baseline_id>/
```

矩阵索引区分编排终态和子运行终态：`orchestration_status=complete` 只表示全部
AtomicRun 已成功创建并核对 TaskInstance digest；`status` 才汇总子运行状态。因此
不执行模型的矩阵正常写为 `status=planned`，而不是 `complete`。

Run 冻结 Dataset/Task/Baseline、TaskInstance、job、预测、日志、评估和所有关键
fingerprint。除模型代码、权重与可重建 cache 外，输出必须位于当前 run。

可选过程视频也遵守这一所有权边界：canonical 结果写入
`evaluation/visualizations/`，并存式重评写入各 variant 自己的
`evaluation/visualizations/`；禁止使用机器级全局目录或逃逸 symlink。

## 10. Evaluation

Evaluator 以 canonical jobs 为唯一主表：

```text
prediction
→ common timeline
→ masks / observations
→ scene physical state
→ reference similarity
→ strict Task aggregation
```

不同分辨率和帧数由 evaluator 的 timeline、几何标准化和 reference-bounded sampling
处理。无同 case GT 的 OOD case 只可使用 Dataset 明确登记且物理对应的 parent
reference；没有可信 reference 时返回明确错误状态。Coverage 不完整时正式 Task score
为 `null`。

## 11. 兼容边界

历史冻结产物可能仍含旧 schema、旧 prompt metadata 或 evaluator compatibility
projection。兼容路径只消费已有 run 做重评，不再生成 plan、prompt 或 prediction。
它们不是新实验的输入契约：

- 新 Dataset/Case/Task/TaskInstance 使用 schema 3.0；
- 新 Baseline 使用 schema 5.0；
- 新 Task 禁止模型输入策略字段；
- 当前 Registry 不加载旧 v3/v4 Bundle；
- 历史记录里若出现旧字段，只能按 legacy metadata 解释，不能据此生成新 plan。
