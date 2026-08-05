# Task、Canonical Plan 与运行矩阵

## 1. Task 的职责

Task 是模型无关的评测定义。当前schema 4.0包含：

| 字段 | 含义 |
| --- | --- |
| `task_id` | 稳定任务身份 |
| `family` | `finetune_eval` 或 `direct_eval` |
| `dataset_id` / `dataset_view` | 数据快照与 View |
| `selection` | scene、test regime、group或显式case |
| `seeds` | 训练与推理 seed |
| `evaluation` | Benchmark-owned 评估协议 |

Task 不声明模型、输入范式、prompt 模板，也不声明是否使用结构化物理信息。后两者都是
Baseline 的固定属性。

所有可能进入文件路径的 Dataset、Task、Case、Baseline、run、matrix、job 与
adaptation ID 都必须匹配 `^[A-Za-z0-9][A-Za-z0-9_.-]*$`。路径分隔符、`..`、空白与
冒号不属于合法 ID；CLI、planner、compiler 和 TaskInstance loader 都会重复校验。

## 2. 官方 Task

当前Dataset有六个scene，但推水瓶评估器尚未定义。两份官方Task因此继续选择已有评估器
的五个scene，并已切换到10.0.0 Dataset：

| 文件 | family | View | 训练 |
| --- | --- | --- | --- |
| `tasks/official/five_scene_finetune_eval.json` | `finetune_eval` | A | 是 |
| `tasks/official/five_scene_direct_eval.json` | `direct_eval` | B | 否 |

推水瓶可以进入Dataset训练和基线数据适配，但在专用评估器及协议完成前不能计入官方
物理分数。不得用其它scene的评估器代替。

Fine-tune + eval 示例：

```json
{
  "schema_version": "4.0",
  "task_id": "five_scene_finetune_eval_v10",
  "family": "finetune_eval",
  "dataset_id": "physics_video_six_scene_v10",
  "dataset_view": "view_a",
  "selection": {
    "scene_ids": [
      "pendulum",
      "collision_1d",
      "inclined_plane_slide",
      "uniform_circular_motion",
      "parabolic_motion"
    ],
    "test_regimes": ["id"]
  },
  "seeds": {
    "training": [42],
    "inference": [42]
  },
  "evaluation": {
    "protocol": "scene_default_v10",
    "reporting": {
      "primary_score": "overall_test",
      "breakdowns": [],
      "minimum_subgroup_jobs": 5
    }
  }
}
```

两份官方Task固定`scene_default_v10`。不同Dataset digest或protocol identity的分数不能
混合。

`finetune_eval` 必须使用 View A，且一个 AtomicRun 恰好有一个 training seed；
`direct_eval` 必须使用 View B，且没有 training seed。多个 seed 应展开成多个独立
AtomicRun，不能在同一模型产物中混合。

schema 4.0不再选择`test_id/test_ood1` partition。finetune Task只运行`test`，总体Test
分是主分；当前View A的test全部为ID，官方Task固定选择`["id"]`，不再报告OOD/mixed或
OOD factor分组。

Baseline仍必须显式声明支持这五个scene。新增推水瓶支持后，还需等推水瓶评估协议完成
再扩展官方Task；Task不会绕过Baseline能力检查。

## 3. CanonicalTaskPlan

Planner 只读取 Dataset 与 Task：

```text
DatasetSnapshot + TaskSpec
→ scene / partition selection
→ train_case_ids
→ inference jobs × seeds
→ CanonicalTaskPlan
```

Plan schema 4.0 的核心结构：

```text
task_id / family
dataset_id / dataset_digest
scene_ids
train_case_ids
training_seed
jobs[]
  ├── job_id
  ├── case_id
  ├── scene_id
  ├── evaluation_partition
  └── seed
evaluation_annotations[job_id]
  ├── generalization_regime
  ├── ood_factors[]
  └── co_varying_factors[]
reporting_policy
```

`jobs` 是生成和评估的唯一主表。schema 4.0的finetune job统一使用`test`作为
`evaluation_partition`；泛化标签是只读报告元数据。Baseline不得重新选例、改变划分、替换seed
或跳过难例。Job ID 由 Task、case 与 seed 构成，不编码 Baseline 的物理使用策略。

## 4. 一份 Task，多种 Baseline

当前比较维度是：

```text
one Dataset × one Task × many Baseline identities
```

例如 `cosmos3_nano_i2v_generic` 和 `cosmos3_nano_i2v_physics` 都编译
`five_scene_direct_eval.json`。两者接收同一Case prompt、首帧与annotated物理标注；
前者通过 `input_policy.physics.usage=ignored` 明确不消费物理字段，后者通过
`usage=required` 与 `structured_text` adapter 追加物理信息。

这两个 Baseline 必须共享相同的：

- Dataset digest；
- scene、case 与 partition；
- 训练/推理 seed；
- canonical job ID；
- 评估协议。

允许不同的是：

- Baseline/model identity；
- input policy 与 adapter fingerprint；
- native model input；
- trainer、runner、checkpoint 与生成结果。

`matrix-run` 会在创建 run 前比较 paired plan signature；任一 Baseline 改变 data、
split、seed 或 job 时立即失败。

## 5. Task 与 Baseline capability

Baseline schema 5.0 通过 `capabilities.task_families` 声明支持的 Task family：

- `cosmos3_nano_i2v_*`：只支持 `direct_eval`；
- `wan22_g15_sparse_motion_r32_e20_*`：只支持 `direct_eval`；
- `wan22_ti2v_5b_lora_r32_v3_*`：支持 `finetune_eval` 和 `direct_eval`。
- `wan22_ti2v_5b_lora_r32_quantity_embedding_v1`：只支持
  `finetune_eval`，联合训练 DiT LoRA 与物理量编码器。

`finetune_eval` 还要求 Bundle 提供 `trainer` recipe。Submission 只支持
`direct_eval`。能力不匹配会在 run 目录创建前失败。

## 6. BaselineTaskInstance

TaskBuilder 把公共 plan 与某个 Baseline 编译成 schema 3.0
`BaselineTaskInstance`：

```text
task_instance
├── identity
│   ├── dataset / task
│   ├── baseline bundle / deployment
│   ├── TaskBuilder / DataAdapter fingerprints
│   └── canonical plan digest
├── semantics
├── canonical_plan
├── source.cases / asset_root
├── adaptations
├── training
├── inference.jobs
├── execution_graph
├── cache_bindings
├── baseline_payload
└── instance_digest
```

它是 canonical JSON seal。修改 plan、prompt、used parameter、checkpoint、DAG 或媒体
binding 都会改变或破坏 digest。

## 7. 编译 TaskInstance

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/releases/10.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --output /tmp/wan22_physics_task_instance.json
```

`--baseline` 接受 Baseline ID、Bundle 目录或具体 manifest 路径。目录引用默认解析
`baseline.json`；若要直接指定同目录的第二份 manifest，可传
`baselines/<bundle>/physics.baseline.json`。

## 8. 创建 AtomicRun

不加 `--execute` 时冻结计划、展开 adapter 与 job，但不启动模型：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/10.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --output-root runs_v2 \
  --run-id wan22_generic_dryrun
```

确认 TaskInstance 与本机部署后，使用新的 run ID 并添加 `--execute`。AtomicRun 不覆盖
已有目录。

过程可视化视频默认不保存，以避免大量占用磁盘。需要人工审计时显式添加：

```text
--save-visualizations
```

启用后结果写入
`runs_v2/<run_id>/evaluation/visualizations/<scene>/<case>/...`；该参数只控制运行内的
诊断视频，不改变评测分数。并存式重评的视频写入对应 reevaluation 的
`evaluation/visualizations/`，不会修改 canonical evaluation。

Direct-eval 工程 smoke 可加：

```text
--scene-id <scene>
--group <view_b_group>
--case-id <case_id>
```

`--group` 与 `--case-id` 只适用于 `direct_eval`。这些运行时筛选会生成新的 Task digest，
仅用于诊断；部分 coverage 不是正式 Task 结果。

## 9. 运行矩阵

同一 Task 比较 generic/physics Baseline：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/releases/10.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --matrix-id wan22_prompt_injection_ablation \
  --output-root runs_v2
```

至少需要两个不同 Baseline ID。加 `--execute` 才运行模型。输出为：

```text
runs_v2/wan22_prompt_injection_ablation.matrix.json
runs_v2/wan22_prompt_injection_ablation__wan22_ti2v_5b_lora_r32_v3_generic/
runs_v2/wan22_prompt_injection_ablation__wan22_ti2v_5b_lora_r32_v3_physics/
```

矩阵索引冻结每个 Bundle、deployment、input policy、TaskBuilder fingerprint 与
TaskInstance digest。`orchestration_status=complete` 表示所有 AtomicRun 已成功创建并
通过预编译 digest 核对；聚合 `status` 才表示子运行状态，所以 dry-run 正常为
`status=planned`。

## 10. 新增 Task 的判断准则

只有以下变化才需要新 Task：

- family 或训练/评估生命周期改变；
- Dataset View、scene、partition 或 case selection 改变；
- seed policy 改变；
- train/test或test generalization selection改变；
- 正式评估协议改变。

以下变化应创建或升级 Baseline，而不是复制 Task：

- 是否使用结构化物理信息；
- 物理信息写入 prompt、token、trajectory、mask、flow 或控制视频；
- T2V/I2V/V2V 范式；
- prompt 后处理；
- 模型、checkpoint、trainer 或 runner 改变。

## 11. 历史兼容

旧任务文件曾把generic/physics作为Task字段或文件名的一部分。schema 3.0与旧Dataset
Task只用于历史run复现；新实验必须使用schema 4.0的当前两份官方Task。schema 4.0
loader会拒绝`ood2`和`eval_partitions`等旧字段，避免把旧三分法重新带入新结果。
