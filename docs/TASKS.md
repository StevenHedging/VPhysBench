# Task、TaskBuilder 与 Baseline 接入

## 1. 原子 Task

正式任务位于 `tasks/official/five_scene_*.json`。每个 TaskSpec 使用 schema `2.0`，
只描述 Benchmark 意图：

```json
{
  "family": "finetune_eval",
  "conditioning": "physics",
  "dataset_id": "physics_video_five_scene_v3",
  "dataset_view": "view_a",
  "selection": {
    "scene_ids": ["pendulum", "free_fall", "collision_1d",
                  "inclined_plane_slide", "uniform_circular_motion"],
    "eval_partitions": ["test_id", "test_ood1"]
  },
  "seeds": {"training": [42], "inference": [42]},
  "evaluation": {"protocol": "scene_default_v1"}
}
```

一个 Task 只能有一个 family 和一个 conditioning。不同条件不能在同一 AtomicRun
中混合。

## 2. CanonicalTaskPlan

Planner 校验 Dataset ID、View、scene、partition、OOD2 和 seed 后生成计划：

```text
plan
├── task_id
├── family
├── conditioning
├── scene_ids
├── training_case_ids
└── jobs[]
    ├── job_id
    ├── case_id
    ├── scene_id
    ├── evaluation_partition
    ├── conditioning
    └── seed
```

`jobs` 是 prediction 和 evaluation 的唯一主表。Baseline 不能丢弃难例后重新构造
评测清单。

## 3. Baseline bundle

Baseline bundle 使用 schema `2.0`：

```text
baseline
├── baseline_id
├── plugin
├── supported_scenes
├── capabilities
├── model
├── runtime
└── components
    ├── task_builder
    ├── trainer
    └── predictor
```

能力检查必须在 materialization 前完成：

- family 是否支持；
- conditioning 是否支持；
- scene 是否完整覆盖；
- `finetune_eval` 是否支持 fine-tuning；
- `direct_eval` 的冻结 checkpoint 是否存在且可读。

## 4. TaskBuilder 接口

TaskBuilder 是 Baseline 所有的编译器：

```python
build(
    dataset_snapshot,
    task_spec,
    canonical_plan,
    baseline_bundle,
) -> BaselineTaskInstance
```

编译步骤：

1. 校验 bundle 能力和 Dataset/Task 一致性；
2. 绑定 Dataset 资产，不复制权威文件；
3. 调用 Baseline 私有 DataAdapter；
4. 生成训练输入和每个 inference job 的 `native_inputs`；
5. 构建 operation DAG；
6. 冻结所有 snapshot、adaptation audit 和 artifact 引用；
7. 计算 canonical SHA-256 指纹。

同一组输入必须确定性地产生相同 TaskInstance 指纹。

## 5. BaselineTaskInstance

公共 envelope 使用 schema `2.0`，模型专有数据必须放在不透明 payload 中：

```text
task_instance
├── schema_version
├── instance_id
├── dataset
├── task
├── baseline
├── canonical_plan
├── adaptations
├── native_jobs
├── operations
├── artifacts
└── fingerprint
```

执行器读取实例前重新计算指纹。任何对 jobs、prompt、媒体绑定、checkpoint 或 DAG 的
修改都会使实例失效。

## 6. Generic 与 Physics 配对

配对任务必须满足：

```text
same DatasetSnapshot
same family
same selected cases
same partitions
same seeds
same media transforms
different conditioning adaptation only
```

generic 分支禁止读取 `case.physics`。physics 分支可将允许的结构化值写入模型原生条件，
但必须记录使用了哪些字段和模板。

## 7. Operation DAG

`finetune_eval`：

```text
materialize_train
→ fine_tune
→ bind_checkpoint
→ materialize_inference
→ generate
→ evaluate
```

`direct_eval`：

```text
bind_frozen_checkpoint
→ materialize_inference
→ generate
→ evaluate
```

每个 operation 明确声明输入 artifact、输出 artifact 和执行组件。运行状态不能代替
TaskInstance；失败恢复仍必须从相同 sealed 实例继续。

## 8. 接入新 Baseline

1. 在 `baselines/<name>/baseline.json` 声明 bundle。
2. 在 `src/physbench/baseline_plugins/` 注册 plugin。
3. 实现 TaskBuilder 和 DataAdapter。
4. 将模型专有字段限制在 `native_inputs`。
5. 让 predictor 产生冻结 `predictions.jsonl`：

```json
{
  "job_id": "...",
  "case_id": "...",
  "status": "complete",
  "video_path": "/absolute/path/to/video.mp4"
}
```

6. 不在 Baseline 中实现正式 evaluator。
7. 增加以下测试：
   - bundle 能力拒绝；
   - TaskBuilder 确定性；
   - generic 不读取 physics；
   - 配对任务共享 case plan；
   - TaskInstance 篡改检测；
   - predictor 缺失和失败状态。

## 9. 命令

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_generic.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/task_instance.json
```

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```
