# Task、TaskBuilder 与 Baseline 接入

## 1. 原子 Task

正式任务位于 `tasks/official/five_scene_*.json`。TaskSpec 使用 schema `2.0`，
只表达 Benchmark 意图：任务族、conditioning、数据视图、case 选择、seed 和评估协议。
模型路径、分辨率、prompt 模板、训练器参数都不属于 TaskSpec。

四个正式组合是：

| family | conditioning | Dataset View | 训练 |
| --- | --- | --- | --- |
| `finetune_eval` | `generic` | View A | 是 |
| `finetune_eval` | `physics` | View A | 是 |
| `direct_eval` | `generic` | View B | 否 |
| `direct_eval` | `physics` | View B | 否 |

一个 Task 只能包含一个 family 和一个 conditioning。generic/physics 的对照实验必须
产生两个独立 AtomicRun。

## 2. Benchmark-owned CanonicalTaskPlan

Planner 校验 Dataset ID、View、scene、partition、OOD2 和 seed 后产生
`CanonicalTaskPlan`：

```text
canonical_plan
├── dataset_id / dataset_digest
├── task_id / family / conditioning
├── scene_ids
├── train_case_ids
├── training_seed
└── jobs[]
    ├── job_id
    ├── case_id
    ├── scene_id
    ├── evaluation_partition
    ├── conditioning
    └── seed
```

`jobs` 是 prediction 和 evaluation 的唯一主表。Baseline 只能把这份计划编译成模型
原生输入，不能重新抽样、丢弃难例或替换 partition。通用宿主会在 TaskBuilder 返回后
逐字比较 canonical plan 并验证其 SHA-256。

## 3. 自注册 Baseline Bundle v3

每个 Baseline 是 `baselines/` 下的一个独立目录。核心通过
`baselines/*/baseline.json` 自动发现，不 import 目录中的代码：

```text
baselines/<name>/
├── baseline.json
├── README.md
├── baseline.local.example.json
├── baseline.local.json              # 可选，本机配置，必须忽略提交
├── plugin/
│   └── main.py
└── ...                              # 模型私有代码、profile 和配置
```

schema `3.0` 的最小 manifest：

```json
{
  "schema_version": "3.0",
  "baseline_id": "my_video_baseline",
  "baseline_version": "1.0.0",
  "implementation": {
    "kind": "command",
    "protocol": "physbench-baseline-v1",
    "entrypoint": ["{python}", "plugin/main.py"],
    "fingerprint_paths": [
      "plugin/**/*.py",
      "profiles/*.json"
    ]
  },
  "capabilities": {
    "task_families": ["finetune_eval", "direct_eval"],
    "conditioning": ["generic", "physics"]
  }
}
```

核心 Registry 只认识通用实现类型 `command`，不认识 WAN、CogVideoX 或任何具体模型名。
因此加入新 Baseline 不需要修改 `src/physbench/baseline_api/registry.py`。

### 发现与引用

`--baseline` 支持三种等价引用：

```text
my_video_baseline
baselines/my_video_baseline
baselines/my_video_baseline/baseline.json
```

按 ID 引用时，重复 `baseline_id` 会立即报错。不存在的路径不会被误当作 ID。

### 两类指纹

Bundle v3 明确区分可移植实现和本机部署：

```text
bundle digest
= canonical(manifest + fingerprint_paths 中每个文件的相对路径和 SHA-256)

deployment digest
= canonical(应用 baseline.local.json 后的 manifest)
```

插件代码或 profile 改变会使 bundle digest 改变。本机 checkpoint、模型根目录、Python
或 GPU 设置改变会使 deployment digest 改变。两者都进入 TaskBuilder 和
BaselineTaskInstance 身份，避免“代码相同但实际模型不同”或“配置相同但代码已变”。

`baseline.local.json` 只允许覆盖 `runtime` 和 `model`；覆盖 capabilities、
implementation、ID 或版本会被拒绝。

仓库内被多个 Bundle 复用的实现不复制进每个目录，而由 TaskBuilder 对共享文件逐个
计算 SHA-256，写入 `runtime_dependency_fingerprints`。因此 bundle digest 负责
Bundle-local 边界，TaskBuilder fingerprint 负责完整可执行依赖；两者不能互相替代。

## 4. Command Protocol

通用宿主使用临时 request/response JSON 文件调用 Bundle：

```bash
python plugin/main.py --request request.json --response response.json
```

协议固定为 `physbench-baseline-v1`，包含四个操作：

| operation | 输入 | 输出 |
| --- | --- | --- |
| `describe` | Bundle snapshot | TaskBuilder/DataAdapter 描述和指纹 |
| `adapt_case` | case、conditioning、role | 可审计 adaptation record |
| `build_task_instance` | Dataset、Task、canonical plan | sealed instance document |
| `run_task` | sealed instance、run_dir、执行开关 | training stage、predictions |

`{python}` 由宿主替换为当前 Benchmark Python。工作目录固定为 Bundle root，Bundle 内
相对路径不会依赖调用者所在目录。入口脚本和 fingerprint glob 禁止绝对路径、`..` 与
符号链接逃逸。

发现阶段只读取 JSON。只有显式 `inspect`、`validate`、`task-build` 或运行任务时才会
执行 Bundle command。

## 5. TaskBuilder 与 DataAdapter

Benchmark 核心保留抽象接口，命令型 Baseline 通过 proxy 实现：

```python
TaskBuilder.build(dataset, task)
  -> Benchmark planner creates CanonicalTaskPlan
  -> command build_task_instance(...)
  -> host verifies identity and seal
  -> BaselineTaskInstance
```

Baseline 自己负责：

1. 检查 family、conditioning、scene 和模型部署能力；
2. 绑定 Dataset 资产但不修改它们；
3. 通过 DataAdapter 构建训练与评测输入；
4. 生成训练节点、推理 jobs、operation DAG 和 artifact 引用；
5. 把模型专有数据限制在 `native_inputs` 与 `baseline_payload`；
6. 返回确定性、已封印的 TaskInstance；
7. 执行 TaskInstance 并为每个 job 返回 prediction record。

通用宿主负责验证：

- Dataset ID/digest 与 Task ID/digest 未变；
- baseline ID、bundle digest、deployment digest 未变；
- TaskBuilder/DataAdapter fingerprint 与 `describe` 一致；
- canonical plan 内容和 digest 未变；
- `instance_digest` 正确；
- 执行时使用的是同一 Baseline 部署。

## 6. BaselineTaskInstance

公共 envelope 使用 schema `2.1`：

```text
task_instance
├── instance_id / instance_digest
├── identity
│   ├── dataset
│   ├── task
│   ├── baseline
│   │   ├── baseline_id / baseline_version
│   │   ├── digest
│   │   └── deployment_digest
│   ├── task_builder
│   ├── data_adapter
│   └── canonical_plan_digest
├── semantics
├── canonical_plan
├── source
├── adaptations
├── training
├── inference.jobs
├── execution_graph
├── cache_bindings
└── baseline_payload
```

修改 jobs、conditioning、媒体绑定、checkpoint、DAG 或 identity 后，实例 seal 都会
失效。

## 7. 接入新 Baseline

接入流程不需要修改核心源码：

1. 新建 `baselines/<name>/`。
2. 编写 Bundle v3 `baseline.json`，给出全局唯一的 `baseline_id`。
3. 实现 `physbench-baseline-v1` 四个操作；可使用
   `physbench.baseline_api.endpoint.main` 作为无模型逻辑的协议分发器。
4. 把所有 Bundle 自有、会影响输出的代码和配置加入 `fingerprint_paths`；不可避免的
   外部代码依赖必须逐文件进入 TaskBuilder fingerprint 和 `describe` 审计。
5. 提供 `baseline.local.example.json`，把机器路径放入被忽略的
   `baseline.local.json`。
6. 让 generic adapter 不观察 `case.physics`；physics adapter记录使用字段。
7. 为 canonical plan 不变性、TaskBuilder 确定性、部署指纹、数据不可变性和失败状态
   增加测试。
8. 不在 Baseline 内定义正式 evaluator；只输出 `predictions`。
9. 若模型已有训练语料，按 source identity 审计 Dataset 重叠；有污染的预训练模型
   必须声明 diagnostic/non-comparable。
10. executor 必须接收 `run_dir` 并把预测、stdout/stderr、训练曲线和中间审计写入
    该目录；除模型代码、权重和可重建 cache 外，不得让 AtomicRun 依赖外部文件。

已有模型族应复用一个经过测试的共享实现。例如两个 WAN Bundle 的入口都调用
`src/physbench/baseline_plugins/wan22.py`，不会复制 DataAdapter、media adapter 或
executor。全新模型族可以把实现放在自己的 Bundle 中；核心 Registry 无需改动。

Prediction 的最低执行边界：

```json
{
  "job_id": "...",
  "case_id": "...",
  "baseline_id": "...",
  "evaluation_partition": "test_id",
  "status": "complete",
  "video_path": "/absolute/path/to/runs_v2/<run_id>/predictions/video.mp4"
}
```

核心不会信任 `video_path` 声明本身：完成态 prediction 写入
`predictions.jsonl` 前，必须通过 run-local 路径、文件存在性和 SHA-256 检查。历史或
人工生成的视频使用 `physbench prediction-import` 复制进入 run；不能直接引用模型
仓库，也不能用软链接规避边界。

## 8. 管理与运行命令

```bash
# 只扫描 manifests，不执行插件
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list

# 解析本地覆盖并查看组件描述
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect wan22_ti2v_5b_lora_r32_v3

# 验证 manifest、代码指纹和 command endpoint
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3

# 编译 sealed TaskInstance
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_generic.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output /tmp/task_instance.json
```

## 9. 当前 Bundle 与能力

| baseline ID | family | conditioning | 备注 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3` | `finetune_eval`, `direct_eval` | generic, physics | View A LoRA 基线 |
| `cosmos3_nano_i2v` | `direct_eval` | generic, physics | base Cosmos3-Nano |
| `wan22_g15_sparse_motion_r32_e20` | `direct_eval` | generic, physics | 冻结 G15，诊断型 |

TaskBuilder 会在创建 run 前拒绝 manifest 未声明的 family。特别地，Cosmos 和 G15
收到 `finetune_eval` 时必须失败；它们不会伪造空训练阶段来绕过 View A 语义。
