# Task、TaskBuilder 与 Baseline 接入

本文说明 Task、TaskBuilder 与 Baseline 接口的架构和契约。需要实际接入新模型时，请从
[自定义 Baseline 集成指南](BASELINE_INTEGRATION.md) 开始，按其中的脚手架、配置、
dry-run 和验收流程操作。

## 1. Task 与 canonical plan

正式任务位于 `tasks/official/five_scene_*.json`。TaskSpec schema 是 `2.0`，
只描述 Benchmark 意图：

| family | conditioning | Dataset View | 是否训练 |
| --- | --- | --- | --- |
| `finetune_eval` | `generic` | View A | 是 |
| `finetune_eval` | `physics` | View A | 是 |
| `direct_eval` | `generic` | View B | 否 |
| `direct_eval` | `physics` | View B | 否 |

模型路径、prompt 模板、分辨率、帧数和训练器参数不进入 TaskSpec。Planner 根据冻结的
Dataset View 产生唯一的 `CanonicalTaskPlan`：

```text
canonical_plan
├── dataset_id / dataset_digest
├── task_id / family / conditioning
├── scene_ids
├── train_case_ids / training_seed
└── jobs[]
    ├── job_id / case_id / scene_id
    ├── evaluation_partition
    ├── conditioning
    └── seed
```

`jobs` 是生成和评估的唯一主表。任何 Baseline 都不能重新抽样、跳过难例、改变 seed
或替换 partition。TaskBuilder 只能把计划编译成模型输入。

## 2. 三档 Baseline 接口

Registry 自动扫描 `baselines/*/baseline.json`。具体模型名不进入 Registry；加入和
移除 Baseline 都只改变一个目录。接口按复杂度分为三档：

| kind | schema | 适用情况 | 维护者需要实现 |
| --- | --- | --- | --- |
| `submission` | `4.0` | 已经生成好视频，只做统一评估 | manifest + submission JSONL |
| `managed` | `4.0` | 标准 T2V/I2V 推理；默认选择 | manifest + 薄 driver |
| `command` | `3.0` | 训练、多进程控制、特殊输入协议 | 完整四操作 endpoint |

`schema_version=3.0` command 是稳定的高级接口，没有废弃。`schema_version=4.0`
是低成本接口；managed 和 submission 复用 Benchmark 的 canonical compiler、
DataAdapter envelope、identity 校验和 prediction 组装。

### 2.1 Managed：标准接入

典型目录：

```text
baselines/my_i2v/
├── baseline.json
├── driver.py                       # 模型专有 prepare/execute
├── baseline.local.example.json
├── baseline.local.json             # 本机路径，Git ignored
├── README.md
└── provenance/                     # 可选，训练来源或污染审计
```

manifest 的关键部分：

```json
{
  "schema_version": "4.0",
  "baseline_id": "my_i2v",
  "baseline_version": "1.0.0",
  "implementation": {
    "kind": "managed",
    "driver": "driver.py",
    "fingerprint_paths": ["provenance/*.json"]
  },
  "supported_scenes": [
    "pendulum",
    "free_fall",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion"
  ],
  "capabilities": {
    "task_families": ["direct_eval"],
    "conditioning": ["generic", "physics"]
  },
  "model": {"model_id": "vendor/model", "checkpoint": null},
  "runtime": {},
  "adapter": {
    "preset": "standard_i2v_v1",
    "profile_set": "five_scene_i2v_v1",
    "first_frame_policy": "require_asset",
    "spatial": {
      "scene_profiles": {
        "pendulum": {"width": 480, "height": 832}
      }
    },
    "temporal": {
      "fps": 24,
      "num_frames": 121,
      "valid_frame_rule": "4n+1"
    }
  },
  "runner": {
    "type": "my_model_v1",
    "config": {"num_inference_steps": 50}
  }
}
```

`StandardDataAdapter` 统一完成：

- scene capability 与任务类型检查；
- generic/physics prompt 解析和物理字段使用审计；
- generic 分支不读取 `case.physics`；
- I2V 首帧来源、空间 profile 与时间规格；
- adaptation fingerprint、媒体 materialization fingerprint；
- canonical jobs、operation DAG、cache binding 和 TaskInstance seal。

因此 driver 不再重复 Dataset、Task、prompt 和 identity 逻辑。普通 driver 只需继承
`DirectManagedDriver`，实现：

```python
class Driver(DirectManagedDriver):
    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        ...

    def execute_job(self, spec, *, log_path):
        ...
```

`prepare_job` 返回的 `output_video` 必须在
`<run_dir>/predictions/` 内。公共 runtime 负责写 job spec、prediction 公共字段、
GT/reference 绑定、planned/staged/complete 状态和 run-local 路径检查。driver 返回值
不能覆盖 job ID、case ID、conditioning、seed、status 或 video path。

对于常见的命令行 I2V 模型，可直接复用
`StandardI2VCLIDriver`。其外部程序契约是：

```text
<command> --prompt TEXT --image PATH --output PATH --seed INT [extra_args...]
```

Bundle 的 `driver.py` 只需一行：

```python
from physbench.baseline_runtime.drivers.subprocess_i2v import StandardI2VCLIDriver as Driver
```

Cosmos 使用自有薄 driver，因为它需要多 GPU `torchrun`、Cosmos payload 和 checkpoint
identity；G15 的 Bundle driver 也只有一行，复用共享的 WAN managed driver。

### 2.2 Submission：只提交输出

submission Bundle 不包含 driver：

```text
baselines/my_submission/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json
└── README.md
```

portable manifest 声明 `implementation.kind=submission`、capabilities 和标准 adapter。
本机 `baseline.local.json` 指向 JSONL：

```json
{
  "runtime": {
    "submission_manifest": "/absolute/path/to/submission.jsonl"
  }
}
```

每条 submission 记录必须精确对应一个 canonical job：

```json
{
  "job_id": "...",
  "case_id": "...",
  "conditioning": "physics",
  "seed": 42,
  "video_path": "/external/model/output.mp4"
}
```

执行时要求覆盖完整且无额外 job，并逐项校验 case、conditioning 和 seed。视频通过
`import_prediction_video` 复制到当前 run；评估不会直接引用外部模型目录。缺失、额外、
重复或身份不匹配的记录会使 run 失败。

### 2.3 Command：高级接入

涉及 View A 微调、复杂常驻 worker、模型原生非标准输入或独立进程隔离时，可以继续
使用 v3 command：

```text
baselines/my_advanced_model/
├── baseline.json
├── plugin/main.py
└── ...
```

```json
{
  "schema_version": "3.0",
  "implementation": {
    "kind": "command",
    "protocol": "physbench-baseline-v1",
    "entrypoint": ["{python}", "plugin/main.py"],
    "fingerprint_paths": ["plugin/**/*.py"]
  }
}
```

endpoint 通过 request/response JSON 实现四个操作：

| operation | 作用 |
| --- | --- |
| `describe` | 描述 TaskBuilder/DataAdapter 和依赖指纹 |
| `adapt_case` | 构建可审计模型输入 |
| `build_task_instance` | 编译并封印 canonical plan |
| `run_task` | 执行训练和推理 |

WAN `wan22_ti2v_5b_lora_r32_v3` 保留此路径，因为它同时支持
`finetune_eval`。command host 会复验 canonical plan、identity 和 seal，不能因为接口
高级而绕过 Benchmark 不变量。

## 3. Bundle 与部署身份

两种 digest 分工不同：

```text
bundle digest
= canonical(portable manifest + Bundle-local fingerprinted files)

deployment digest
= canonical(应用 baseline.local.json 后的 manifest)
```

managed driver 总是自动进入 bundle digest，即使没有写进 `fingerprint_paths`。外部共享
runtime、prompt profile 和执行脚本进入 TaskBuilder 的
`runtime_dependency_fingerprints`。本机 checkpoint、Python、模型根目录与 GPU 配置
进入 deployment digest。

`baseline.local.json` 只能覆盖 `model` 和 `runtime`，不能改变 ID、版本、capability、
adapter 或 implementation。checkpoint 还应通过 revision、identity file 或完整
SHA-256 验证；仅记录路径不构成模型身份。

## 4. BaselineTaskInstance

所有三档接口最终产生同一种 schema `2.1` envelope：

```text
task_instance
├── instance_id / instance_digest
├── identity
│   ├── dataset / task
│   ├── baseline: id, version, bundle digest, deployment digest
│   ├── task_builder / data_adapter fingerprints
│   └── canonical_plan_digest
├── semantics
├── canonical_plan
├── source.cases / asset_root
├── adaptations
├── training
├── inference.jobs
├── execution_graph
├── cache_bindings
└── baseline_payload
```

实例是 canonical JSON seal。修改 job、prompt、checkpoint、DAG、identity 或媒体绑定都会
使 `instance_digest` 失效。执行前还会验证实例对应当前部署。

## 5. 新增和移除

完整的逐字段配置、driver 示例、目录树和验收清单见
[自定义 Baseline 集成指南](BASELINE_INTEGRATION.md)。

创建 managed I2V 模板：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_i2v --backend managed-i2v
```

创建 output-only 模板：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_outputs --backend submission
```

生成的目录会立即走正式 Registry 校验。接入时：

1. 修改 portable model identity、capabilities、adapter 和 runner；
2. 把本机路径写入 `baseline.local.json`；
3. managed 模型实现或选择 driver；submission 准备完整 JSONL；
4. 执行 `baseline validate`；
5. 用一个 explicit case 做 dry-run；
6. 验证 prediction、job、payload 和日志均在 AtomicRun；
7. 审计预训练数据与 Dataset 的 source overlap；
8. 增加模型专有 payload、checkpoint 和失败状态测试。

移除时删除 `baselines/<name>/` 即可；无需修改 Registry、CLI 或任务配置。历史 AtomicRun
已冻结 Bundle、deployment 和 TaskInstance，不依赖目录继续存在即可重新查看和评估。

## 6. 管理命令

```bash
# 仅扫描 manifest，不 import driver 或执行 endpoint
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list

# 解析本机覆盖，检查实现、能力和组件身份
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect cosmos3_nano_i2v

# 验证部署和 TaskBuilder/DataAdapter 指纹
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20
```

`--baseline` 接受 baseline ID、Bundle 目录或 `baseline.json` 路径。重复 ID 会立即报错。

## 7. 当前 Bundle

| baseline ID | kind | family | 说明 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3` | command v3 | `finetune_eval`, `direct_eval` | View A LoRA |
| `cosmos3_nano_i2v` | managed v4 | `direct_eval` | Cosmos base |
| `wan22_g15_sparse_motion_r32_e20` | managed v4 | `direct_eval` | 冻结 G15；诊断型 |

Cosmos 与 G15 收到 `finetune_eval` 会在 run 创建前失败。G15 的源数据重叠审计是其
可比性约束，不因集成方式改变。
