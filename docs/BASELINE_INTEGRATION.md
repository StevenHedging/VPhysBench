# 自定义 Baseline 集成指南

本文面向新增模型或算法。当前新实验只使用 schema 5.0 Baseline；Task 不再为物理信息
使用方式复制实验臂。

当前Dataset 11.0.0在每条Case中同时提供内联`case.physics`和锁定的
`assets.physics_annotation`（`physics.v11.json`）。Loader已保证两者完全一致；Baseline仍应使用
`case.physics[annotated=true]`这一稳定运行时API，不应绕过Loader直接解释文件格式。
每个独立quantity都保留非负`value`、`unit`和稳定`symbol`；方向来自Case prompt，不能
再从数值正负号推断。`annotated=false`字段只供Evaluator和审计使用。

## 1. 先确定 Baseline identity

一个 Baseline identity 应固定：

- 模型与 checkpoint；
- T2V/I2V/V2V/hybrid 输入范式；
- `case.text.prompt` 的使用方式；
- 是否使用结构化物理信息；
- 物理 representation 与 adapter；
- trainer、runner 和生成超参数。

如果同一底模要比较“不使用物理”和“追加物理文本”，应注册两个 Baseline：

```text
my_model_generic
my_model_physics
```

两者运行同一 Task。不要创建两份 Task，也不要用运行时 flag 在同一个 Baseline ID 下
切换语义。

## 2. 选择实现类型

schema 5.0 支持两种 implementation：

| kind | 适用场景 |
| --- | --- |
| `managed` | Benchmark 调用 Bundle driver 执行训练/推理 |
| `submission` | 已有完整外部预测，Benchmark 负责身份校验与复制归档 |

Managed 又有两种 adapter：

| adapter kind | 适用场景 |
| --- | --- |
| `standard` | 文本 + 常规 T2V/I2V/V2V；可选结构化物理文本追加 |
| `python` | token、trajectory、mask、flow、控制视频或其它模型原生表示 |

历史冻结 run 中可能仍含已废弃的 Baseline metadata；当前 Registry 不加载它们。
Legacy evaluator 只消费已有产物做兼容性重评，不能编译新 Task 或启动旧执行面。

## 3. 创建脚手架

```bash
# 标准 I2V
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_i2v --backend managed-i2v

# 标准 V2V 协议模板
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_v2v --backend managed-v2v

# 已有输出
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_outputs --backend submission
```

生成目录：

```text
baselines/my_i2v/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json          # 本机创建，Git ignored
├── driver.py
├── README.md
└── .gitignore
```

Registry 自动扫描：

```text
baselines/*/baseline.json
baselines/*/*.baseline.json
```

同一目录可放共享 driver 的多个 manifest，例如 `baseline.json` 与
`physics.baseline.json`。它们共享一个 `baseline.local.json`，适合底模与部署完全相同
的输入策略变体；若 checkpoint 或 runtime 不同，应使用不同目录。

## 4. Manifest 最小结构

下面是只支持 pendulum direct-eval 的 I2V 示例：

```json
{
  "schema_version": "5.0",
  "baseline_id": "my_i2v_generic",
  "baseline_version": "1.0.0",
  "description": "My I2V model using the Case prompt as-is.",
  "implementation": {
    "kind": "managed",
    "driver": "driver.py",
    "fingerprint_paths": []
  },
  "supported_scenes": ["pendulum"],
  "capabilities": {
    "task_families": ["direct_eval"],
    "generation_modes": ["i2v"],
    "train": false,
    "finetune": false,
    "generate": true
  },
  "input_policy": {
    "schema_version": "1.0",
    "case_view": "conditionable_case_v1",
    "text": {
      "source": "case.text.prompt",
      "usage": "required"
    },
    "physics": {
      "source": "case.physics[annotated=true]",
      "usage": "ignored",
      "representations": []
    }
  },
  "model": {
    "model_id": "org/model",
    "checkpoint": null
  },
  "runtime": {
    "python": "python",
    "project_root": "."
  },
  "adapter": {
    "kind": "standard",
    "preset": "standard_i2v_v1",
    "first_frame_policy": "require_asset",
    "physics_transform": {"type": "none"},
    "spatial": {
      "scene_profiles": {
        "pendulum": {"width": 832, "height": 480}
      }
    },
    "temporal": {
      "fps": 24,
      "num_frames": 121,
      "valid_frame_rule": "4n+1"
    }
  },
  "runner": {
    "type": "my_i2v_v1",
    "config": {}
  }
}
```

要注册结构化文本物理版本，复制为 `physics.baseline.json` 并至少修改：

```json
{
  "baseline_id": "my_i2v_physics",
  "input_policy": {
    "schema_version": "1.0",
    "case_view": "conditionable_case_v1",
    "text": {
      "source": "case.text.prompt",
      "usage": "required"
    },
    "physics": {
      "source": "case.physics[annotated=true]",
      "usage": "required",
      "representations": ["structured_text"]
    }
  },
  "adapter": {
    "kind": "standard",
    "preset": "standard_i2v_v1",
    "first_frame_policy": "require_asset",
    "physics_transform": {
      "type": "append_structured_text_v1",
      "template_set": "six_scene_physics_clauses_v2"
    }
  }
}
```

实际 manifest 仍需保留 generic 版本中的 spatial、temporal、model、runtime、runner
等完整字段；上段仅展示差异。

## 5. `input_policy` 是不可变模型契约

所有 Baseline 都要求：

```text
text.source = case.text.prompt
text.usage  = required
```

物理策略：

| usage | 要求 |
| --- | --- |
| `ignored` | `representations=[]`，不得产生 physics channel |
| `optional` | representation 非空；按 case 使用时登记 channel 与参数 |
| `required` | representation 非空；每条 adaptation 必须使用 annotated 参数 |

对 standard adapter：

- `ignored` 配 `physics_transform.type=none`；
- `optional|required` 当前配
  `append_structured_text_v1 + representations=["structured_text"]`。

其它注入方式使用 Python adapter。Manifest 声明的是实际可审计行为，不应夸大模型
能力，也不应由 Task 或 local override 改写。

## 6. Standard adapter

### I2V

`preset=standard_i2v_v1` 使用：

```text
case.text.prompt
assets.first_frame
scene spatial profile
temporal recipe
optional Baseline-owned physics transform
```

`first_frame_policy=require_asset` 是当前官方路径。缺失首帧应修复 Dataset，不得从 GT
临时提取。

### T2V

`preset=standard_t2v_v1` 不声明媒体 channel，但仍要求非空文本。Driver 不应读取
Case 资产。

### V2V

`preset=standard_v2v_v1` 还要声明：

```json
{
  "video_asset_key": "input_video"
}
```

Adapter audit 中的 `conditioning_video` 表示 V2V 输入媒体角色，并非 Task 的物理信息
分组。该资产必须独立于 GT/reference/source。当前 Dataset 11.0.0 没有正式
`assets.input_video`，所以 V2V 脚手架不能直接运行官方 Task。

## 7. Python adapter

Manifest：

```json
{
  "adapter": {
    "kind": "python",
    "entrypoint": "adapter.py",
    "config": {},
    "cache_policy": "content_addressed_immutable"
  }
}
```

Entrypoint：

```python
from physbench.baseline_api.interfaces import DataAdapter


class MyAdapter(DataAdapter):
    ...


def create_adapter(bundle):
    return MyAdapter(bundle)
```

Adapter 必须：

- 实现 `adapt_case(case, *, role)`；
- 输出 JSON-serializable `native_inputs` 和 `input_contract`；
- 精确登记 `used_parameters`；
- 只使用`annotated=true`独立量并保持值、单位和symbol一致；
- 给出完整与 media-materialization 两种 SHA-256 fingerprint；
- 通过 `dependency_paths()` 登记 Bundle 外的输出相关实现。

大型 trajectory/mask/flow/control 不得内嵌进 TaskInstance；使用带 producer/source
provenance 的内容寻址 artifact URI。完整契约见
[DataAdapter 与输入策略](DATA_ADAPTER.md)。

## 8. Managed driver

`implementation.driver` 指向 Bundle-local Python 文件，并导出 `Driver`。

普通 direct-eval 模型继承 `DirectManagedDriver`：

```python
from physbench.baseline_runtime import DirectManagedDriver


class Driver(DirectManagedDriver):
    def validate_deployment(self):
        ...

    def dependency_paths(self):
        return {}

    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        output = run_dir / "predictions" / f"{job['job_id']}.mp4"
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "seed": int(job["seed"]),
            "prompt": job["native_inputs"]["text"]["prompt"],
            "output_video": str(output.resolve())
        }

    def execute_job(self, spec, *, log_path):
        ...
        return {"return_code": 0, "log_path": str(log_path)}
```

职责：

- `validate_deployment()` 验证 Python、checkpoint 与 identity，不加载大模型；
- `dependency_paths()` 登记 Bundle 外的 inference/preprocess 等关键代码；
- `prepare_job()` 只构造 payload，原样保留 canonical job ID 和 seed；
- `execute_job()` 执行一个 job；
- 所有预测必须位于当前 `run_dir/predictions/`。

公共 runtime 只在 `return_code == 0` 且文件存在时标记 complete。Driver 不能返回或
覆盖 reference、canonical identity、status、video path 等受保护字段。

高成本模型可覆盖 `execute_jobs()` 实现常驻 worker。需要训练时实现完整
`ManagedDriver.run_task()`；WAN 是当前参考实现。

常见 CLI driver 可直接复用：

```python
from physbench.baseline_runtime.drivers.subprocess_i2v import (
    StandardI2VCLIDriver as Driver,
)
```

其外部程序接收 `--prompt --image --output --seed`，新脚手架还通过
`runner.config.job_spec_arg` 传 `--job-spec`。V2V 版本使用 `--video`。

## 9. Submission

Submission Bundle 仍需要 input policy 和 adapter，因为 Benchmark 必须先编译同一
TaskInstance。`baseline.local.json` 指向外部 JSONL：

```json
{
  "runtime": {
    "submission_manifest": "/absolute/path/to/submission.jsonl"
  }
}
```

每行精确对应一个 canonical job：

```json
{
  "job_id": "five_scene_direct_eval_v5__case_id__seed000042",
  "case_id": "case_id",
  "seed": 42,
  "video_path": "/external/output/case_id.mp4"
}
```

不包含 Task 层的物理开关字段。Benchmark 要求：

- job 完整覆盖且无额外/重复记录；
- case 与 seed 完全一致；
- 视频存在；
- submission manifest 加载后不变。

执行时视频会原子复制到当前 run，并记录源/目标 SHA-256；正式
`predictions.jsonl` 不直接引用外部模型目录。

## 10. Portable 配置与本机配置

提交到 Git 的 manifest 保存稳定、可迁移语义。本机路径只写入 Git-ignored
`baseline.local.json`：

```json
{
  "model": {
    "checkpoint": "/absolute/path/to/checkpoint"
  },
  "runtime": {
    "python": "/absolute/path/to/python",
    "project_root": "/absolute/path/to/model/repo"
  }
}
```

Local override 只能覆盖 `model` 与 `runtime`，不能修改：

```text
baseline_id / version
implementation / capabilities
input_policy / adapter
runner / trainer
```

身份分层：

```text
bundle digest
= portable manifest + Bundle-local fingerprinted files

deployment digest
= 应用 local override 后的 manifest

TaskBuilder fingerprint
= bundle/deployment + adapter + runner/trainer + runtime dependencies
```

schema v5 会自动纳入 Bundle 内所有 Python 文件；其它资源用
`implementation.fingerprint_paths`。Checkpoint 应另有 revision、identity file 或
完整 SHA-256，仅记录路径不够。

## 11. 验证流程

### 发现

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list
```

### 检查解析结果

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect my_i2v_generic
```

重点检查 `input_policy`、adapter、bundle/deployment digest、model/runtime 与
TaskBuilder/DataAdapter fingerprint。

### 验证部署

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate my_i2v_generic
```

### 编译 TaskInstance

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline my_i2v_generic \
  --output /tmp/my_i2v_generic_task.json
```

### 单 case dry-run

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline my_i2v_generic \
  --scene-id pendulum \
  --case-id CASE_ID \
  --run-id my_i2v_generic_dryrun \
  --output-root runs_v2
```

检查：

- prompt 是否源自 `case.text.prompt`；
- ignored Baseline 的 `used_parameters` 与 physics channel 是否为空；
- required Baseline 是否只使用预期字段和值/单位；
- 首帧、shape、FPS、帧数与 seed；
- checkpoint、adapter 与 TaskInstance identity；
- prediction/job/log path 是否 run-local；
- Dataset 是否保持未修改。

用新的 run ID 加 `--execute` 做单 case 推理。部分 run 只用于工程 smoke，不是正式
Task score。

## 12. 推荐测试

至少覆盖：

```text
bundle discovery / schema validation
deployment identity
input_policy 与 adapter 一致
ignored physics produces no channel
required physics uses annotated parameters
canonical plan unchanged across Baselines
materialization fingerprint shared when media recipe identical
TaskInstance deterministic and sealed
unsupported family rejected
output is run-local
missing video cannot be complete
Dataset remains immutable
```

关键不变量：

```text
相同 Dataset + Task + Baseline deployment
→ 相同 TaskInstance digest
```

```text
不同 Baseline 执行同一 Task
→ 相同 canonical plan
```

```text
修改物理使用方式
→ Baseline/DataAdapter/TaskInstance identity 改变，Task 不变
```

## 13. 当前参考实现

```text
baselines/cosmos3_nano_i2v/
├── baseline.json
├── physics.baseline.json
└── driver.py

baselines/wan22_lora/
├── baseline.json
├── physics.baseline.json
└── driver.py

baselines/wan22_g15_sparse_motion/
├── baseline.json
├── physics.baseline.json
└── driver.py
```

三组均使用 schema 5.0 managed runtime。WAN 两组共享
`Wan22ManagedDriver`；Cosmos 使用薄模型专属 driver。完整运行命令见
[运行与故障排查](OPERATIONS.md)。
