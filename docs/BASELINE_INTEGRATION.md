# 自定义 Baseline 集成指南

本文面向需要把新视频生成模型接入 Physics Video Benchmark 的开发者，给出从目录创建、
manifest 配置、driver 编写，到 dry-run、正式执行和结果验收的完整流程。

命令默认从仓库根目录执行：

```text
/root/Steven/physics_video_benchmark
```

正式 Benchmark Python：

```text
/root/miniconda3/envs/phybench/bin/python
```

## 1. 先选择接入方式

Benchmark 提供三档接口：

| kind | schema | 适用情况 | 需要实现 |
| --- | --- | --- | --- |
| `submission` | `4.0` | 模型已在外部生成完整视频，只需要统一评估 | manifest + submission JSONL |
| `managed` | `4.0` | 标准 T2V/I2V direct evaluation | manifest + 薄 driver |
| `command` | `3.0` | 微调、训练、复杂多进程或特殊模型原生输入 | 完整 command endpoint |

选择顺序：

```text
已经有完整预测视频？
├── 是：submission
└── 否
    ├── 标准 T2V/I2V 推理：managed
    └── 训练、微调或复杂控制流：command
```

推荐原则：

- 对外部结果和闭源模型，优先使用 `submission`。
- 对仓库维护的普通开源推理 Baseline，优先使用 `managed`。
- 只有 managed 无法自然表达时才使用 `command`。

三档接口都会生成同一种 sealed `BaselineTaskInstance`，并进入同一套 scene evaluator。
具体模型名不会写入核心 Registry，因此增加或移除 Baseline 不需要修改
`src/physbench/baseline_api/registry.py`。

## 2. 接入边界

Benchmark 负责：

```text
Dataset + Task
→ canonical train/evaluation case plan
→ canonical job_id / case_id / partition / seed
→ TaskInstance identity and seal
→ run-local artifact validation
→ scene-local evaluation
→ strict Task aggregation
```

Baseline 负责：

```text
capabilities
model/checkpoint identity
Case-to-model input adaptation
model-native payload
model execution
prediction video
output-affecting dependency fingerprints
```

Baseline 不得：

- 重新选择或跳过 canonical jobs；
- 修改 case、partition、conditioning 或 seed；
- 回写 `datasets/`；
- 在 generic 分支读取结构化 physics；
- 在模型工程目录保存正式 prediction；
- 自行替换 Benchmark evaluator；
- 用外部视频路径或软链接绕过 run-local 归档。

## 3. Managed：推荐的标准接入

### 3.1 目录树

推荐目录：

```text
baselines/
└── my_custom_i2v/
    ├── baseline.json                  # portable contract
    ├── driver.py                      # 模型专有边界
    ├── baseline.local.example.json    # 本机配置模板，提交
    ├── baseline.local.json            # 本机真实路径，不提交
    ├── .gitignore
    ├── README.md
    ├── inference.py                   # 可选：Bundle-local CLI wrapper
    └── provenance/                    # 可选
        ├── model_revision.json
        ├── training_data_audit.json
        └── benchmark_overlap.json
```

最小必需内容：

```text
my_custom_i2v/
├── baseline.json
└── driver.py
```

建议始终补齐 local template、`.gitignore`、README 和训练来源审计。

### 3.2 创建模板

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_custom_i2v --backend managed-i2v
```

命令会创建：

```text
baselines/my_custom_i2v/
├── baseline.json
├── baseline.local.example.json
├── driver.py
├── README.md
└── .gitignore
```

生成后会立即走正式 Registry 校验。命令不会创建真实
`baseline.local.json`，需要开发者从模板复制：

```bash
cp \
  baselines/my_custom_i2v/baseline.local.example.json \
  baselines/my_custom_i2v/baseline.local.json
```

当前脚手架直接支持 `managed-i2v`。纯 T2V 模型可以先生成该模板，再按本文 T2V
小节修改 preset 和 driver。

### 3.3 `baseline.json` 完整示例

```json
{
  "schema_version": "4.0",
  "baseline_id": "my_custom_i2v",
  "baseline_version": "1.0.0",
  "description": "My custom image-to-video baseline.",
  "implementation": {
    "kind": "managed",
    "driver": "driver.py",
    "fingerprint_paths": []
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
    "conditioning": ["generic", "physics"],
    "train": false,
    "finetune": false,
    "generate": true
  },
  "model": {
    "model_id": "vendor/my-video-model",
    "variant": "base",
    "checkpoint": null,
    "revision": "frozen-model-revision"
  },
  "runtime": {
    "python": "python",
    "cuda_visible_devices": "0"
  },
  "adapter": {
    "preset": "standard_i2v_v1",
    "profile_set": "five_scene_i2v_v1",
    "first_frame_policy": "require_asset",
    "spatial": {
      "scene_profiles": {
        "pendulum": {
          "width": 480,
          "height": 832
        },
        "free_fall": {
          "width": 480,
          "height": 832
        },
        "collision_1d": {
          "width": 832,
          "height": 480
        },
        "inclined_plane_slide": {
          "width": 832,
          "height": 480
        },
        "uniform_circular_motion": {
          "width": 480,
          "height": 832
        }
      }
    },
    "temporal": {
      "fps": 24,
      "num_frames": 121,
      "valid_frame_rule": "4n+1"
    }
  },
  "runner": {
    "type": "my_custom_i2v_v1",
    "config": {
      "num_inference_steps": 50,
      "guidance_scale": 5.0,
      "negative_prompt": "camera shake, flicker, blur, physically implausible motion"
    }
  }
}
```

### 3.4 Manifest 字段

#### `baseline_id`

ID 必须全仓库唯一，只能包含字母、数字、点、下划线和连字符。建议让 ID 表达模型族和
训练身份：

```text
cogvideox_5b_i2v
my_model_base_v1
wan22_custom_lora_r32_e10
```

避免使用 `test`、`latest` 或 `baseline1` 等不稳定名称。

#### `baseline_version`

当 checkpoint、prompt recipe、推理参数、输入适配、FPS、帧数、训练方法或 scene
profile 发生语义变化时，应提升版本。本机路径变化只影响 deployment digest，不要求
提升 portable version。

#### `implementation`

Managed 固定使用：

```json
{
  "kind": "managed",
  "driver": "driver.py",
  "fingerprint_paths": []
}
```

driver 必须是 Bundle 内相对路径，不能使用绝对路径或 `../`。driver 会自动进入
portable bundle digest，不需要重复写入 `fingerprint_paths`。

其他影响输出的 Bundle-local 文件应显式登记：

```json
{
  "fingerprint_paths": [
    "inference.py",
    "configs/*.json",
    "provenance/*.json"
  ]
}
```

每个 glob 必须匹配至少一个文件。没有额外文件时使用空列表。

#### `supported_scenes`

当前正式 scene：

```text
pendulum
free_fall
collision_1d
inclined_plane_slide
uniform_circular_motion
```

只声明模型实际支持的 scene。即使使用 `"all"`，标准 adapter 仍需要为实际请求的每个
scene 提供 spatial profile。

#### `capabilities`

普通 `DirectManagedDriver` 使用：

```json
{
  "task_families": ["direct_eval"],
  "conditioning": ["generic", "physics"],
  "train": false,
  "finetune": false,
  "generate": true
}
```

不要为普通 direct driver 声明 `finetune_eval`。如果模型只支持 generic prompt，可以
只声明 `["generic"]`。

#### `model`

portable manifest 应记录可迁移模型身份：

- model repository ID；
- base/SFT/LoRA variant；
- Hugging Face revision 或 Git commit；
- checkpoint step/epoch；
- LoRA rank/target modules；
- checkpoint SHA-256 或轻量 identity files；
- 训练数据来源与 benchmark overlap 分类。

本机 checkpoint 绝对路径由 local override 提供。

#### `runtime`

runtime 是 driver 私有配置，可以包含：

```text
python
project_root
framework_root
model_base
torchrun
cuda_visible_devices
hf_home
cache_root
offline
compile
parallelism_preset
```

Benchmark 不解释这些字段。机器相关绝对路径放进 `baseline.local.json`。

#### `runner`

runner 保存影响生成结果的模型配置，例如：

- inference steps；
- guidance/CFG；
- sampler 和 scheduler；
- negative prompt；
- LoRA alpha；
- motion strength；
- conditioning strength；
- tiling；
- quality；
- audio 开关。

driver 通过以下方式读取：

```python
config = self.bundle.value["runner"]["config"]
```

不要依赖模型工程中未冻结的隐藏默认值。

### 3.5 Adapter 配置

Managed runtime 当前提供：

```text
standard_i2v_v1
standard_t2v_v1
```

它统一负责：

- family、conditioning 和 scene capability 校验；
- generic/physics prompt；
- physics 使用字段审计；
- first-frame 来源；
- scene spatial profile；
- temporal profile；
- adaptation fingerprint；
- materialization fingerprint；
- canonical jobs、operation DAG、cache binding 和 TaskInstance seal。

#### I2V

```json
{
  "preset": "standard_i2v_v1",
  "profile_set": "five_scene_i2v_v1",
  "first_frame_policy": "require_asset"
}
```

支持的首帧策略：

```text
require_asset
asset_or_reference_frame0
```

`require_asset` 要求 `case.assets.first_frame` 存在。
`asset_or_reference_frame0` 允许在缺失首帧时从 physics reference 第 0 帧确定性提取。

`StandardI2VCLIDriver` 当前只直接支持真实 `first_frame_asset`。如果使用 fallback
策略，应在自定义 driver 中实现 frame-0 提取，并把派生图像写入 run 或内容寻址 cache。

#### T2V

```json
{
  "preset": "standard_t2v_v1",
  "profile_set": "five_scene_i2v_v1"
}
```

T2V 不声明 `first_frame_policy`，driver 也不读取 `native_inputs.vision` 中的首帧。

#### Spatial profile

宽高型模型：

```json
{
  "pendulum": {
    "width": 480,
    "height": 832
  },
  "collision_1d": {
    "width": 832,
    "height": 480
  }
}
```

模型原生 token 也可以直接声明，例如 Cosmos：

```json
{
  "pendulum": {
    "resolution": 480,
    "aspect_ratio": "9,16"
  },
  "collision_1d": {
    "resolution": 480,
    "aspect_ratio": "16,9"
  }
}
```

这些字段会进入 `native_inputs.generation_shape`，由 driver 解释。Benchmark 不要求所有
模型生成同一原始分辨率。

#### Temporal profile

固定长度：

```json
{
  "fps": 24,
  "num_frames": 121,
  "valid_frame_rule": "4n+1"
}
```

动态长度：

```json
{
  "fps": 24,
  "min_frames": 5,
  "max_frames": 121,
  "valid_frame_rule": "4n+1",
  "policy": "physical-time prefix"
}
```

当规则是 `4n+1` 时，所有帧数边界必须满足：

```text
(frame_count - 1) % 4 == 0
```

不要通过补 GT 首帧、复制末帧或拼接 GT 片段来满足模型帧数。

### 3.6 Generic/physics 隔离

标准 adapter 的条件边界：

```text
generic
├── scene/process description
├── no structured physics input
└── used_parameters = {}

physics
├── same scene/process description
├── whitelisted physical values and units
└── used_parameters audit
```

driver 应直接使用：

```python
prompt = job["native_inputs"]["text"]["prompt"]
```

不要在 driver 中重新读取 `case["physics"]` 并拼接 prompt，否则会破坏条件对照和
fingerprint 审计。

## 4. 本机配置

`baseline.local.example.json` 可以提交：

```json
{
  "model": {
    "checkpoint": "/absolute/path/to/checkpoint"
  },
  "runtime": {
    "python": "/absolute/path/to/model/python",
    "project_root": "/absolute/path/to/model/project",
    "cuda_visible_devices": "0,1"
  }
}
```

真实配置写入 Git-ignored `baseline.local.json`。local override 只允许覆盖：

```text
model
runtime
```

不能覆盖：

```text
baseline_id
baseline_version
implementation
supported_scenes
capabilities
adapter
runner
trainer
```

Benchmark 区分：

```text
bundle digest
= portable manifest + Bundle-local fingerprinted files

deployment digest
= 应用 baseline.local.json 后的 manifest
```

因此 driver、profile、checkpoint、Python 和 GPU 配置变化都会被身份系统捕获。

## 5. Driver 实现

### 5.1 最简单：标准 I2V CLI

如果推理程序支持：

```text
<command>
  --prompt TEXT
  --image PATH
  --output PATH
  --seed INT
```

`driver.py` 只需：

```python
from physbench.baseline_runtime.drivers.subprocess_i2v import (
    StandardI2VCLIDriver as Driver,
)
```

runner：

```json
{
  "type": "standard_i2v_cli_v1",
  "config": {
    "command": ["python", "inference.py"],
    "extra_args": [
      "--num-inference-steps",
      "50"
    ]
  }
}
```

适合：

- 模型有稳定 CLI；
- command 不依赖机器专有绝对路径；
- 不需要首帧预处理；
- 不需要多 GPU `torchrun`；
- 不需要模型常驻 worker。

如果需要本机模型 Python、外部 project root、复杂环境变量或 payload，使用自定义 driver。

### 5.2 自定义 `DirectManagedDriver`

需要实现：

```text
Driver(DirectManagedDriver)
├── validate_deployment()      # 可选
├── dependency_paths()         # 可选
├── prepare_job()              # 必需
├── execute_job()              # 必需
└── execute_jobs()             # 可选，多 GPU/常驻 worker
```

完整骨架：

```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from physbench.baseline_runtime import DirectManagedDriver
from physbench.io import sha256_file, write_json


class Driver(DirectManagedDriver):
    def validate_deployment(self) -> None:
        model = self.bundle.value["model"]
        runtime = self.bundle.value["runtime"]

        checkpoint = Path(model["checkpoint"])
        python = Path(runtime["python"])
        project_root = Path(runtime["project_root"])
        inference = project_root / "inference.py"

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint}"
            )
        if not python.is_file():
            raise FileNotFoundError(
                f"model Python not found: {python}"
            )
        if not inference.is_file():
            raise FileNotFoundError(
                f"inference script not found: {inference}"
            )

        expected = model.get("checkpoint_sha256")
        if expected and checkpoint.is_file():
            actual = sha256_file(checkpoint)
            if actual != expected:
                raise ValueError(
                    "checkpoint digest mismatch: "
                    f"expected={expected}, actual={actual}"
                )

    def dependency_paths(self) -> dict[str, Path]:
        project_root = Path(
            self.bundle.value["runtime"]["project_root"]
        )
        return {
            "external/my_model/inference.py": (
                project_root / "inference.py"
            )
        }

    def prepare_job(
        self,
        *,
        job: dict[str, Any],
        case: dict[str, Any],
        adaptation: dict[str, Any],
        source_root: Path,
        run_dir: Path,
    ) -> dict[str, Any]:
        native = job["native_inputs"]
        first_frame_asset = native["vision"].get(
            "first_frame_asset"
        )
        if not first_frame_asset:
            raise ValueError(
                f"job has no first-frame asset: {job['job_id']}"
            )

        first_frame = (
            source_root / first_frame_asset
        ).resolve()
        if not first_frame.is_file():
            raise FileNotFoundError(
                f"first frame not found: {first_frame}"
            )

        output_video = (
            run_dir
            / "predictions"
            / adaptation["conditioning"]
            / f"{job['job_id']}.mp4"
        ).resolve()
        payload_path = (
            run_dir
            / "jobs"
            / f"{job['job_id']}.payload.json"
        )

        runner = self.bundle.value["runner"]["config"]
        payload = {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "prompt": native["text"]["prompt"],
            "first_frame": str(first_frame),
            "generation_shape": native["generation_shape"],
            "seed": int(job["seed"]),
            "num_inference_steps": int(
                runner["num_inference_steps"]
            ),
            "guidance_scale": float(
                runner["guidance_scale"]
            ),
            "output_video": str(output_video),
        }
        write_json(payload_path, payload)

        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "seed": int(job["seed"]),
            "payload_path": str(payload_path),
            "output_video": str(output_video),
        }

    def execute_job(
        self,
        spec: dict[str, Any],
        *,
        log_path: Path,
    ) -> dict[str, Any]:
        runtime = self.bundle.value["runtime"]
        project_root = Path(runtime["project_root"])
        command = [
            str(runtime["python"]),
            str(project_root / "inference.py"),
            "--payload",
            spec["payload_path"],
        ]

        Path(spec["output_video"]).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        visible = runtime.get("cuda_visible_devices")
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(visible)

        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        return {
            "return_code": completed.returncode,
            "command": command,
            "log_path": str(log_path),
        }
```

### 5.3 Driver 各方法的职责

`validate_deployment()` 应检查 checkpoint、Python、关键脚本和模型 identity，但不应
加载完整模型、占用 GPU 或产生 prediction。

`dependency_paths()` 应登记 Bundle 外部但会影响输出的关键代码，例如 inference entry、
自定义 scheduler 或 preprocessing。不要登记日志、cache、prediction 和临时文件。

`prepare_job()` 应是轻量、确定性的，只构造 payload 和 run-local output path。它必须
原样返回 canonical `job_id`，不得修改 seed 或重新选择 prompt。

`execute_job()` 执行一个 job。公共 runtime 根据：

```text
return_code == 0 AND output_video exists
```

决定 prediction 是否 complete。

driver 返回值不能覆盖：

```text
job_id
case_id
baseline_id
conditioning
prompt_profile_id
evaluation_partition
status
video_path
seed
```

### 5.4 多 GPU 或常驻 worker

默认 lifecycle 会逐 job 调用 `execute_job()`。模型加载代价很高时，可以覆盖：

```python
def execute_jobs(
    self,
    specs,
    *,
    run_dir,
):
    ...
```

返回以 canonical job ID 为 key 的结果：

```python
{
    spec["job_id"]: {
        "return_code": 0,
        "worker_index": 0,
    }
    for spec in specs
}
```

需要训练或完全不同生命周期时，实现完整 `ManagedDriver.run_task()`，或直接使用
command 接口。不要把训练逻辑塞进 `DirectManagedDriver`。

### 5.5 T2V driver

T2V adapter：

```json
{
  "preset": "standard_t2v_v1",
  "profile_set": "five_scene_i2v_v1",
  "spatial": {
    "scene_profiles": {
      "pendulum": {
        "width": 832,
        "height": 480
      }
    }
  },
  "temporal": {
    "fps": 24,
    "num_frames": 121,
    "valid_frame_rule": "4n+1"
  }
}
```

T2V `prepare_job()` 不读取 first frame：

```python
def prepare_job(
    self,
    *,
    job,
    case,
    adaptation,
    source_root,
    run_dir,
):
    native = job["native_inputs"]
    output_video = (
        run_dir
        / "predictions"
        / adaptation["conditioning"]
        / f"{job['job_id']}.mp4"
    ).resolve()
    return {
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "seed": int(job["seed"]),
        "prompt": native["text"]["prompt"],
        "generation_shape": native["generation_shape"],
        "output_video": str(output_video),
    }
```

## 6. Submission：已有视频接入

### 6.1 目录树

```text
baselines/
└── my_submission/
    ├── baseline.json
    ├── baseline.local.example.json
    ├── baseline.local.json
    ├── README.md
    └── .gitignore
```

submission 没有 driver 或 command endpoint。

### 6.2 创建模板

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_submission --backend submission
```

portable manifest 使用：

```json
{
  "schema_version": "4.0",
  "implementation": {
    "kind": "submission",
    "fingerprint_paths": []
  },
  "capabilities": {
    "task_families": ["direct_eval"],
    "conditioning": ["generic", "physics"]
  }
}
```

submission 仍需声明标准 adapter，使 TaskInstance、prompt、shape 和 canonical jobs 可审计。
submission 当前只支持 `direct_eval`。

### 6.3 获取 canonical jobs

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline my_submission \
  --output /tmp/my_submission_task.json
```

外部模型应严格按照：

```text
/tmp/my_submission_task.json
└── inference.jobs[]
    ├── job_id
    ├── case_id
    ├── scene_id
    ├── evaluation_partition
    ├── conditioning
    └── seed
```

生成视频。

### 6.4 Submission JSONL

每行对应一个 canonical job：

```json
{
  "job_id": "five_scene_direct_eval_physics_v3__case_id__physics__seed000042",
  "case_id": "case_id",
  "conditioning": "physics",
  "seed": 42,
  "video_path": "/external/model/results/video.mp4"
}
```

要求：

- 每个 canonical job 恰好出现一次；
- 不允许缺失、额外或重复 job；
- case、conditioning 和 seed 必须一致；
- 视频必须存在；
- 后缀必须是 `.mp4`、`.mov`、`.mkv` 或 `.webm`；
- submission manifest 在 deployment 加载后不能变化。

`baseline.local.json`：

```json
{
  "runtime": {
    "submission_manifest": "/absolute/path/to/submission.jsonl"
  }
}
```

执行：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline my_submission \
  --output-root runs_v2 \
  --execute
```

外部视频会原子复制到当前 run，并记录源/目标 SHA-256 和 import provenance。正式
`predictions.jsonl` 不会直接引用外部模型目录。

## 7. Command：高级接入

Command 适用于 View A fine-tuning、训练 + 推理、多阶段 pipeline、复杂非标准输入或
独立进程隔离。

目录：

```text
baselines/
└── my_advanced_baseline/
    ├── baseline.json
    ├── baseline.local.example.json
    ├── baseline.local.json
    ├── README.md
    ├── .gitignore
    └── plugin/
        ├── main.py
        ├── implementation.py
        └── resources/
            ├── generic.json
            └── physics.json
```

manifest：

```json
{
  "schema_version": "3.0",
  "baseline_id": "my_advanced_baseline",
  "baseline_version": "1.0.0",
  "implementation": {
    "kind": "command",
    "protocol": "physbench-baseline-v1",
    "entrypoint": ["{python}", "plugin/main.py"],
    "fingerprint_paths": [
      "plugin/*.py",
      "plugin/resources/*.json"
    ]
  },
  "capabilities": {
    "task_families": ["finetune_eval", "direct_eval"],
    "conditioning": ["generic", "physics"]
  },
  "model": {},
  "runtime": {},
  "components": {}
}
```

endpoint 实现：

```text
physbench-baseline-v1
├── describe
├── adapt_case
├── build_task_instance
└── run_task
```

`plugin/main.py` 通常只做分发：

```python
from physbench.baseline_api.endpoint import main
from implementation import MyBaselinePlugin


if __name__ == "__main__":
    raise SystemExit(main(MyBaselinePlugin))
```

具体实现需要提供 `DataAdapter`、`TaskBuilder` 和 `BaselinePlugin`。可参考：

```text
baselines/wan22_lora/
src/physbench/baseline_plugins/wan22.py
```

Command endpoint 由 `phybench` Python 启动。模型实际训练/推理应由 executor 使用
`runtime.python` 启动模型环境，避免把模型专有依赖导入 Benchmark 进程。

## 8. 验证流程

### 8.1 发现

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list
```

`baseline list` 只扫描 manifest，不 import driver 或执行 endpoint。

### 8.2 检查解析结果

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect my_custom_i2v
```

重点检查：

```text
bundle_digest
deployment_digest
implementation
capabilities
supported_scenes
model
runtime
adapter_recipe
runner
task_builder
data_adapter
```

### 8.3 验证部署

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate my_custom_i2v
```

Managed validation 会调用 `validate_deployment()` 和 `dependency_paths()`，但不应加载完整
GPU 模型。

### 8.4 构建 TaskInstance

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline my_custom_i2v \
  --output /tmp/my_custom_i2v_task.json
```

检查：

```text
identity
canonical_plan
adaptations
inference.jobs
execution_graph
cache_bindings
baseline_payload
instance_digest
```

generic adaptation 应满足：

```text
used_parameters == {}
stages.physics.enabled == false
```

### 8.5 单 case dry-run

首次接入不要直接执行全量任务：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline my_custom_i2v \
  --scene-id pendulum \
  --case-id <pendulum_case_id> \
  --output-root runs_v2 \
  --run-id my_custom_i2v_dry_run
```

不加 `--execute` 时仍会完成：

```text
deployment validation
canonical plan
adaptation
prepare_job
job/payload materialization
planned prediction
evaluation state
```

但不会启动模型推理。

重点检查：

- prompt 与 conditioning；
- physics 字段使用审计；
- first frame；
- generation shape；
- FPS/帧数；
- seed；
- checkpoint identity；
- output path；
- job spec 和 payload；
- Dataset 未被修改。

### 8.6 单 case execute

使用新 run ID：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline my_custom_i2v \
  --scene-id pendulum \
  --case-id <pendulum_case_id> \
  --output-root runs_v2 \
  --run-id my_custom_i2v_execute_smoke \
  --execute
```

AtomicRun 不允许覆盖已存在目录，因此 dry-run 和 execute 必须使用不同 run ID。

单 case 只用于工程 smoke，不是完整正式 Task score。正式结果必须覆盖完整 canonical
plan。

## 9. AtomicRun 输出

```text
runs_v2/<run_id>/
├── run.json
├── state.json
├── plan.json
├── report.md
├── frozen/
│   ├── dataset.json
│   ├── cases.jsonl
│   ├── task.json
│   ├── baseline.json
│   ├── views.json
│   └── assets.lock.json
├── task_instance/
│   ├── manifest.json
│   ├── canonical_plan.json
│   ├── adaptations.jsonl
│   ├── inference_jobs.jsonl
│   ├── execution_graph.json
│   ├── cache_bindings.json
│   └── baseline_payload.json
├── jobs/
│   ├── <job_id>.json
│   └── <job_id>.payload.json
├── predictions/
│   └── <conditioning>/
│       └── <job_id>.mp4
├── predictions.jsonl
├── logs/
│   └── <baseline_id>/
│       └── <job_id>.log
├── artifacts/
│   └── prediction_artifacts.json
└── evaluation/
    ├── manifest.json
    ├── case_results.jsonl
    ├── task_result.json
    └── cases/<job_id>/
```

模型代码、权重和可重建 cache 可以在 Bench 外部。prediction、job payload、日志、训练
曲线、import provenance 和 evaluation 必须在当前 run 内。

正确输出：

```text
runs_v2/<run_id>/predictions/<conditioning>/<job_id>.mp4
```

错误输出：

```text
/root/MyModel/outputs/video.mp4
/tmp/video.mp4
```

每个非空 prediction path 会经过 run-local 检查、文件存在检查、大小记录和 SHA-256
计算。`status=complete` 但视频不存在会使 run 失败。

## 10. 推荐测试

为新 Baseline 增加：

```text
tests/
└── test_my_custom_baseline.py
    ├── test_bundle_is_discovered
    ├── test_manifest_is_valid
    ├── test_checkpoint_identity
    ├── test_generic_has_no_physics_leak
    ├── test_physics_uses_expected_parameters
    ├── test_scene_profiles
    ├── test_temporal_shape
    ├── test_canonical_plan_is_unchanged
    ├── test_task_instance_is_deterministic
    ├── test_output_is_run_local
    ├── test_direct_only_rejects_finetune
    ├── test_dry_run_payload
    └── test_dataset_is_not_modified
```

关键不变量：

```text
相同 Dataset + Task + Baseline deployment
→ 相同 TaskInstance digest
```

```text
改变 case.physics
→ generic native_inputs 不变
```

```text
不同 Baseline 执行同一 Task
→ canonical_plan 相同
```

```text
return_code == 0 但 output 不存在
→ prediction failed
```

执行完整测试：

```bash
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v
```

编译和差异检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  -m compileall -q src tests baselines

git diff --check
```

## 11. 常见错误

### `unknown baseline ID`

确认目录直接位于 `baselines/` 下，文件名为 `baseline.json`，且 ID 全局唯一。运行
`baseline list` 查看发现结果。

### `managed baseline driver not found`

`implementation.driver` 必须是 Bundle 内实际存在的相对路径，不能使用绝对路径或
`../`。

### `managed driver must export Driver subclass`

driver 模块必须导出名为 `Driver` 的 `ManagedDriver` 或 `DirectManagedDriver` 子类。

### `baseline fingerprint glob matched no files`

`fingerprint_paths` 中存在没有匹配任何文件的 glob。删除该 glob，或创建并提交对应
文件。

### `managed adapter has no spatial profile`

任务请求了某个 scene，但 `spatial.scene_profiles` 没有该 scene。

### `managed I2V case has no first-frame asset`

使用了 `require_asset`，但 case 没有首帧。检查 Dataset，或切换到
`asset_or_reference_frame0` 并实现确定性 frame-0 提取。

### `managed driver changed or omitted canonical job_id`

`prepare_job()` 必须原样返回输入的 `job["job_id"]`。

### `managed output must be inside run predictions`

driver 把 output 指向了模型工程目录或临时目录。使用 `run_dir / "predictions" / ...`
构造路径。

### `task instance targets a different managed Baseline deployment`

TaskInstance 构建后，manifest、driver、local override、checkpoint、Python、profile 或
外部依赖发生了变化。重新构建 TaskInstance。

### `submission coverage mismatch`

Submission JSONL 缺少 canonical job、混入另一 Task 的 job，或出现重复 job。应从当前
TaskInstance 的 `inference.jobs` 重新生成 submission 清单。

### 子进程返回 0，但 prediction 仍 failed

公共 runtime 同时要求 return code 为 0 且 output 文件存在。检查模型实际输出路径、
扩展名、异步落盘和视频封装过程。

## 12. 完成检查单

### Manifest

- [ ] `baseline_id` 唯一且有稳定语义。
- [ ] `baseline_version` 正确。
- [ ] `supported_scenes` 没有虚假声明。
- [ ] capabilities 与 driver 能力一致。
- [ ] portable manifest 不含本机绝对路径。
- [ ] fingerprint glob 全部有效。

### 模型身份

- [ ] 记录 model ID 和 revision。
- [ ] 记录 base/SFT/LoRA 身份。
- [ ] 记录 checkpoint step/epoch。
- [ ] checkpoint 有 digest 或 identity file 验证。
- [ ] 训练来源已记录。
- [ ] 与 Benchmark Dataset 的 source overlap 已审计。

### Adapter

- [ ] 每个支持 scene 都有 spatial profile。
- [ ] FPS 和帧数符合模型约束。
- [ ] I2V first-frame 来源明确。
- [ ] generic 不读取 physics。
- [ ] physics 只使用白名单字段。
- [ ] driver 不重复拼接 prompt。
- [ ] Dataset 资产未被修改。

### Driver

- [ ] deployment validation 不加载 GPU 模型。
- [ ] 关键外部代码进入 dependency fingerprint。
- [ ] dry-run 不执行昂贵推理。
- [ ] canonical job ID 和 seed 不变。
- [ ] prediction 和日志 run-local。
- [ ] return code 被记录。
- [ ] 视频完成落盘后才返回成功。

### 验证

- [ ] `baseline list` 能发现。
- [ ] `baseline inspect` 内容正确。
- [ ] `baseline validate` 通过。
- [ ] generic/physics TaskInstance 均可构建。
- [ ] 单 case dry-run 通过。
- [ ] 单 case execute 通过。
- [ ] prediction artifact SHA-256 已生成。
- [ ] evaluator 能读取输出。
- [ ] 完整测试和 compileall 通过。
- [ ] `git diff --check` 通过。

## 13. 当前参考实现

标准 managed Cosmos：

```text
baselines/cosmos3_nano_i2v/
├── baseline.json
└── driver.py
```

一行 Bundle driver + 共享 managed driver：

```text
baselines/wan22_g15_sparse_motion/
├── baseline.json
├── driver.py
└── provenance/

src/physbench/baseline_runtime/drivers/wan22.py
```

高级 command + fine-tuning：

```text
baselines/wan22_lora/
├── baseline.json
└── plugin/main.py

src/physbench/baseline_plugins/wan22.py
```

通用架构和操作参考：

- `docs/TASKS.md`：Task、三档接口和 identity contract；
- `docs/DATA_ADAPTER.md`：五阶段 adapter 和条件隔离；
- `docs/ARCHITECTURE.md`：系统边界与不变量；
- `docs/OPERATIONS.md`：运行、评估和故障排查。
