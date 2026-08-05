# Cosmos3-Nano I2V Baseline

## 1. Bundle

```text
baselines/cosmos3_nano_i2v/
├── baseline.json                 # cosmos3_nano_i2v_generic
├── physics.baseline.json         # cosmos3_nano_i2v_physics
├── baseline.local.example.json
├── baseline.local.json           # 本机部署，Git ignored
├── driver.py
└── README.md
```

两份 manifest 都是 schema 5.0 managed Baseline：

| Baseline ID | 物理策略 | Task family |
| --- | --- | --- |
| `cosmos3_nano_i2v_generic` | `ignored` | `direct_eval` |
| `cosmos3_nano_i2v_physics` | `required/structured_text` | `direct_eval` |

二者共享 Cosmos3-Nano base checkpoint、driver、本机部署、首帧与 generation shape；
Baseline ID、Bundle digest、完整 adapter fingerprint 与 TaskInstance 不同。它们运行同一
`tasks/official/five_scene_direct_eval.json`。

仓库外可能存在 Cosmos physics SFT checkpoint，但当前 Bundle 不会静默替换 base
snapshot。不同训练身份必须建立新的 Baseline ID、版本与 checkpoint identity。

## 2. 本机部署

从模板创建：

```bash
cp baselines/cosmos3_nano_i2v/baseline.local.example.json \
  baselines/cosmos3_nano_i2v/baseline.local.json
```

需要填写：

```text
model.checkpoint
runtime.framework_root
runtime.python
runtime.torchrun
runtime.hf_home
runtime.uv_cache_dir
runtime.cuda_visible_devices
```

Driver 验证 checkpoint 中 `config.json` 与
`model.safetensors.index.json` 的冻结 SHA-256、Cosmos inference entry、Python 和
torchrun。只记录路径不构成模型身份。

Benchmark orchestration 使用 `phybench` 环境；Cosmos subprocess 使用
`runtime.python/torchrun`。不要把模型环境当作 CLI 环境。

## 3. Case 输入

两种 Baseline 都从 Dataset 读取：

```text
case.text.prompt
assets.first_frame
physics[annotated=true]
```

generic：

```text
input_policy.physics.usage = ignored
physics_transform          = none
native prompt              = case.text.prompt
used_parameters            = {}
```

physics：

```text
input_policy.physics.usage           = required
input_policy.physics.representations = ["structured_text"]
physics_transform                    = append_structured_text_v1
native prompt                        = case.text.prompt + audited clauses
```

Renderer 使用：

```text
src/physbench/baseline_plugins/resources/six_scene_physics_clauses_v2.json
```

它只选择V12中`annotated=true`的独立量，验证字段、单位和symbol。Cosmos driver只读取TaskInstance的最终
`native_inputs.text.prompt`，不读取 raw physics 或另一个 prompt profile。

## 4. I2V 与时间规格

首帧固定来自 Dataset `assets.first_frame`。Cosmos driver 将源图像和 shape token 交给
模型原生预处理器：

| scene | resolution | aspect ratio |
| --- | ---: | --- |
| pendulum | 480 | `9,16` |
| free_fall | 480 | `9,16` |
| collision_1d | 480 | `16,9` |
| inclined_plane_slide | 480 | `16,9` |
| parabolic_motion | 480 | `9,16`（完整 `1:2` 内容 contain 后评估时去 margin） |
| uniform_circular_motion | 480 | `9,16` |

时间固定为：

```text
fps:        24
num_frames: 121
rule:       4n+1
```

GT 不需要与 prediction 具有相同分辨率或帧数；统一 timeline、几何归一化和
reference-bounded sampling 属于 evaluator。

generic/physics 两个 Baseline 的 materialization fingerprint 相同，只有文本/物理
stage 与完整 adapter identity 不同。

## 5. 推理

默认 runner：

```text
num_inference_steps: 35
guidance:            6.0
shift:               10.0
fps:                 24
num_frames:          121
sound:               disabled
```

Driver 为每个 job 写：

```text
jobs/<job_id>.payload.json
jobs/<job_id>.json
logs/<baseline_id>/worker_<NN>.log
predictions/_workers/worker_<NN>/<job_id>/vision.mp4
```

实际预测、payload 和日志都在当前 AtomicRun。Cosmos checkpoint、HF cache 与 uv cache
可以位于 run 外并只读。

`execute=false` 会生成 job/payload 与 planned prediction，不启动 torchrun；
`execute=true` 才按 `cuda_visible_devices` 启动多进程推理。同一 worker 只加载一次
模型并顺序消费分配给它的全部 payload；`gpus_per_worker=4` 时，4 卡部署创建一个
worker，8 卡部署创建两个并行 worker。各 worker 使用互不相交的 GPU 与输出目录，
运行前应检查全部配置设备空闲。

## 6. 验证与运行

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate cosmos3_nano_i2v_generic

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate cosmos3_nano_i2v_physics
```

单 case dry-run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_physics \
  --scene-id pendulum \
  --case-id CASE_ID \
  --run-id cosmos3_physics_pendulum_dryrun \
  --output-root runs_v2
```

同 Task 对照矩阵：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --baseline cosmos3_nano_i2v_physics \
  --matrix-id cosmos3_generic_vs_physics \
  --output-root runs_v2
```

确认 payload 中 base prompt、物理 clause、首帧、shape、seed 与 checkpoint identity 后，
使用新的 matrix ID 并加 `--execute`。

## 7. Evaluation

Cosmos 只生成 prediction。首帧在 run-local 目录中等比 contain 到模型画布，空余区域
用边缘像素复制；Dataset 资产不变，评估时该临时 margin 会按 sealed contract 排除。
Reference、parent reference、mask、timeline 与 scene-local
物理评分由 Benchmark evaluator 解析。无可信物理 reference 的 OOD case 返回明确错误，
不会伪造 GT。

Case 产物包括：

```text
evaluation/cases/<job_id>/result.json
evaluation/cases/<job_id>/per_frame.csv
evaluation/cases/<job_id>/physical_subject_iou_curve.png
```

完整评测与故障排查见 [运行指南](OPERATIONS.md)。
