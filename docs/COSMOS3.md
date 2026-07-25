# Cosmos3-Nano I2V Baseline

## 1. 身份与范围

Bundle：

```text
baselines/cosmos3_nano_i2v/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json              # 本机部署，Git ignored
├── README.md
└── plugin/
    ├── main.py                      # 通用 command endpoint
    └── implementation.py            # Cosmos TaskBuilder/DataAdapter/executor
```

正式 ID 是 `cosmos3_nano_i2v`，模型身份固定为 base
`nvidia/Cosmos3-Nano` snapshot
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`。它只支持
`direct_eval × {generic, physics}`，没有 trainer，也不会接受 `finetune_eval`。

`/root/Nico/cosmos` 中另有 physics SFT iter-300，但该模型没有被静默替换为本
Baseline。base 与 SFT 是不同训练身份；如果以后评 SFT，必须用新的 baseline ID、
manifest 和训练来源审计。

## 2. 本机部署

当前部署：

```text
framework:  /root/Nico/cosmos/packages/cosmos3
python:     /root/Nico/cosmos/packages/cosmos3/.venv/bin/python
torchrun:   /root/Nico/cosmos/packages/cosmos3/.venv/bin/torchrun
checkpoint: /root/Nico/cosmos/models/Cosmos3-Nano
GPUs:       0,1,2,3
offline:    true
```

新机器从 `baseline.local.example.json` 创建 Git-ignored local override。portable
manifest 记录 HF revision，并验证 checkpoint 的 `config.json` 与
`model.safetensors.index.json` SHA-256。这样无需每次重扫 35 GB shard，也不会把错误
snapshot 当成同一部署。

TaskBuilder 还记录实际
`cosmos_framework/scripts/inference.py` 的 SHA-256 和 Cosmos Git commit。外部框架
入口变化会改变 TaskBuilder fingerprint。

## 3. DataAdapter

输入范式严格为 I2V。v3 的 214 个 case 都有 `assets.first_frame`；adapter 不从 GT
视频补帧，也不修改 Dataset。

| scene | Cosmos resolution | aspect token |
| --- | ---: | --- |
| pendulum | 480p | `9,16` |
| free_fall | 480p | `9,16` |
| collision_1d | 480p | `16,9` |
| inclined_plane_slide | 480p | `16,9` |
| uniform_circular_motion | 480p | `4,3` |

默认 timeline 是 24 FPS、121 帧，满足 Cosmos `4n+1`。这些是生成规格，不是
evaluator 对 GT 的要求。Evaluator 按 scene reference 决定物理时间轴，再分别采样、
letterbox 生成视频与 reference，因此两者原始分辨率和帧数可以不同。

generic adaptation 接收一个移除 `case.physics` 的视图，`used_parameters` 必须为空。
physics adaptation 通过 `five_scene_i2v_v1` 白名单把值和单位写入 Cosmos 原生 prompt。
两者共享 first-frame、shape 与 materialization fingerprint。

## 4. 执行

每个 job 生成两份 run-private 文件：

```text
jobs/<job_id>.payload.json  # cosmos_framework 原生 inference payload
jobs/<job_id>.json          # Benchmark prediction/job 审计
```

executor 使用 local `torchrun` 启动
`cosmos_framework.scripts.inference`。一个 job 使用
`runtime.cuda_visible_devices` 中的全部 GPU；job 之间串行，避免多个 FSDP 组争用同一
设备。环境显式设置 HF/UV cache、CUDA library path 与 offline 模式。输出位于：

```text
predictions/<conditioning>/<job_id>/vision.mp4
```

不加 `--execute` 时只写 payload、job spec 和 `planned` prediction，不加载模型。

## 5. 验证

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate cosmos3_nano_i2v
```

单 case physics dry-run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline cosmos3_nano_i2v \
  --scene-id collision_1d \
  --case-id collision_r2_medium_steel_medium_steel_medium_steel_v02818 \
  --output-root runs_v2
```

执行前应确认四张 GPU 同时空闲。GPU 被其他训练占用时，保持 dry-run，或复用身份明确
的既有生成视频做非官方 evaluator smoke；不能在资源不足时降低并行度却沿用同一
deployment identity。

## 6. 评估边界

Cosmos 只输出 prediction，不拥有 case score。所有视频统一进入 Benchmark 的
scene-local evaluator。无同 case GT 的 OOD case 只允许使用 Dataset 登记且 physics
逐项相同的 parent reference；无可信 parent 时显式 unavailable。
