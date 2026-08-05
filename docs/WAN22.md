# WAN2.2 Baseline

## 1. Bundle 与 Baseline identity

WAN 有三个 Bundle 目录、五个 Baseline identity：

```text
baselines/wan22_lora/
├── baseline.json                 # ..._generic
├── physics.baseline.json         # ..._physics
├── baseline.local.example.json
├── baseline.local.json           # 本机部署，Git ignored
└── driver.py

baselines/wan22_g15_sparse_motion/
├── baseline.json                 # ..._generic
├── physics.baseline.json         # ..._physics
├── baseline.local.example.json
├── baseline.local.json           # 本机部署，Git ignored
├── driver.py
└── provenance/benchmark_overlap_v3.json

baselines/wan22_quantity_embedding/
├── baseline.json                 # ..._quantity_embedding_v1
├── baseline.local.example.json
├── baseline.local.json           # 本机部署，Git ignored
├── adapter.py
├── driver.py
├── quantity_registry.json             # 历史V1资源
└── quantity_registry_v2.json          # 当前V12独立量资源
```

| Baseline ID | Task family | 物理策略 | 用途 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3_generic` | `finetune_eval`, `direct_eval` | `ignored` | 正式 WAN LoRA |
| `wan22_ti2v_5b_lora_r32_v3_physics` | `finetune_eval`, `direct_eval` | `required/structured_text` | 同模型，物理文本追加 |
| `wan22_ti2v_5b_lora_r32_quantity_embedding_v1` | `finetune_eval` | `required/quantity_token_embedding_v1` | SI 数值/量纲编码 |
| `wan22_g15_sparse_motion_r32_e20_generic` | `direct_eval` | `ignored` | 冻结 G15，诊断型 |
| `wan22_g15_sparse_motion_r32_e20_physics` | `direct_eval` | `required/structured_text` | 冻结 G15，诊断型 |

五者都使用 schema 5.0 managed runtime。同目录两份 manifest 共享 driver、模型部署与
`baseline.local.json`，但拥有不同 Baseline ID、Bundle digest、adapter fingerprint
和 TaskInstance。它们运行相同的官方 Task，不再用 Task 文件区分物理信息注入。

共享执行实现：

```text
baselines/*/driver.py
└── Wan22ManagedDriver
    └── Wan22ExecutionEngine
        ├── Wan22LoraAdapter
        └── Wan22MediaAdapter
```

Quantity Baseline 使用专有 Python adapter 和 managed execution engine，从结构化标注
中读取 registry 明确筛选的物理量子集，并在冻结 UMT5 输出与 DiT
cross-attention 之间注入编码。算法、命令、审计产物与结果模板见
[WAN2.2 物理量编码 Baseline](WAN22_QUANTITY_EMBEDDING.md)。

模型专有兼容层仍复用既有 WAN 训练/推理代码，但 canonical plan、Case projection、
input policy、adapter audit、TaskInstance seal 和 run identity 由当前公共 runtime
负责。

## 2. Portable 配置与本机部署

Manifest 保存可提交的模型语义。本机依赖写入：

```text
baselines/wan22_lora/baseline.local.json
baselines/wan22_g15_sparse_motion/baseline.local.json
```

从模板创建：

```bash
cp baselines/wan22_lora/baseline.local.example.json \
  baselines/wan22_lora/baseline.local.json
```

本机配置通常包含：

```text
project_root
runtime.python
model_base
cuda_visible_devices
accelerate_config
frozen_lora_checkpoint
```

只允许覆盖 `model` 与 `runtime`。Portable bundle digest 与应用 local override 后的
deployment digest 都会进入 TaskInstance。

G15 checkpoint 固定为 step-2840，预期 SHA-256：

```text
cd19f851133c8370def00991fa778a3bfb581e35713842438ba068495912906f
```

WAN LoRA checkpoint 固定为 step-410，预期 SHA-256：

```text
7f8f28a36faa309431e7ea58e7de3c61cd58266b62653ee69c3b9f666745acfe
```

Driver 在 task build/validate 阶段复核文件内容，不运行时搜索“最新 checkpoint”。

## 3. Case 输入与物理策略

所有 WAN Baseline 都接收同一 Case 投影：

```text
case.text.prompt
assets.first_frame
appearance / temporal / ood
physics[annotated=true]
```

generic Baseline：

```text
input_policy.physics.usage = ignored
adapter.physics_transform  = none
native prompt              = case.text.prompt
used_parameters            = {}
```

physics Baseline：

```text
input_policy.physics.usage            = required
input_policy.physics.representations  = ["structured_text"]
adapter.physics_transform             = append_structured_text_v1
native prompt                         = case.text.prompt + audited clauses
```

物理模板：

```text
src/physbench/baseline_plugins/resources/six_scene_physics_clauses_v2.json
```

Renderer只读取scene白名单内`annotated=true`的独立量，验证单位与symbol，并记录
value/unit/symbol和渲染值。速度为非负大小，方向由原始Case prompt表达。WAN
driver 只消费 TaskInstance 中已封印的最终 prompt，不再自行选择 prompt profile 或读取
raw physics side channel。

generic/physics 两个 Baseline 使用相同首帧、空间/时间 recipe 和
materialization fingerprint；文本/物理阶段与完整 adapter fingerprint 不同。

## 4. I2V 媒体适配

空间 bucket：

| bucket | scene | target |
| --- | --- | --- |
| portrait | pendulum, free_fall, uniform_circular_motion | 480 × 832 |
| landscape | collision_1d, inclined_plane_slide | 832 × 480 |

时间规格：

- 24 FPS；
- 5–121 帧；
- 帧数满足 `4n+1`；
- 当前推理固定使用Baseline原生最大长度121帧，不读取GT时长；
- 不把 GT 首帧或末帧补进 prediction。

I2V 只使用显式 `assets.first_frame`。Dataset 原件只读，resize/pad/抽帧等派生物进入
内容寻址 cache 或当前 run。

Reference解码、共同物理时间前缀、统一采样率与分辨率归一化属于evaluator。生成视频
不要求与GT具有相同像素尺寸、FPS、帧数或物理时长；评估区间取双方实际物理时长的
较短者。

## 5. View A fine-tuning

`wan22_ti2v_5b_lora_r32_v3_*` 的 trainer recipe：

| 参数 | 值 |
| --- | --- |
| algorithm | FlowMatch SFT + LoRA |
| rank | 32 |
| target modules | q, k, v, o, ffn.0, ffn.2 |
| learning rate | 1e-4 |
| epochs | 10 |
| dataset repeat | 4 |
| precision | bf16 |
| optimizer | AdamW |
| scheduler | ConstantLR |
| scene balancing | oversample each scene to largest |
| seed | canonical plan training seed |

两种 WAN LoRA Baseline 都运行同一
`tasks/official/five_scene_finetune_eval.json`。每个 Baseline 建立独立 AtomicRun、
TaskInstance、训练 prompt、checkpoint artifact 与预测，不共享训练后模型。

训练输出通过 `artifact://train/model` 绑定到同一 TaskInstance 的推理阶段；predictor
不扫描外部目录选择 checkpoint。

G15 manifest 不含 task-specific trainer，只支持 direct-eval。

## 6. G15 数据重叠与可比性

G15 是 rank-32、20 epoch 的 sparse-motion LoRA，固定使用 step-2840。它的训练素材与
Benchmark 的底层 capture/trial 有重叠；case ID 不同也不能视为独立样本。

Source-aware audit：

| 集合 | 总数 | G15 见过 |
| --- | ---: | ---: |
| Dataset | 214 | 176 |
| View A train | 121 | 121 |
| View A test ID | 23 | 8 |
| View A test OOD1 | 43 | 20 |

审计文件名保留 `v3`，因为它最初针对 3.0.0 生成；Dataset 4.0.0 没有改变 case、View
或媒体，所以 overlap 结论仍适用。

G15 全量分数必须标记为 `diagnostic_pretrained`，不能进入无泄漏排名。Source-unseen
subset 也只能作为明确的 clean-subset diagnostic，不能冒充完整 Task score。

## 7. Generation

当前 WAN predictor：

- 50 inference steps；
- CFG 5.0；
- LoRA alpha 1.0；
- tiled inference；
- quality 5；
- 使用 manifest 冻结的 negative prompt。

`execute=false` 会物化 TaskInstance、adaptation、训练/推理 job 和 planned prediction，
但不加载 GPU 模型。`execute=true` 时按可用 GPU 启动常驻 worker 并分片 canonical
jobs。

模型代码、base checkpoint 与初始 LoRA 可以位于 run 外并只读；以下内容必须 run-local：

```text
training specs / checkpoints
job payloads
worker logs
generated videos
predictions.jsonl
evaluation
```

从历史工程复用的视频必须先用 `physbench prediction-import` 复制进 run。

## 8. 验证

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3_generic

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3_physics

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_quantity_embedding_v1

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20_generic

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20_physics
```

编译 fine-tune Task：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --output /tmp/wan22_physics_finetune_task.json
```

同 Task 的 direct-eval 对照矩阵：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --matrix-id wan22_generic_vs_physics \
  --output-root runs_v2
```

首次接入或迁移机器时先做单 case dry-run，再用新 run ID 加 `--execute`。完整操作见
[运行、验证与故障排查](OPERATIONS.md)。
