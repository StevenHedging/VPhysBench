# WAN2.2 Baseline：View A LoRA 与 G15

## 1. Bundle 与共享实现

当前有两个 WAN Bundle：

```text
baselines/wan22_lora/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json                 # 本机部署，Git ignored
└── plugin/main.py                       # thin command endpoint

baselines/wan22_g15_sparse_motion/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json                 # 本机部署，Git ignored
├── plugin/main.py                      # thin command endpoint
└── provenance/benchmark_overlap_v3.json

src/physbench/baseline_plugins/wan22.py # 唯一 WAN 实现
src/physbench/baseline_plugins/resources/five_scene_i2v_v1/
```

身份与能力：

| baseline ID | model | family | 用途 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3` | WAN2.2-TI2V-5B + run-private/frozen LoRA | finetune, direct | 正式四任务 |
| `wan22_g15_sparse_motion_r32_e20` | WAN2.2-TI2V-5B + G15 step-2840 | direct only | 诊断 |

核心 Registry 不包含 WAN 分支。两个 `plugin/main.py` 都使用通用 endpoint，将四个
协议操作分发给同一 `Wan22BaselinePlugin`。模型身份、能力、checkpoint 和 provenance
由各自 manifest 决定，TaskBuilder 不按 baseline ID 写分支。

WAN 当前仍复用旧 runner 的 `wan22_lora.py`、`wan22_media.py` 和三个执行脚本。
TaskBuilder 会逐文件计算共享实现、profile、协议 endpoint、兼容层和执行脚本的
SHA-256，并把它们写入自身 fingerprint 与
`describe.runtime_dependency_fingerprints`。因此去重不会制造身份盲区。

## 2. 便携配置与本机部署

`baseline.json` 保存可提交、可迁移的默认值。当前机器的外部依赖写在
`baseline.local.json`：

```text
pipeline root: /root/Steven/wan22_pendulum_pipeline
model Python:  /root/miniconda3/envs/dlp/bin/python
devices:       0,1,2,3,4,5,6,7
DiffSynth:     fb337fbb90945ff829de69dbd44ded618f73e889
```

另一个机器从模板创建本地配置：

```bash
cd baselines/wan22_lora
cp baseline.local.example.json baseline.local.json
```

只可覆盖 `runtime` 和 `model`。本地文件不进入 portable bundle digest，但其解析结果
进入 deployment digest。`BaselineTaskInstance.identity.baseline` 同时冻结这两个
digest。

Benchmark command endpoint 始终使用 `phybench` 环境启动；WAN 的训练和生成执行器再
使用 `runtime.python` 调用模型环境。这两个 Python 角色不能互换。

G15 本机配置将 `frozen_lora_checkpoint` 指向 step-2840。Task build 会读取该文件并
验证 SHA-256 必须为
`cd19f851133c8370def00991fa778a3bfb581e35713842438ba068495912906f`。

## 3. DataAdapter

空间 bucket：

| bucket | scene | target |
| --- | --- | --- |
| portrait | pendulum, free_fall, uniform_circular_motion | 480 × 832 |
| landscape | collision_1d, inclined_plane_slide | 832 × 480 |

时间规格：

- 24 FPS；
- 最多 121 帧，至少 5 帧；
- 维持 WAN 所需 `4n+1`；
- 按物理时间读取视频前缀；
- 不把 GT 首帧补进生成视频。

Reference derivative 与 generation 使用不同且有审计的取整方向：reference 只能向下选择
源视频可安全解码的 `4n+1` 帧数；generation 必须向上选择最后时间戳覆盖 reference
物理区间的 `4n+1` 帧数。两者不能共用一个 frame count，否则会出现“容器 duration
足够、最后一帧时间戳却少一帧”的系统性错误。generation 最多仍为 121 帧，不复制
GT 帧或末帧。

输入范式是 I2V。首帧优先来自 Dataset `assets.first_frame`；缺失时从 canonical
reference 第 0 帧确定性提取。Dataset 原件只读，派生媒体进入内容寻址 cache。

DataAdapter 的五个阶段 fingerprint 都包含共享实现和 profile digest。因此代码改变会使 adapter
fingerprint 失效；空间、时间或媒体实现改变也会使 materialization fingerprint 和
cache namespace 失效。generic/physics profile 内容只进入文本或物理 stage，单纯修改
prompt 不会使媒体 cache 失效；TaskBuilder fingerprint 仍会变化。

## 4. Conditioning

generic profile：

- 只生成 scene 通用描述；
- 文本阶段接收的 case 视图不包含结构化 physics；
- `used_parameters` 必须为空；
- 物理注入 stage 显式记录为 disabled。

physics profile：

- 按 scene 白名单读取结构化物理量；
- 保留值和单位；
- 追加到 WAN 原生 `native_inputs.text.prompt`；
- 在 adaptation audit 中记录使用字段。

generic 与 physics 使用相同的空间、时间和输入范式配置，但生成不同的完整 adapter
fingerprint 和 TaskInstance。

## 5. View A Fine-tuning

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

每个 conditioning 产生独立 TaskInstance、部署身份和训练 artifact。训练 checkpoint
通过 `artifact://train/model` 绑定到后续推理；predictor 不搜索“最新 checkpoint”。
这一节只适用于 `wan22_ti2v_5b_lora_r32_v3`；G15 manifest 不含 trainer component，
收到 `finetune_eval` 会在 run 创建前失败。

## 6. G15 身份、训练来源与可比性

G15 是 rank-32 LoRA，目标模块为 q/k/v/o/ffn.0/ffn.2；训练 20 epochs，采用 sparse
tube FlowMatch boost=1.0。集成固定选择 epoch 20 的 step-2840，而不是运行时搜索最新
checkpoint。

G15 的训练集使用了五场景源素材。污染审计不能只比较 case ID：斜面和圆周在 G15
训练目录中使用另一套 ID，但底层 capture/trial 与 benchmark v3 相同。source-aware
结果：

| 集合 | 总数 | G15 见过 |
| --- | ---: | ---: |
| benchmark v3 | 214 | 176 |
| View A train | 121 | 121 |
| View A test ID | 23 | 8 |
| View A test OOD1 | 43 | 20 |

因此 G15 全量结果只能标为 `diagnostic_pretrained`，不能进入无泄漏排名。38 个 source
unseen case 位于旧三场景测试集合；只评它们时也必须明确标为 clean-subset
diagnostic，而不是完整 Task score。逐 scene 证据与输入清单 digest 在 Bundle
provenance 文件中。

## 7. Generation

Predictor 配置：

- 50 inference steps；
- CFG 5.0；
- LoRA alpha 1.0；
- tiled inference；
- quality 5；
- 使用 Bundle 冻结的 negative prompt。

`run_task` 在 execute=false 时仍会完整 materialize job specs，但不会加载 GPU 模型。
execute=true 时，每个 GPU worker 常驻一个模型并处理自己的 job 分片。每个 canonical
job 必须返回一条 complete、failed 或 staged prediction record。

Worker 只把模型代码、base checkpoint 和 LoRA 当作外部只读依赖；生成视频写入当前
`run_dir/predictions/`，worker 日志写入
`run_dir/artifacts/wan22/inference_workers/`。从旧实验目录复用的视频必须先通过
`physbench prediction-import` 复制进 run，不能把 Brady 工程路径作为正式
`video_path`。

## 8. 验证与 dry-run

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20
```

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output /tmp/wan22_task_instance.json
```

单 case AtomicRun dry-run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --scene-id pendulum \
  --case-id <pendulum_case_id> \
  --output-root /tmp/physbench-runs
```

加入 `--execute` 前应检查：

- local override 中 pipeline、Python、model base 和 accelerate config；
- 冻结 LoRA checkpoint；
- bundle/deployment digest；
- TaskInstance 的 canonical jobs、bucket 和 cache root；
- GPU 列表和输出目录。

## 9. 评估边界

WAN 只训练和生成，不定义正式分数。生成完成后统一调用 Benchmark TaskEvaluator。
Bundle 不得替换 canonical jobs、跳过失败记录、修改 Dataset reference、在无 GT case
伪造 reference，或覆盖 scene evaluator 协议。
