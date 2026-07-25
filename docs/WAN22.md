# WAN2.2 + LoRA Baseline

## 1. 自包含 Bundle

当前 WAN Bundle：

```text
baselines/wan22_lora/
├── baseline.json
├── baseline.local.example.json
├── baseline.local.json                 # 本机部署，Git ignored
├── plugin/
│   ├── main.py                         # command protocol endpoint
│   └── implementation.py               # TaskBuilder/DataAdapter/executor
└── task_builder/data_adapter/profiles/
    ├── generic.json
    └── physics.json
```

身份：

```text
baseline_id:      wan22_ti2v_5b_lora_r32_v3
baseline_version: 1.0.0
implementation:   command / physbench-baseline-v1
model:            WAN2.2-TI2V-5B
```

核心 Registry 不包含 WAN 分支。`plugin/main.py` 实现通用 `describe`、
`adapt_case`、`build_task_instance` 和 `run_task` 操作。

WAN 当前仍复用旧 runner 的 `wan22_lora.py`、`wan22_media.py` 和三个执行脚本。
TaskBuilder 会逐文件计算这些兼容依赖的 SHA-256，并把它们写入自身 fingerprint 与
`describe.runtime_dependency_fingerprints`。因此兼容代码或脚本变化同样会使实例身份
失效，不会落入 Bundle 目录指纹之外的盲区。

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

输入范式是 I2V。首帧优先来自 Dataset `assets.first_frame`；缺失时从 canonical
reference 第 0 帧确定性提取。Dataset 原件只读，派生媒体进入内容寻址 cache。

DataAdapter 的五个阶段 fingerprint 都包含实现文件 digest。因此代码改变会使 adapter
fingerprint 失效；空间、时间或媒体实现改变也会使 materialization fingerprint 和
cache namespace 失效。generic/physics profile 内容只进入文本或物理 stage，单纯修改
prompt 不会使媒体 cache 失效；portable bundle digest 与 TaskBuilder fingerprint
仍会变化。

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

## 5. Fine-tuning

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

## 6. Generation

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

## 7. 验证与 dry-run

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3
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

## 8. 评估边界

WAN 只训练和生成，不定义正式分数。生成完成后统一调用 Benchmark TaskEvaluator。
Bundle 不得替换 canonical jobs、跳过失败记录、修改 Dataset reference、在无 GT case
伪造 reference，或覆盖 scene evaluator 协议。
