# WAN2.2 + LoRA Baseline 适配

> v2 新实验以 `baselines/wan22_lora/baseline.json` 为准。WAN Baseline 内置
> `Wan22TaskBuilder`；统一 DataAdapter 是该 Builder 的内部组件。Builder 接收
> `DatasetSnapshot + TaskSpec`，输出经过 SHA-256 封印的 `BaselineTaskInstance`，
> WAN 的训练器和预测器只在该实例上运行。通用契约见
> [TaskBuilder 架构](TASK_BUILDER_ARCHITECTURE.md)和
> [统一 Data Adapter 架构](DATA_ADAPTER_ARCHITECTURE.md)。

## 1. 基线定义

模型为本地 `WAN2.2-TI2V-5B`，训练算法沿用已有实验的 FlowMatch SFT + LoRA：rank 32，目标模块
`q,k,v,o,ffn.0,ffn.2`，AdamW 由既有 DiffSynth 训练入口管理。Benchmark 不复制模型和权重，只保存绝对路径、DiffSynth commit 和每次 run 的冻结配置。

Baseline bundle 的 v2 组件关系为：

```text
WAN22 Baseline
├── Wan22TaskBuilder
│   └── Wan22DataAdapter
│       ├── spatial
│       ├── temporal
│       ├── paradigm (TI2V first frame)
│       ├── text
│       └── physics (WAN-specific text injection)
├── FlowMatch LoRA Trainer
└── TI2V Predictor
```

TaskBuilder 构建阶段不调用 GPU，也不改写 Dataset。它先接受 Benchmark 已确定的
`CanonicalTaskPlan`，完成能力检查和五阶段适配，再编译 training spec、inference
jobs、缓存绑定与 operation DAG。实例中的 `native_inputs` 是 WAN 私有结构；核心框架
不会假定其提示词、首帧或物理注入的具体字段。

## 2. v2 Task 1：视图 A 调推一体

AtomicTask：`tasks/official/finetune_eval_generic.json` 或
`tasks/official/finetune_eval_physics.json`。

1. 读取 task 选中的多 scene `train_case_ids`。
2. TaskBuilder 为训练和推理 case 生成 WAN-native adaptation record。
3. 执行时在 run/cache 内创建 WAN 专属派生训练集与 metadata，Dataset 资产保持只读。
4. 从 WAN2.2 base model 联合微调一个新 LoRA；不同 scene 不偷看测试集。
5. TaskBuilder 用 `artifact://train/model` 把所有 `test_id`、`test_ood1` 推理 job
   绑定到本实例的训练产物。
6. 冻结最终 checkpoint manifest，并按实例 operation DAG 执行推理和评测。

所有训练素材仅来自视图 A 的 `train`。OOD1 只参加推理，不进入微调缓存。

`generic` 与 `physics` 是两份独立 AtomicTask，因此分别构建实例、分别训练 LoRA，
不会共享 Adapter。二者的 canonical plan 可以由 `matrix-run` 验证为相同的 case、
划分和 seed，以形成严格受控的物理注入对比。

## 3. v2 Task 2：视图 B 零训练评测

AtomicTask：`tasks/official/direct_eval_generic.json` 或
`tasks/official/direct_eval_physics.json`。可通过 `atomic-run` 的 `--scene-id`、
`--group`、`--case-id` 覆盖选择范围。

该模式不在 Benchmark 内训练或微调，直接加载既有
`baseline.json` 中的 `model.frozen_lora_checkpoint`。TaskBuilder 以
`baseline://frozen_model` 作为模型引用，并生成 `infer → evaluate` DAG；如果 checkpoint
不存在、scene/conditioning/任务族不受支持，会在运行前拒绝构建，而不是产生无效结果。

当前 bundle 声明支持 `pendulum`、`free_fall` 和 `collision_1d`，冻结 checkpoint 来自
已发布的三 scene LoRA。若改用某个单 scene checkpoint，应同步收窄
`supported_scenes`，避免把领域错配结果当成正式 baseline。

## 4. v2 Case 到 WAN 输入的映射

| Case 内容 | WAN 输入 |
|---|---|
| `assets.first_frame` 可用 | 归一化该首帧并作为 TI2V 图像条件 |
| 没有独立首帧但有真实/物理参考视频 | 在模型缓存中抽取视频第 0 帧 |
| `generic` conditioning | text stage 只描述 scene 物理过程，不读取详细物理参数 |
| `physics` conditioning | physics stage 从结构化 `physics` 标注生成固定格式信息，并追加到已适配文本 |
| 结构化物理标注 | 保留在 source case 与审计记录；只有 physics stage 可将其写入 WAN-native 输入 |

这保证 Dataset 不携带 WAN 提示词或模型 input view，可以同时服务其他 T2V/I2V/V2V
Baseline。空间、时间和输入范式适配也全部留在 WAN TaskBuilder 内，不改写原始 case。

## 5. 不同分辨率、FPS、帧数的模型侧适配

源文件始终只读。v2 的通用编译产物位于：

```text
runs_v2/<run_id>/task_instance/
├── manifest.json
├── canonical_plan.json
├── adaptations.jsonl
├── training.json
├── inference_jobs.jsonl
├── execution_graph.json
├── cache_bindings.json
└── baseline_payload.json
```

WAN 执行阶段的兼容派生文件仍位于：

```text
<run_dir>/artifacts/wan22/
├── dataset/videos/                 # Task 1 训练派生视频
├── first_frames/                   # 推理首帧
├── evaluation_references/          # 与 WAN 输出规格对齐的评测参考
├── training_media_audit.jsonl      # 每条时空变换及 ffmpeg 命令
├── training_spec.json
├── checkpoint.json
└── checkpoints/
```

默认策略：

- 空间：支持由 baseline 配置声明 scene 分桶。当前三场景 8 卡配置中，单摆/自由落体进入
  `480×832` 竖屏桶，碰撞进入 `832×480` 横屏桶；每个桶内保持宽高比缩放再 padding，
  不拉伸、不裁掉物理主体。原始数据保持只读，不做横竖屏转换。
- 时间：先按 case 的 `temporal.encoded_to_physical_speed` 恢复物理时间，再依据时间戳重采样为 24 FPS，不用原始帧序号冒充统一时间。
- 长视频：从真实 `t=0` 取最多 121 帧，对应释放后的统一前缀。
- 短视频：不循环、不慢放、不冻结尾帧；向下取最近的合法 `4n+1` 帧，最低 5 帧。这个下限允许真实时间仅约 0.3 秒的自由落体 case 保持原有物理速度。
- 推理：若有物理参考，生成帧数与该 case 的派生参考一致；无参考 OOD 首帧使用121帧。
- 评测：Prediction/VisualJudgment 优先使用该 baseline 的派生 reference，从而与生成视频具有相同空间与时间基准；原始 GT 不被覆盖。

adapter 仍支持通过非 1 的 `encoded_to_physical_speed` 适配外部慢放数据，但当前官方自由落体数据已经在数据侧恢复为真实时间，manifest 因子为 `1.0`，WAN 不会再次加速。每条 job 会保留 source probe、物理时长、速度因子和完整 ffmpeg 命令，可追踪是否发生了时间尺度污染。

## 6. v2 构建与运行

只构建 sealed task instance，不训练、不推理：

```bash
PYTHONPATH=src python3 -m physbench task-build \
  --dataset datasets/physics_video/releases/2.0.0/dataset.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/wan22_finetune_physics.instance.json
```

执行单个 AtomicTask。`atomic-run` 会先调用同一个 TaskBuilder，再冻结实例并执行；
不加 `--execute` 时只完成编译、冻结与模型侧计划：

```bash
PYTHONPATH=src python3 -m physbench atomic-run \
  --dataset datasets/physics_video/releases/2.0.0/dataset.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

要做 generic/physics 配对实验，使用两个独立 Task 和两个独立 LoRA：

```bash
PYTHONPATH=src python3 -m physbench matrix-run \
  --dataset datasets/physics_video/releases/2.0.0/dataset.json \
  --task tasks/official/finetune_eval_generic.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --matrix-id wan22_task1_conditioning_ablation \
  --output-root runs_v2
```

确认实例、媒体审计与计划后，加 `--execute` 才实际启动 LoRA 微调和 WAN 推理。

## 7. v1 兼容运行与历史产物

以下 `physbench run`、`configs/tasks/` 和 `configs/baselines/` 用法保留用于复现实验，
不代表 v2 TaskBuilder 契约。

不执行模型，仅生成全部训练/推理计划：

```bash
PYTHONPATH=src python3 -m physbench run \
  --task configs/tasks/view_a_three_scene_finetune.json \
  --baseline configs/baselines/wan22_ti2v_5b_lora_three_scene_8gpu_buckets.json \
  --manifest datasets/physics_video/releases/1.0.0/cases.jsonl \
  --split datasets/physics_video/releases/1.0.0/views/view_a.json \
  --output-root runs
```

上述官方三场景 task 使用 `physics_natural` 生成唯一一份训练 metadata，并在同一
Adapter 上评测 `generic` 和 `physics_natural`。也可在命令行显式覆盖：

```bash
PYTHONPATH=src python3 -m physbench run \
  --task configs/tasks/view_a_three_scene_finetune.json \
  --baseline configs/baselines/wan22_ti2v_5b_lora_three_scene_8gpu_buckets.json \
  --manifest datasets/physics_video/releases/1.0.0/cases.jsonl \
  --split datasets/physics_video/releases/1.0.0/views/view_a.json \
  --train-prompt-profile physics_natural \
  --eval-prompt-profile generic \
  --eval-prompt-profile physics_natural \
  --output-root runs
```

同一 case 的两个结果分别写入 `predictions/generic/` 与
`predictions/physics_natural/`；报告按 profile 独立聚合。

确认冻结 job 后，加 `--execute` 才会实际调用配置中声明的多卡 LoRA 微调和 WAN 推理。

第一类任务还可以在常规 `test_id` / `test_ood1` 之外，随机抽取每个 scene 的已见训练样本进行
回放推理。以下参数抽取每类 2 条，抽样可由 seed 完整复现；设为 `0` 或不传即关闭：

```bash
PYTHONPATH=src python3 -m physbench run \
  --task configs/tasks/view_a_three_scene_finetune.json \
  --baseline configs/baselines/wan22_ti2v_5b_lora_three_scene_8gpu_buckets.json \
  --manifest datasets/physics_video/releases/1.0.0/cases.jsonl \
  --split datasets/physics_video/releases/1.0.0/views/view_a.json \
  --train-preview-per-scene 2 \
  --train-preview-seed 42 \
  --output-root runs
```

这些额外 job 的 `evaluation_partition` 为 `train_seen`，不会改变训练集、ID/OOD1 划分，也不会进入
官方 ID/OOD 总分；它们只保留独立 breakdown 与辅助统计。
抽中的 case ID 会冻结在 `plan.json` 的 `train_preview.selected_case_ids_by_scene` 中。
并行推理结束后还会自动生成 `train_seen_gallery.md`，其中包含生成视频链接、均匀时间采样分镜及
对应训练参考视频分镜，便于直接检查“记住训练样本”的程度。

完整的“8 卡训练 → loss 导出 → 8 卡并行推理 → train_seen 图集”也可以用监督脚本一次启动：

```bash
python3 scripts/run_wan22_task1_full.py \
  --run-id <唯一实验名> \
  --train-preview-per-scene 2 \
  --train-preview-seed 42 \
  --gpus 0,1,2,3,4,5,6,7
```

阶段状态会持续写入 `runs/<run-id>.supervisor.json`；训练日志、worker 日志与最终产物仍全部保存在
该 run 的独立目录中。

Task 2 将 task/baseline 分别换成 `view_b_zero_shot_pendulum.json` 与
`wan22_ti2v_5b_lora_task2_pendulum.json`；不发生训练。
