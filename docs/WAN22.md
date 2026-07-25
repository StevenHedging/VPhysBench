# WAN2.2 + LoRA Baseline

## 1. Bundle

当前 bundle：

```text
baselines/wan22_lora/baseline.json
baseline_id: wan22_ti2v_5b_lora_r32_v2
plugin:      wan22_lora
model:       WAN2.2-TI2V-5B
```

支持五个正式 scene、两个 task family 和 generic/physics 两种 conditioning。

## 2. Runtime

Bundle 当前绑定：

```text
pipeline root: /root/Steven/wan22_pendulum_pipeline
python:        /root/miniconda3/envs/dlp/bin/python
devices:       0,1,2,3,4,5,6,7
precision:     bf16
DiffSynth:     fb337fbb90945ff829de69dbd44ded618f73e889
```

这些是可执行依赖。执行前应验证目录、Python、模型权重、accelerate config 和冻结
LoRA checkpoint 均存在。TaskBuilder dry-run 不要求加载 GPU 模型。

## 3. DataAdapter

空间 bucket：

| bucket | scene | target |
| --- | --- | --- |
| portrait | pendulum, free_fall, uniform_circular_motion | 480 × 832 |
| landscape | collision_1d, inclined_plane_slide | 832 × 480 |

时间：

- 24 FPS；
- 最多 121 帧；
- 至少 5 帧；
- 保持 `4n+1`；
- 按物理时间取前缀。

输入范式是 I2V。首帧来自 Dataset `assets.first_frame`，缺失时从 reference frame 0
确定性提取。所有派生媒体写入内容寻址 cache。

## 4. Conditioning

Profiles：

```text
baselines/wan22_lora/task_builder/data_adapter/profiles/generic.json
baselines/wan22_lora/task_builder/data_adapter/profiles/physics.json
```

generic：

- 只使用 scene 和外观可见事实；
- 禁止读取详细结构化物理值；
- 不允许通过预渲染 prompt 间接泄漏。

physics：

- 使用 scene profile 白名单物理量；
- 保留值和单位；
- 追加到 `native_inputs.text.prompt`；
- 在 adaptation audit 中记录字段。

## 5. Fine-tuning

Trainer 配置：

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
| scene balancing | oversample to largest scene |
| seed | 42 |

每个 conditioning 产生独立 TaskInstance 和独立训练 artifact。Checkpoint 必须通过
operation DAG 绑定，不能由 predictor 在运行时搜索“最新文件”。

## 6. Generation

Predictor：

- 50 inference steps；
- CFG 5.0；
- LoRA alpha 1.0；
- tiled inference；
- quality 5；
- 使用 bundle 中冻结的 negative prompt。

每个 job 必须产生一条 prediction record。失败时保存明确状态和原因，不能省略记录。

## 7. TaskBuilder dry-run

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/wan22_task_instance.json
```

检查重点：

- 五场景能力通过；
- View A training/eval case 完整；
- portrait/landscape bucket 正确；
- generic/physics 媒体绑定相同；
- operation DAG 中 checkpoint 路由明确；
- TaskInstance 指纹可重复。

## 8. AtomicRun

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2 \
  --execute
```

`--execute` 会调用 bundle 声明的外部 runtime。执行前先 dry-run 并检查
`task_instance.json`，尤其是模型根目录、冻结 checkpoint、GPU 列表和 cache 目标。

## 9. 评估边界

WAN predictor 只负责生成视频。生成结束后统一调用 Benchmark TaskEvaluator。
WAN bundle 不得：

- 使用自己的 case 清单替代 `plan.jobs`；
- 把模型内部 loss 当作正式 case score；
- 跳过失败 job；
- 修改 Dataset reference；
- 在 OOD case 中伪造 GT；
- 覆盖 scene evaluator 配置。
