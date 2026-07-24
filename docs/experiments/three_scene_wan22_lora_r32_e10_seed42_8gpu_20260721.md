# WAN2.2 三场景 8 卡联合 LoRA 微调与 ID/OOD1 推理

**Run ID：** `three_scene_wan22_lora_r32_e10_seed42_8gpu_20260721`  
**任务：** Type-1 / View-A，`finetune_and_eval`  
**状态：** 完成；训练 410/410 steps，推理 38/38 成功，0 失败。  
**目标：** 从 WAN2.2-TI2V-5B 基础权重重新开始，在单摆、自由落体、一维对心碰撞训练集上联合 LoRA 微调，然后统一测试全部 ID 与 OOD1 case。

## 1. 冻结数据与任务

| Scene | 原子训练 case | 均衡后 metadata 行 | ID | OOD1 |
|---|---:|---:|---:|---:|
| 单摆 `pendulum` | 27 | 27 | 8 | 5 |
| 自由落体 `free_fall` | 7 | 27 | 4 | 0 |
| 一维碰撞 `collision_1d` | 11 | 27 | 3 | 18 |
| 合计 | 45 | 81 | 15 | 23 |

- 使用与上一轮相同的 `cases.jsonl`、View-A 划分和 Type-1 任务语义；manifest SHA-256 为 `5df494d190dac7c4683bc2637045b1101f18cfbc59a24f604ce3f7819d511301`。
- 场景均衡策略为 `oversample_each_scene_to_largest`：自由落体和碰撞只在 baseline 私有 metadata 中确定性循环采样到各 27 行，不改写源数据。
- 原视频保持原始分辨率、帧率和总帧数；WAN baseline 私有缓存执行 24 fps 采样、等比例 fit + 黑边 padding，并保留每条 case 的有效 `4n+1` 帧数，上限 121。
- 自由落体 8 倍慢放已在 Benchmark 数据侧恢复为真实物理时间，本轮 baseline 不再做额外倍速恢复。
- OOD1 覆盖单摆外观变化与碰撞外观/组合变化；自由落体当前无 OOD1 case。

## 2. 训练配置

| 项目 | 配置 |
|---|---|
| 基础模型 | WAN2.2-TI2V-5B |
| 算法 | FlowMatch SFT + LoRA |
| 条件 | 视频首帧 + case 的 TI2V physics prompt |
| LoRA | rank 32；`q,k,v,o,ffn.0,ffn.2` |
| Optimizer / Scheduler | AdamW / ConstantLR |
| LR / Weight decay | `1e-4` / `0.01` |
| 精度 | BF16 + gradient checkpointing |
| GPU / Batch | 8×A100-SXM4-40GB；每卡 batch 1；gradient accumulation 1；global batch 8 |
| Repeat × Epoch | 4 × 10 |
| 优化步数 | 41 steps/epoch，共 410 steps |
| Checkpoint | 每 41 steps 保存，10 份 adapter 全部保留 |
| 模型侧规格 | 480×832；24 fps；每条 5–121 帧且满足 `4n+1` |
| Seed | 42 |
| DiffSynth | commit `fb337fbb90945ff829de69dbd44ded618f73e889` |

本轮 `initial_lora_checkpoint=null`，即从基础 WAN2.2 权重重新微调，没有续接上一轮 adapter。8 卡使每 epoch 的 optimizer step 数从双卡方案的 162 降为 41，但每 step 全局 batch 从 2 增至 8；10 epoch 的样本曝光量基本等价，末尾差异来自 DistributedSampler padding。

训练阶段约为 2026-07-21 03:13:29–04:02:43 UTC，耗时约 49 分 13 秒。没有发生 OOM、NCCL 错误、NaN/Inf、进程退出或 checkpoint 写入失败。

## 3. Loss

- 记录点：410/410，有限值比例 `1.0`。
- 总体均值 / 中位数：`0.091299 / 0.038954`。
- 范围：`0.000000–0.519609`。
- 前 100 step 均值：`0.099681`；后 100 step 均值：`0.074303`；比值 `0.745406`，后段下降约 25.5%。

| Epoch | Count | Mean | Median | Min | Max | Std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | 0.097417 | 0.037699 | 0.002633 | 0.437000 | 0.123008 |
| 2 | 41 | 0.100508 | 0.052771 | 0.001293 | 0.428075 | 0.119473 |
| 3 | 41 | 0.129772 | 0.049286 | 0.000000 | 0.519609 | 0.153254 |
| 4 | 41 | 0.084914 | 0.042977 | 0.001316 | 0.509526 | 0.111178 |
| 5 | 41 | 0.093057 | 0.039637 | 0.000000 | 0.503416 | 0.126713 |
| 6 | 41 | 0.088032 | 0.033032 | 0.000915 | 0.422117 | 0.116562 |
| 7 | 41 | 0.090330 | 0.041178 | 0.000000 | 0.395931 | 0.117818 |
| 8 | 41 | 0.078718 | 0.047440 | 0.002322 | 0.428783 | 0.089695 |
| 9 | 41 | 0.097971 | 0.032411 | 0.000999 | 0.477986 | 0.125145 |
| 10 | 41 | 0.052274 | 0.029147 | 0.000792 | 0.294210 | 0.068065 |

原始 loss 不单调是 FlowMatch 随机 diffusion timestep、不同 scene/帧长和样本难度共同造成的；应结合 rolling mean、EMA 和生成质量判断。本轮后 100 step 均值明显低于前 100 step，未观察到数值发散。

Loss 图：`artifacts/wan22/loss_analysis/loss_curve.png`  
逐步 CSV：`artifacts/wan22/loss_analysis/loss_curve.csv`  
统计 JSON：`artifacts/wan22/loss_analysis/loss_summary.json`

## 4. 推理配置与完成情况

- 使用最终 `step-410.safetensors`、case 首帧、case 自带 physics prompt、seed 42。
- 50 inference steps，CFG 5.0，LoRA alpha 1.0，tiled generation。
- 8 个常驻 worker 分配到 GPU 0–7，worker 内复用模型，避免每条 case 重复加载。
- 38/38 完成、0 失败；8 个 worker 返回码均为 0。

| Scene / Partition | 完成数 |
|---|---:|
| `pendulum/test_id` | 8 |
| `pendulum/test_ood1` | 5 |
| `free_fall/test_id` | 4 |
| `collision_1d/test_id` | 3 |
| `collision_1d/test_ood1` | 18 |

生成视频审计：38 个文件均可由 ffprobe 读取，全部为 480×832、24 fps；帧数分布为 9–121 帧，时长 0.375–5.042 秒。短视频来自相应 case 的真实物理时长，baseline 没有循环或强制补齐到 121 帧。

## 5. 指标边界

本轮已完成训练和视频生成，但 `summary.json` 中 `scored_jobs=0`、`mean_score=null`。这是因为 CommonSense、Prediction 和 VisualJudgment 的 VLM、分割与感知评价器仍是占位接口，不是生成失败，也不应解释为零分。视频完成状态以 `predictions.jsonl` 和推理 worker summary 为准。

## 6. 产物

- Run 根目录：`runs/three_scene_wan22_lora_r32_e10_seed42_8gpu_20260721/`
- 最终 adapter：`artifacts/wan22/checkpoints/step-410.safetensors`
- 全部 adapter：`step-41.safetensors` 至 `step-410.safetensors`，间隔 41 steps
- 训练配置：`frozen_baseline.json`、`training_stage.json`、`artifacts/wan22/checkpoints/run.env`
- 训练日志 / TensorBoard：`artifacts/wan22/checkpoints/train.log`、`tensorboard_log/`
- Loss：`artifacts/wan22/loss_analysis/`
- 生成视频：`predictions/`
- 逐 job 配置：`jobs/`
- 推理日志与汇总：`artifacts/wan22/inference_workers/`
- Benchmark 状态：`run.json`、`predictions.jsonl`、`summary.json`、`report.md`
