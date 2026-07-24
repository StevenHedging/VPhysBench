# WAN2.2 三场景联合 LoRA 微调与 ID/OOD 评测

**Run ID：** `three_scene_wan22_lora_r32_e10_seed42_20260720`  
**状态：** 正式训练进行中；本文档会在训练与推理结束后补齐实测 loss 和生成结果。  
**目标：** 在单摆、自由落体、一维对心碰撞的 View-A 训练集上联合微调 WAN2.2-TI2V-5B，并对全部可用 `test_id` 与 `test_ood1` case 做首帧条件视频生成。

## 1. 数据冻结

| Scene | 原子训练 case | 场景均衡后 metadata 行 | ID 测试 | OOD1 测试 |
|---|---:|---:|---:|---:|
| 单摆 `pendulum` | 27 | 27 | 8 | 5 |
| 自由落体 `free_fall` | 7 | 27 | 4 | 0 |
| 一维碰撞 `collision_1d` | 11 | 27 | 3 | 18 |
| 合计 | 45 | 81 | 15 | 23 |

- 原始视频保持各自的分辨率、帧率、总帧数，不在 Benchmark 数据侧统一规格。
- WAN baseline 的私有缓存才执行等比例缩放加 padding、24 fps 时间戳采样和 `4n+1` 帧约束；不循环、不补帧。
- 自由落体的 8 倍慢放已在数据侧通过无重编码时间戳恢复为真实时间，baseline 读取的 `reference.mp4` 已是物理时间。
- 场景均衡使用确定性过采样：以单摆 27 条为目标，自由落体与碰撞在 metadata 中循环重复到 27 行；不会复制或改写源视频。
- 单摆 5 个 OOD1 为首帧外观变化（背景 2、小球材质 2、支架 1），没有真实续帧；它们只允许常识/物理参考类评测，不冒充真实外观参考视频。
- 自由落体当前没有 OOD1 case，因此本轮不能报告其 OOD 分数。

## 2. 微调配置

| 项目 | 配置 |
|---|---|
| 基础模型 | WAN2.2-TI2V-5B |
| 训练算法 | FlowMatch SFT + LoRA |
| 输入条件 | 每条视频首帧 + 含物理参数的 TI2V prompt |
| 模型侧空间规格 | 480×832，等比例 fit + 黑边 padding |
| 模型侧时间规格 | 24 fps；每条 case 保留自己的有效 `4n+1` 帧数，上限 121 |
| LoRA rank / modules | 32；`q,k,v,o,ffn.0,ffn.2` |
| Optimizer / Scheduler | AdamW / ConstantLR |
| 学习率 / Weight decay | `1e-4` / `0.01` |
| Batch | 每卡 1；2×A100-SXM4-40GB；gradient accumulation 1 |
| 精度 | BF16 + gradient checkpointing |
| Repeat × Epoch | 4 × 10 |
| 预计步数 | 162 steps/epoch，共 1620 optimizer steps |
| Checkpoint | 每 162 步保存；全部 adapter 保留 |
| 训练 seed | 42 |
| DiffSynth | `2.0.17`，commit `fb337fbb90945ff829de69dbd44ded618f73e889` |

正式训练前已完成同规格双卡反向冒烟：两条 121 帧样本各卡一步，约 8.4 秒，loss `0.0631745`，成功写出 LoRA checkpoint 与 TensorBoard scalar。该数值仅用于运行链路验证，不属于正式训练曲线。

## 3. 推理配置

- 每个 case 使用首帧条件、case 自带 physics prompt、seed 42。
- 50 inference steps，CFG 5.0，LoRA alpha 1.0，tiled generation。
- 正式训练结束后一次加载一个模型到每张 GPU，8 卡并行处理 38 个冻结作业；每个 worker 内复用模型，避免每条 case 重复加载权重。

## 4. 待补充实测结果

训练完成后补充：

- 1620 点 raw loss、25/100 步滚动均值、25 步滚动中位数与 EMA 曲线；
- 每 epoch 的 loss 均值、中位数、标准差、极值；
- 前 100 / 后 100 步均值及其比值；
- 10 个中间 adapter 与最终 adapter 路径；
- 38 个推理作业的完成状态、逐 scene/ID/OOD 覆盖和耗时；
- 当前占位 metrics 的适用性边界，以及后续接入正式物理评测器的位置。

## 5. 产物位置

- 冻结配置、数据与任务：`runs/three_scene_wan22_lora_r32_e10_seed42_20260720/`
- 训练缓存与审计：`artifacts/wan22/dataset/`、`training_media_audit.jsonl`、`training_sampling_plan.json`
- Adapter / 日志：`artifacts/wan22/checkpoints/`
- Loss 分析：`artifacts/wan22/loss_analysis/`
- 生成视频：`predictions/`
- 并行推理日志：`artifacts/wan22/inference_workers/`
