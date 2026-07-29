# WAN2.2 Quantity-Embedding 多场景实验报告（2026-07-28）

> 本文冻结记录 quantity-embedding 原始 v1/v2 结果。相同 66 个预测在
> `scene_default_v3` 下的鲁棒主体重评见
> [EVALUATION_V3_20260729.md](EVALUATION_V3_20260729.md)。协议之间的 observed mean
> 不可直接解释为模型性能变化。

## 1. 结论与发布状态

`wan22_ti2v_5b_lora_r32_quantity_embedding_v1` 已完成一次五场景联合训练、66 个
ID/OOD1 job 的推理，以及两套评测协议下的逐 Case 评测。模型实现、训练 checkpoint、
66 个生成视频、Case 级物理曲线和两套 Task 聚合结果都保存在同一个 AtomicRun 中。

最重要的结果边界是：

- 66/66 个预测视频均成功生成；
- canonical `scene_default_v1` 成功评测 45/66，coverage 为 `0.681818`；
- alternate `scene_default_v2` 成功评测 46/66，coverage 为 `0.696970`；
- 两套协议的严格 Task score 都是 `null`，Task status 都是 `partial`；
- v2 的部分覆盖分层宏平均 `0.414271` 只是描述性 observed mean，不是可发布的严格
  Benchmark 分数；
- 本次没有运行匹配的 generic 或 structured-text 对照，因此不能据此声称
  quantity-embedding 优于“不注入物理信息”或“直接拼接物理文本”。

完整 run：

```text
runs_v2/five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728/
```

canonical v1 的加固汇总：

```text
results/five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728/
  canonical_v1_hardened_15b4a0b_20260728/
```

## 2. 落地后的模型设计

用户最初设想中的“数值 MLP + 量纲 MLP + 融合 MLP”被保留，但实现对自由文本抽取、
量纲系统和 token 替换位置作了工程化修正：

```text
Case 首帧 + 原始 prompt + 结构化 physics registry
    │
    ├─ literal audited prompt（供审计）
    └─ UMT5 sentinel prompt（每个物理量恰好一个 token 槽）
              │
          frozen UMT5
              │
SI 数值的 8 维解析特征 ── MLP ───────────────┐
七维 SI 指数 [L,M,T,I,Θ,N,J] ── MLP ───────┼─ concat + fusion MLP
物理量语义类型 ── learned embedding ─────────┘
                                               │
                                        4096 维 z_phys
                                               │
                         替换 sentinel contextual embedding
                                               │
                         WAN2.2 DiT cross-attention + 首帧条件
                                               │
                                           生成视频
```

具体实现位于：

- [`adapter.py`](../../baselines/wan22_quantity_embedding/adapter.py)：结构化字段选择、
  SI 归一化、literal/sentinel prompt 和唯一 token span 审计；
- [`quantity_registry.json`](../../baselines/wan22_quantity_embedding/quantity_registry.json)：
  每个 scene 可消费字段、单位、精度、物理量类型及 primary/derived 角色；
- [`wan22_quantity_model.py`](../../src/physbench/baselines/wan22_quantity_model.py)：
  QuantityEncoder、UMT5 后替换和 checkpoint 合并/加载；
- [`wan22_quantity_train.py`](../../scripts/wan22_quantity_train.py) 与
  [`wan22_quantity_generate_batch.py`](../../scripts/wan22_quantity_generate_batch.py)：
  联合训练和持久化多 GPU 推理。

主要修正如下：

1. 不从自由文本正则猜测 `5 m/s`。物理量由 scene-specific registry 从结构化标注中
   读取，literal prompt 只作为可审计投影；字段、单位、有限值和 token span 不一致时
   fail closed。
2. 量纲使用完整七维 SI 基底 `[L,M,T,I,Θ,N,J]`，而非只支持米、千克、秒。
3. 单一 raw scalar 对跨数量级训练不稳定，因此数值支路使用零值、符号、对数尺度、
   尾数和指数等 8 个确定性特征。
4. 相同量纲不代表相同语义，例如半径和一般长度、角度和无量纲系数；因此加入
   quantity-type embedding。
5. 不在 UMT5 输入端替换不稳定的 subword span。adapter 使用 UMT5 原生 sentinel，
   冻结 UMT5 编码后再用 `z_phys` 替换唯一的 4096 维上下文槽。这样 QuantityEncoder
   的梯度不需要穿过 UMT5-XXL。
6. Classifier-free guidance 只向 positive branch 注入物理编码；negative prompt
   不注入。

QuantityEncoder 的实际拓扑为：

```text
numeric(8)   -> 8→256→256 MLP + LayerNorm
dimension(7) -> 7→128→128 MLP + LayerNorm
type_id      -> Embedding(10, 64)
concat(256+128+64) -> 448→768→4096 MLP + LayerNorm
```

隐藏层使用 SiLU。QuantityEncoder 以 FP32 训练，注入时转换为 UMT5 context dtype；
UMT5 保持冻结，WAN2.2 DiT LoRA 与 QuantityEncoder 联合优化。

## 3. 实验身份与可追溯性

正式训练 run 密封了 bundle `1.0.0` 和 baseline digest。执行时仓库 HEAD 为
`918e9f7`，但该 commit 未被 run manifest 单独密封。当前仓库中的 bundle 已升级到
`1.0.1`，后者的 checkpoint/sampler/prediction 加固不应被追溯描述为旧 run 当时
就具有的能力。

| 字段 | 记录值 |
| --- | --- |
| Run ID | `five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728` |
| Baseline ID | `wan22_ti2v_5b_lora_r32_quantity_embedding_v1` |
| 冻结 bundle | `1.0.0` |
| 执行时 benchmark HEAD（run manifest 未单独密封） | `918e9f78f7618073f9dc782a9ac357f60c82ce86` |
| 加固汇总与 v2 evaluator commit | `15b4a0b15fdd081c4fe85c70e262984057ae5f7c` |
| Dataset | `physics_video_five_scene_v4` / release `4.0.0` |
| Dataset digest | `be5ea8880be3cf8e0d0d316ee025cf02905ef5aeb544f3df5f4b966e5e14782d` |
| Task | `five_scene_finetune_eval_v4` |
| Task digest | `f137d2afe8da1eba8eb99b9ee27c8c7e9288502616016e63ea4d0a6e05d71e37` |
| TaskInstance ID | `five_scene_finetune_eval_v4__wan22_ti2v_5b_lora_r32_quantity_embedding_v1__affc9193926e` |
| TaskInstance digest | `a0f1e35c5c8401e39092e186c0a73f40817a4dfdbc1e36efa4bfb127032826ab` |
| Baseline digest | `68af559e73260ead8995cb1322016bfe6764dc3876fcef572de8fc280aee3423` |
| Deployment digest | `5fce28e43134cb062d759c9c914fe8bf6660dda65105e4dc70a84db313935dca` |
| Canonical run digest | `f2d3927aa5fe0e3b1acc9b6f044a7a1e7c0dd6455b3c1d406a475ec0ab7a6545` |
| WAN base | `WAN2.2-TI2V-5B` |
| DiffSynth commit | `fb337fbb90945ff829de69dbd44ded618f73e889` |
| Base-model asset manifest SHA-256 | `5ca3e1387969fcfe68a7336e062c611e9b99ec6504bccb17edcd16c590984eb7` |
| Training / inference seed | `42 / 42` |

`base_model_assets.json` 逐文件固定了 UMT5、三片 DiT、VAE 和 tokenizer 资产的 size 与
SHA-256。训练由 `/root/miniconda3/envs/dlp/bin/python` 执行；Benchmark 编排与 v2
重评使用 `/root/miniconda3/envs/phybench/bin/python`。canonical hardened summary
给出 `integrity_issues=[]`、`training_acceptance.passed=true`；这说明现有证据内部一致，
不改变评测覆盖不足导致的 `benchmark_score_publishable=false`。

## 4. 多场景训练

### 4.1 数据暴露

| scene | unique train Case | scene-balanced metadata rows | ID jobs | OOD1 jobs |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 27 | 58 | 8 | 5 |
| free_fall | 7 | 58 | 4 | 0 |
| collision_1d | 11 | 58 | 3 | 18 |
| inclined_plane_slide | 58 | 58 | 6 | 16 |
| uniform_circular_motion | 18 | 58 | 2 | 4 |
| **total** | **121** | **290** | **23** | **43** |

每个 scene 被过采样到最大 scene 的 58 行，再应用 `dataset_repeat=4`。因此每个 epoch
的 repeated Dataset 长度是 1160，8 个 rank 对应每 epoch 145 个 optimizer step，
10 epochs 共 1450 step。这里平衡的是 scene exposure，不是 121 个 unique Case 的
等频采样；例如只有 7 个 Case 的 free-fall 会比 58 个 Case 的 incline 获得更高的
单 Case 重复次数。

训练 token audit 覆盖 121/121 个 unique Case 和 481 个物理量：

| scene | audited quantities |
| --- | ---: |
| pendulum | 108 |
| free_fall | 28 |
| collision_1d | 77 |
| inclined_plane_slide | 232 |
| uniform_circular_motion | 36 |

### 4.2 超参数与硬件

| 配置 | 值 |
| --- | --- |
| Optimizer / scheduler | AdamW / ConstantLR |
| Learning rate / weight decay | `1e-4` / `0.01` |
| Epochs / optimizer steps | `10 / 1450` |
| Precision | bf16；QuantityEncoder FP32 |
| LoRA | rank 32，targets `q,k,v,o,ffn.0,ffn.2` |
| Gradient accumulation | 1 |
| Gradient checkpointing | enabled |
| Training media | 121 帧上限、24 FPS、scene aspect-ratio bucket |
| Distributed world size | 8，GPU 0–7 |
| Host GPU | 8 × NVIDIA A100-SXM4-40GB |

run 本身密封了 world size 和 GPU assignment，但没有把 GPU 型号写入不可变 manifest；
上表 GPU 型号来自同一主机在报告日的 `nvidia-smi` 核验。`train.log` 的文件时间跨度约
10,464 秒（2:54:24），包含模型加载、优化和 checkpoint 写入，不是单独插桩的纯训练
耗时。AtomicRun 从 `frozen` 到 `training_complete_or_staged` 的状态间隔为
11,552.825 秒（3:12:32.825）；由于该 bundle 在同一 baseline stage 内完成训练和
推理，这一间隔还包含数据准备与并行推理，不能当作纯训练耗时。

### 4.3 可训练参数与 checkpoint

最终 checkpoint：

```text
artifacts/wan22/checkpoints/step-1450.safetensors
SHA-256: d46f96d183219799c7d09cdeffdcfa4cee4bc562b2a0ddacbf3f75f6b95f4b5e
size:    175,649,752 bytes
```

| 参数组 | tensors | parameters | dtype |
| --- | ---: | ---: | --- |
| WAN DiT LoRA | 600 | 80,609,280 | BF16 |
| QuantityEncoder | 19 | 3,589,888 | FP32 |
| **总计** | **619** | **84,199,168** | mixed |

加固汇总从同一个 `O_NOFOLLOW` file descriptor 读取的 bytes 同时重算 SHA-256、
safetensors layout、finite payload 和 inventory，核验得到 300 个 rank-32 LoRA
pair、预期的 30×10 target topology，以及 19 个 QuantityEncoder tensor，全部通过。
源 bundle `1.0.0` 的 manifest 只声明旧版五字段 inventory；其余 hardened 字段标记为
`derived_not_declared`，而不是伪装成源 run 当时已声明。

### 4.4 训练证据

- 1450/1450 个 loss 均为有限值；
- loss mean `0.108210647`，median `0.060494121`；
- first-100 mean `0.105739985`，last-100 mean `0.084146020`；
- 1450/1450 step 均记录到 19 个 QuantityEncoder gradient tensor，且每步聚合梯度
  L2 严格为正；
- UMT5 gradient tensor count 的最大值为 0。

TensorBoard loss 是每个 optimizer step 的 rank-0 本地 batch loss，不是八卡
all-reduce 后的全局 batch loss。它能证明训练轨迹有限且后段均值下降，不能单独证明
跨 rank 的全局优化损失或模型优于对照。

对应证据位于：

```text
artifacts/wan22/loss_analysis/loss_summary.json
artifacts/wan22/loss_analysis/loss_curve.csv
artifacts/wan22/loss_analysis/loss_curve.png
artifacts/wan22/checkpoints/gradient_audit.json
artifacts/wan22/training_quantity_token_audit.jsonl
```

源 `1.0.0` run 有冻结 sampling plan、optimizer/scheduler state 和 8 个 rank RNG
sidecar，但没有后来 `1.0.1` 新增的 `training_sampling_runtime.json`，RNG sidecar
也没有 sampler generator state。它按 versioned legacy acceptance profile 通过，
不能宣称具有新版的精确 sampler runtime 证据或 state-complete resume 能力。

## 5. 推理产物

66/66 个 job 均为 `status=complete`，由 8 个 persistent model worker 分配到 GPU 0–7。

| 指标 | 值 |
| --- | ---: |
| 完成预测 | 66 / 66 |
| 注入物理量 | 319 |
| `generation_seconds`（各 job worker wall-time）总和 | 7,067.005 秒 |
| per-job worker wall-time | mean 107.076 秒；median 105.357 秒 |
| per-job worker wall-time 范围 | 104.014–118.366 秒 |
| 8-worker 日志 wall span | 约 1,001 秒 |
| 视频帧数 / FPS / duration | 121 / 24 / 5.041667 秒 |
| portrait `480×832` | 23 |
| landscape `832×480` | 43 |
| 预测视频总大小 | 9,464,327 bytes |

`generation_seconds` 由 worker 的 monotonic wall clock 记录，包含输入准备、模型推理、
MP4 编码和 token-audit 写入；它不是 CUDA event 意义下的纯 GPU 时间。66 个 job
并行执行，因此其求和也不是端到端运行 wall time。

所有视频都经 `ffprobe -count_frames` 独立核对为 121 帧。预测文件位于：

```text
predictions/wan22_ti2v_5b_lora_r32_quantity_embedding_v1/<job_id>.mp4
```

每个 prediction 都有独立的 quantity token audit，记录 registry fingerprint、原始值、
SI 值、七维量纲、quantity type、sentinel token ID 和唯一 token span。

## 6. 评测协议、参考视频与分数语义

### 6.1 两套协议并存

| 身份 | canonical v1 | alternate v2 |
| --- | --- | --- |
| Protocol ID | `scene_default_v1` | `scene_default_v2` |
| Fingerprint | `7e40b69368a60282109e4e34fc99d380b186b5e65e537e113e601a3a066fa352` | `84733a984480b4ec8a66b2d31976bd72ddca0a284f525b9d66525436e88dfe88` |
| Evaluator version | 1.1 | 1.2 |
| Decode policy | `legacy_random_seek` | `sequential_forward` |
| Evaluation ID | native canonical | `duration_fix_v2_20260728` |
| Canonical source mutation | N/A | forbidden |

v2 不只是一个“补单摆短视频”的分数补丁。它同时把五个 scene 的视频解码改为顺序
前向解码，并把 evaluator identity 从 1.1 升为 1.2；因此 v2 必须作为独立协议结果
并存，不能静默覆盖 v1。

v2 变体审计记录：

```text
source files:          215
source files digest:   4cb31c4ed18acde5189f2703489a3d85c272a3949e8b72430703951c7f66c5bb
evaluator files:       51
evaluator tree digest: 275832b35878171af90122fe8750d0dc7ed10dbbad2236960b61871958422b94
sealed variant files:  199
artifact files digest: 1930210dbdda4f24de99566390d730835719062b6f0cf1155fb8bdfca748f456
prediction videos:     66
reference assets:      63 unique assets for 66 bindings
```

### 6.2 OOD 参考与物理主体曲线

评估器不要求生成视频与参考视频具有相同分辨率或帧数。每个 scene 先按协议统一物理
时间轴和 letterbox 空间，再提取物理主体 mask/轨迹。OOD Case 若没有同背景 GT，可绑定
相同物理条件的 parent reference；评估目标是物理主体状态，不是背景像素复刻。

每个成功 Case 都保留：

- `result.json`：总分、分项物理指标、quality 和 provenance；
- `per_frame.csv`：对齐后的逐帧观测；
- `physical_subject_iou_curve.png`：生成视频与参考视频物理主体的 IoU 曲线；
- scene-specific trajectory curve（适用时）。

canonical v1 有 45 份 IoU/per-frame artifact，v2 有 46 份。失败 Case 仍保留
`result.json` 和明确的 reason code，不生成虚假的曲线或零分。

### 6.3 三种聚合量不能混用

1. `strict Task score`：只有所有必需 partition 完整覆盖才存在；本次为 `null`。
2. `official observed_mean_score`：先对成功 Case 求 partition observed mean，再对
   partition 求 scene macro，最后对五个 scene 求 Task macro；失败 Case 被排除，因此
   只是部分覆盖描述。
3. `evaluated-only case mean`：所有成功 Case 的普通算术平均，会让 Case 数较多的
   scene/partition 权重更高。

## 7. 总体结果

| 指标 | canonical v1 | alternate v2 | v2−v1 |
| --- | ---: | ---: | ---: |
| Evaluated | 45 | 46 | +1 |
| Error | 20 | 20 | 0 |
| Unavailable | 1 | 0 | −1 |
| Coverage | 0.681818 | 0.696970 | +0.015152 |
| Official observed macro mean | 0.413955199 | 0.414271405 | +0.000316206 |
| Evaluated-only case mean | 0.429668862 | 0.433147034 | +0.003478172 |
| Strict Task score | `null` | `null` | — |
| Task status | `partial` | `partial` | — |

`reporting_status.benchmark_score_publishable=false`。run 的模型推理阶段是 complete，
但 Task 评测是 partial；二者并不矛盾。

## 8. Scene × partition

状态列为 `evaluated / error / unavailable`。Observed 是部分覆盖上的描述性均值；
Strict 只有该 partition 完整覆盖时才非空。

| scene / partition | expected | v1 状态 | v1 coverage | v1 observed | v1 strict | v2 状态 | v2 coverage | v2 observed | v2 strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| collision_1d / ID | 3 | 1/2/0 | 0.333333 | 0.290194365 | `null` | 1/2/0 | 0.333333 | 0.290201624 | `null` |
| collision_1d / OOD1 | 18 | 4/14/0 | 0.222222 | 0.161056544 | `null` | 4/14/0 | 0.222222 | 0.161618388 | `null` |
| free_fall / ID | 4 | 3/1/0 | 0.750000 | 0.362257015 | `null` | 3/1/0 | 0.750000 | 0.362257015 | `null` |
| free_fall / OOD1 | 0 | — | N/A | N/A | N/A | — | N/A | N/A | N/A |
| inclined_plane_slide / ID | 6 | 6/0/0 | 1.000000 | 0.578775892 | 0.578775892 | 6/0/0 | 1.000000 | 0.578775892 | 0.578775892 |
| inclined_plane_slide / OOD1 | 16 | 15/1/0 | 0.937500 | 0.368402160 | `null` | 15/1/0 | 0.937500 | 0.368440314 | `null` |
| pendulum / ID | 8 | 7/0/1 | 0.875000 | 0.566399398 | `null` | 8/0/0 | 1.000000 | 0.568954205 | 0.568954205 |
| pendulum / OOD1 | 5 | 5/0/0 | 1.000000 | 0.483334814 | 0.483334814 | 5/0/0 | 1.000000 | 0.483334814 | 0.483334814 |
| uniform_circular_motion / ID | 2 | 2/0/0 | 1.000000 | 0.374383224 | 0.374383224 | 2/0/0 | 1.000000 | 0.374383224 | 0.374383224 |
| uniform_circular_motion / OOD1 | 4 | 2/2/0 | 0.500000 | 0.592491559 | `null` | 2/2/0 | 0.500000 | 0.592491559 | `null` |

Scene 级 observed macro：

| scene | v1 observed | v2 observed | Δ | v2 strict |
| --- | ---: | ---: | ---: | ---: |
| collision_1d | 0.225625455 | 0.225910006 | +0.000284552 | `null` |
| free_fall | 0.362257015 | 0.362257015 | 0 | `null` |
| inclined_plane_slide | 0.473589026 | 0.473608103 | +0.000019077 | `null` |
| pendulum | 0.524867106 | 0.526144510 | +0.001277404 | 0.526144510 |
| uniform_circular_motion | 0.483437392 | 0.483437392 | 0 | `null` |

v2 使 pendulum ID 和整个 pendulum scene 首次获得完整 strict score，但其它 scene 的
缺失仍使整个 Task score 保持 `null`。

## 9. 失败原因

| reason code | v1 | v2 | 含义 |
| --- | ---: | ---: | --- |
| `insufficient_valid_masks` | 8 | 8 | 有效主体 mask 比例低于 scene quality gate |
| `collision_frame_zero_objects_missing` | 5 | 5 | 碰撞首帧没有识别到预期的三个物体 |
| `collision_striker_missing` | 4 | 4 | 无法识别从目标左侧进入的撞击球 |
| `insufficient_instance_tracks` | 2 | 2 | 圆周运动的实例轨迹有效率低于 0.9 |
| `insufficient_vertical_motion` | 1 | 1 | 自由落体主体垂直位移不足 |
| `insufficient_duration` | 1 | 0 | v1 固定 5 秒要求超过短参考视频时长 |

这 20 个 `error` 是 Case evaluator 的观测/质量门失败，不是推理进程失败；对应的 20 个
预测视频都存在。它们不能被自动记为 0，也不能从 coverage 中隐藏。

## 10. v1 → v2 的变化

唯一状态翻转：

```text
case: pendulum_r2_ltot0130mm_lrope0120mm_r010mm_a020deg
partition: test_id
v1: unavailable / insufficient_duration / score=null
v2: evaluated / score=0.586837856608
```

参考视频为 584 帧、120 FPS，最后帧时间 `4.858333s`。v1 固定要求覆盖 5 秒；v2
采用 reference-bounded 时间轴，实际以 16 Hz 评测 78 帧、`0–4.8125s`。该 Case
的 v2 分项为：

| 分项 | score |
| --- | ---: |
| angle trajectory | 0.481734881 |
| period | 0.846481725 |
| amplitude | 0.652573183 |
| structural consistency | 0.632432645 |
| **final** | **0.586837857** |

另外 11 个原本 evaluated 的 Case 出现很小的浮点漂移，来自五个 scene 统一切换为
`sequential_forward` 解码；45 个共同成功 Case 的总变化为 `+0.002826935`，平均
`+0.000062821`。最大变化是 collision `v01985` 的 `+0.002646666`。没有其它 status
或 reason-code 变化，也没有修改预测视频或冻结 Case 清单。

## 11. 三臂对照状态

| Baseline | 本次是否完成匹配训练/评测 | 可报告结果 |
| --- | --- | --- |
| WAN2.2 generic | 否 | 未运行 |
| WAN2.2 structured text | 否 | 未运行 |
| WAN2.2 quantity embedding | 是，single seed 42 | partial observed；strict Task score `null` |

因此本报告只能说明 quantity-embedding baseline 已可训练、可推理并得到一组部分覆盖
结果，不能回答它是否比另外两臂更优。公平对照还需要固定相同 Dataset digest、
TaskInstance、WAN base、LoRA rank/target、训练步数、seed、生成参数和 evaluator，
并分别独立训练三臂。

## 12. 产物与摘要校验值

canonical v1：

```text
evaluation/task_result.json
  SHA-256 636b882f24aca9f3f1bb91514ae4bc5195581fb0ce92082844bcc7a082f97db0
evaluation/case_results.jsonl
  SHA-256 e03eb3026bcfc6f1e3e9aebfe8d962b508a41a7ce734112acb6387a8ef8672ff
evaluation/cases/<job_id>/
```

加固汇总：

```text
results/.../canonical_v1_hardened_15b4a0b_20260728/quantity_run_summary.json
  SHA-256 dbe67c04af7ff89243f9b322cf3c5ef82b318df084bc061e576edaad8a9aac39
results/.../canonical_v1_hardened_15b4a0b_20260728/quantity_run_summary.md
  SHA-256 a26bb4e4575c27532807676cc299a51a03fb998bd6e8ad61553216e75eab7467
```

alternate v2：

```text
reevaluations/scene_default_v2/
  84733a984480b4ec8a66b2d31976bd72ddca0a284f525b9d66525436e88dfe88/
    duration_fix_v2_20260728/
      evaluation/task_result.json
        SHA-256 3a970511152a21cb457673c588bc70e32a338bd042ed6e34f20da697a5789a60
      evaluation/case_results.jsonl
        SHA-256 e2b88e83dd33048fcb49189a0876c6864dc0bf0b9ea43eea1398370a123c3fb8
      evaluation/cases/<job_id>/
```

重新生成 canonical v1 汇总时应使用新的 summary 目录，避免覆盖既有证据：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/summarize_quantity_run.py \
  --run-dir runs_v2/five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728 \
  --output-dir results/five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728/<new_summary_id>
```

重评必须使用新的 `evaluation-id`；已有 `duration_fix_v2_20260728` 不允许覆盖：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate \
  --run-dir runs_v2/five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728 \
  --protocol-id scene_default_v2 \
  --evaluation-id <new_evaluation_id>
```

## 13. 解释限制

- 只有 seed 42，没有跨 seed 均值、方差或置信区间。
- generic/structured-text 对照未运行，不能做算法优劣结论。
- strict Task score 为 `null`；observed mean 不能冒充 Benchmark score。
- `free_fall/OOD1` 没有规划 Case，应报告 N/A。
- 当前 OOD1 不是统一的纯环境变化：collision 同时包含未见速度；部分 pendulum OOD
  含训练未见角度；circular 还引入第二物体和新增条件字段。
- scene-balanced exposure 不等于 unique Case 均衡。
- collision train 仅占该 scene 34.4%，全局 train 占 View A 64.7%，并非统一 75%。
- quantity registry 是面向当前五场景的 closed-world 白名单，只消费明确登记的字段；
  未登记的物理字段不会被消费；已登记的 required 字段缺失、单位不匹配或值非法会
  fail closed。它不是任意自然语言单位解析器。
- 物理编码只通过 DiT 文本 cross-attention 进入模型；当前没有显式轨迹条件、
  物理方程 residual loss 或仿真器约束。
- evaluator 质量门对生成失败与分割/观测失败不做自动因果区分；reason code 表示无法
  获得满足协议的物理状态，不能直接解释为某一种模型错误。
- v1 与 v2 的小幅分数漂移来自解码协议变化，因此两个结果必须连同 protocol
  fingerprint 一起引用。

## 附录 A：66 个 Case 的逐项结果

`—` 表示没有合法分数或 reason code。表格按 `scene_id → partition → case_id` 排序。

| scene | partition | case_id | v1 status | v1 score | v1 reason | v2 status | v2 score | v2 reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | --- |
| collision_1d | test_id | `collision_r2_large_steel_large_steel_large_steel_v02229` | error | — | collision_frame_zero_objects_missing | error | — | collision_frame_zero_objects_missing |
| collision_1d | test_id | `collision_r2_medium_steel_medium_steel_medium_steel_v02818` | evaluated | 0.290194365 | — | evaluated | 0.290201624 | — |
| collision_1d | test_id | `collision_r2_small_steel_small_steel_small_steel_v07845` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_glass_marble_glass_marble_glass_marble_v04374` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_glass_marble_glass_marble_glass_marble_v05716` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_glass_marble_glass_marble_glass_marble_v07545` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_large_steel_medium_steel_small_steel_v01165` | error | — | collision_frame_zero_objects_missing | error | — | collision_frame_zero_objects_missing |
| collision_1d | test_ood1 | `collision_r2_large_steel_medium_steel_small_steel_v01313` | error | — | collision_frame_zero_objects_missing | error | — | collision_frame_zero_objects_missing |
| collision_1d | test_ood1 | `collision_r2_large_steel_medium_steel_small_steel_v02174` | error | — | collision_striker_missing | error | — | collision_striker_missing |
| collision_1d | test_ood1 | `collision_r2_large_steel_medium_steel_small_steel_v02358` | error | — | collision_striker_missing | error | — | collision_striker_missing |
| collision_1d | test_ood1 | `collision_r2_large_steel_medium_steel_small_steel_v03033` | error | — | collision_striker_missing | error | — | collision_striker_missing |
| collision_1d | test_ood1 | `collision_r2_small_steel_glass_marble_glass_marble_v01509` | error | — | collision_frame_zero_objects_missing | error | — | collision_frame_zero_objects_missing |
| collision_1d | test_ood1 | `collision_r2_small_steel_glass_marble_glass_marble_v02179` | error | — | collision_striker_missing | error | — | collision_striker_missing |
| collision_1d | test_ood1 | `collision_r2_small_steel_glass_marble_glass_marble_v02935` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_small_steel_glass_marble_glass_marble_v05212` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_small_steel_glass_marble_glass_marble_v05533` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| collision_1d | test_ood1 | `collision_r2_small_steel_medium_steel_large_steel_v01985` | evaluated | 0.314687796 | — | evaluated | 0.317334462 | — |
| collision_1d | test_ood1 | `collision_r2_small_steel_medium_steel_large_steel_v03759` | error | — | collision_frame_zero_objects_missing | error | — | collision_frame_zero_objects_missing |
| collision_1d | test_ood1 | `collision_r2_small_steel_medium_steel_large_steel_v04097` | evaluated | 0.122266964 | — | evaluated | 0.122266964 | — |
| collision_1d | test_ood1 | `collision_r2_small_steel_medium_steel_large_steel_v06332` | evaluated | 0.141177619 | — | evaluated | 0.140776517 | — |
| collision_1d | test_ood1 | `collision_r2_small_steel_medium_steel_large_steel_v07389` | evaluated | 0.066093798 | — | evaluated | 0.066095611 | — |
| free_fall | test_id | `freefall_r2_l_h080cm` | evaluated | 0.469664854 | — | evaluated | 0.469664854 | — |
| free_fall | test_id | `freefall_r2_m_h080cm` | error | — | insufficient_vertical_motion | error | — | insufficient_vertical_motion |
| free_fall | test_id | `freefall_r2_s_h080cm` | evaluated | 0.041913148 | — | evaluated | 0.041913148 | — |
| free_fall | test_id | `freefall_r2_xl_h080cm` | evaluated | 0.575193042 | — | evaluated | 0.575193042 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bggreen_img_0358` | evaluated | 0.262743687 | — | evaluated | 0.262743687 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bggreen_img_0359` | evaluated | 0.304043974 | — | evaluated | 0.304043974 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bgoil_img_0326` | evaluated | 0.710321438 | — | evaluated | 0.710321438 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bgoil_img_0327` | evaluated | 0.856420508 | — | evaluated | 0.856420508 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bgwhite_img_0287` | evaluated | 0.763245828 | — | evaluated | 0.763245828 | — |
| inclined_plane_slide | test_id | `incline_r1_a38deg_bgwhite_img_0288` | evaluated | 0.575879918 | — | evaluated | 0.575879918 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a32deg_bgblack_img_0341` | evaluated | 0.090184147 | — | evaluated | 0.090247787 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a32deg_bgblack_img_0342` | evaluated | 0.057507275 | — | evaluated | 0.057517277 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a32deg_bgblack_img_0343` | evaluated | 0.104401952 | — | evaluated | 0.104412203 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a32deg_bgblack_img_0344` | evaluated | 0.209218994 | — | evaluated | 0.209219324 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a35deg_bgblack_img_0375` | evaluated | 0.289355839 | — | evaluated | 0.289355839 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a35deg_bgblack_img_0376` | evaluated | 0.350246102 | — | evaluated | 0.350246102 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a35deg_bgblack_img_0377` | evaluated | 0.339647165 | — | evaluated | 0.339647165 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a35deg_bgblack_img_0378` | evaluated | 0.371871668 | — | evaluated | 0.371871668 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a41deg_bgblack_img_0385` | evaluated | 0.496922769 | — | evaluated | 0.496922769 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a41deg_bgblack_img_0386` | evaluated | 0.584266468 | — | evaluated | 0.584804478 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a41deg_bgblack_img_0387` | evaluated | 0.624214757 | — | evaluated | 0.624214757 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a41deg_bgblack_img_0388` | evaluated | 0.395432404 | — | evaluated | 0.395432404 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a44deg_bgblack_img_0390` | evaluated | 0.583619186 | — | evaluated | 0.583613689 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a44deg_bgblack_img_0392` | error | — | insufficient_valid_masks | error | — | insufficient_valid_masks |
| inclined_plane_slide | test_ood1 | `incline_r1_a44deg_bgblack_img_0393` | evaluated | 0.509970672 | — | evaluated | 0.509926235 | — |
| inclined_plane_slide | test_ood1 | `incline_r1_a44deg_bgblack_img_0394` | evaluated | 0.519173006 | — | evaluated | 0.519173006 | — |
| pendulum | test_id | `pendulum_r1_ltot0210mm_lrope0200mm_r010mm_a020deg` | evaluated | 0.480612525 | — | evaluated | 0.480612525 | — |
| pendulum | test_id | `pendulum_r1_ltot0240mm_lrope0230mm_r010mm_a020deg` | evaluated | 0.570648182 | — | evaluated | 0.570648182 | — |
| pendulum | test_id | `pendulum_r1_ltot0280mm_lrope0270mm_r010mm_a020deg` | evaluated | 0.758938215 | — | evaluated | 0.758938215 | — |
| pendulum | test_id | `pendulum_r1_ltot0305mm_lrope0295mm_r010mm_a020deg` | evaluated | 0.657543524 | — | evaluated | 0.657543524 | — |
| pendulum | test_id | `pendulum_r2_ltot0110mm_lrope0100mm_r010mm_a020deg` | evaluated | 0.438051558 | — | evaluated | 0.438051558 | — |
| pendulum | test_id | `pendulum_r2_ltot0130mm_lrope0120mm_r010mm_a020deg` | unavailable | — | insufficient_duration | evaluated | 0.586837857 | — |
| pendulum | test_id | `pendulum_r2_ltot0155mm_lrope0145mm_r010mm_a020deg` | evaluated | 0.555158044 | — | evaluated | 0.555158044 | — |
| pendulum | test_id | `pendulum_r2_ltot0180mm_lrope0170mm_r010mm_a020deg` | evaluated | 0.503843738 | — | evaluated | 0.503843738 | — |
| pendulum | test_ood1 | `pendulum_ltot0110mm_lrope0100mm_r010mm_a010deg_ood01` | evaluated | 0.279456867 | — | evaluated | 0.279456867 | — |
| pendulum | test_ood1 | `pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood02` | evaluated | 0.468649441 | — | evaluated | 0.468649441 | — |
| pendulum | test_ood1 | `pendulum_ltot0130mm_lrope0120mm_r010mm_a030deg_ood03` | evaluated | 0.468063879 | — | evaluated | 0.468063879 | — |
| pendulum | test_ood1 | `pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood04` | evaluated | 0.615362131 | — | evaluated | 0.615362131 | — |
| pendulum | test_ood1 | `pendulum_ltot0155mm_lrope0145mm_r010mm_a020deg_ood05` | evaluated | 0.585141754 | — | evaluated | 0.585141754 | — |
| uniform_circular_motion | test_id | `circular_r1_silver04cm_img_0369` | evaluated | 0.639450472 | — | evaluated | 0.639450472 | — |
| uniform_circular_motion | test_id | `circular_r1_wood04cm_img_0381` | evaluated | 0.109315976 | — | evaluated | 0.109315976 | — |
| uniform_circular_motion | test_ood1 | `circular_r1_silver02cm_wood06cm_img_0392` | evaluated | 0.529779795 | — | evaluated | 0.529779795 | — |
| uniform_circular_motion | test_ood1 | `circular_r1_silver02cm_wood08cm_img_0393` | error | — | insufficient_instance_tracks | error | — | insufficient_instance_tracks |
| uniform_circular_motion | test_ood1 | `circular_r1_silver06cm_wood02cm_img_0394` | evaluated | 0.655203323 | — | evaluated | 0.655203323 | — |
| uniform_circular_motion | test_ood1 | `circular_r1_silver08cm_wood02cm_img_0395` | error | — | insufficient_instance_tracks | error | — | insufficient_instance_tracks |
