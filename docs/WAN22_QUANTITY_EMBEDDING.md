# WAN2.2 物理量编码 Baseline

## 1. 实验身份与目标

```text
Baseline ID: wan22_ti2v_5b_lora_r32_quantity_embedding_v1
Base model:  WAN2.2-TI2V-5B
Task:        five_scene_finetune_eval_v4
Dataset:     physics_video_five_scene_v4 / View A
```

目标是检验：相对于“不使用结构化物理量”和“把物理量直接写入 prompt”，显式编码数值、
量纲和物理量语义是否能改善生成运动的物理一致性。它仍是 I2V：输入为 Case 首帧、
`case.text.prompt`，并从 `case.physics[annotated=true]` 中消费由 registry 明确筛选的
物理量子集；它不会把全部结构化标注都送入模型。registry 会记录每个字段是 primary
还是 derived，并不假设所选字段彼此统计独立。当前仅支持 `finetune_eval` Task family。

## 2. 从设想到可训练实现

实现没有从自由文本中用正则猜测 `5 m/s`。这会遇到同一数值多次出现、单位别名、
token 跨度不稳定和文本/结构化标注冲突。Bundle-local
`quantity_registry.json` 改为按 scene 白名单读取结构化字段，校验：

- `annotated=true` 且数值有限；
- 字段名和单位与 registry 一致；
- required 字段不缺失；
- 渲染精度、物理量类型和来源角色固定。

Adapter 同时生成两份文本：

```text
audited_prompt:
  ... the striker initial velocity is 5.0000 m/s ...

model prompt:
  ... the striker initial velocity is <extra_id_0> ...
```

`audited_prompt` 用于人类审计；模型 prompt 用 UMT5 原生 sentinel 保证一个物理量恰好
占一个 token 槽。每个 sentinel 在 tokenization 后必须唯一出现，否则立即失败。

注入位置选为：

```text
model prompt
  -> frozen UMT5
  -> 用 z_phys 替换 sentinel 的 4096 维 context 向量
  -> WAN DiT text projection / cross-attention
```

这比在 UMT5 输入 embedding 前替换更适合当前工程：物理 encoder 的梯度无需穿过
UMT5-XXL 全部层，显存和实现复杂度显著降低；同时消除了 `5`、`m`、`/`、`s` 被切成
多个 subword 时的跨度歧义。代价是 `z_phys` 本身不再经过 UMT5 上下文化；上下文仍由
其余文本 token 提供。这是 Baseline v1 的明确设计边界。

## 3. 物理编码

单位先转换到 SI canonical value。数值不直接把一个 raw scalar 喂给 MLP，而使用不依赖
全数据统计量的 8 维解析特征：

```text
is_zero
sign
log1p_abs_si
signed_log1p_abs_si
tanh_si
signed_base10_mantissa
clipped_base10_exponent
inverse_one_plus_abs_si
```

这样同时保留零值、符号、数量级、尾数和小量信息，降低不同物理量数值尺度相差数个数量级
时单一 MLP 的优化难度。

量纲采用完整七维 SI 指数坐标，而不是只用米、千克、秒：

```text
[L, M, T, I, Θ, N, J]
```

例如 `m/s -> [1,0,-1,0,0,0,0]`，`N -> [1,1,-2,0,0,0,0]`。角度会先按
`π/180` 转成 rad 数值，但量纲坐标仍为全零。由于量纲不能区分“角度/摩擦系数”，也不能
区分“半径/一般长度”，实现额外加入 64 维 quantity-type embedding。

当前网络为：

```text
numeric(8)   -> MLP 8->256->256 + LayerNorm       = e_num
dimension(7) -> MLP 7->128->128 + LayerNorm       = e_dimension
type_id      -> Embedding(10, 64)                  = e_type
[e_num; e_dimension; e_type]
             -> MLP 448->768->4096 + LayerNorm     = z_phys
```

隐藏层激活为 SiLU。QuantityEncoder 使用 FP32 计算，输出在注入时转换到 UMT5 context
dtype。负面 prompt 不注入物理量。

## 4. 联合训练与推理

训练冻结 WAN base 与 UMT5，联合优化：

- WAN DiT LoRA，rank 32；
- target modules：`q,k,v,o,ffn.0,ffn.2`；
- QuantityEncoder。

FlowMatch SFT 使用 AdamW、ConstantLR、学习率 `1e-4`、weight decay `0.01`、
10 epochs、dataset repeat 4、bf16、gradient checkpointing、seed 42。五个 scene
共同训练，并把每个 scene 过采样到最大 scene 的 metadata 行数。最终 checkpoint
必须恰好包含当前 WAN2.2-TI2V-5B 拓扑的 300 个 rank-32 LoRA A/B pair（600 个
LoRA tensor，30 blocks × 每 block 10 个 target），以及 19 个
`pipe.quantity_encoder.*` tensor。缺 pair、额外 target、错误 rank、错误 shape 或
非有限权重都会在融合前失败。推理拒绝只有 LoRA 的旧 checkpoint。

训练 DataLoader 的 shuffle 显式绑定由 trainer seed 初始化的
`torch.Generator`。sampler seed 同时写入 `training_sampling_plan.json`、
`checkpoints/training_args.json`、`checkpoints/run.env` 与
`checkpoints/training_sampling_runtime.json`，从而区分公共 sampler seed 和
`TRAIN_SEED + rank` 的进程随机数策略。

每个推理 worker 在加载模型前必须读取 schema-2 `checkpoint.json`，核对 job 中的
checkpoint 路径、文件 size 与 SHA-256；manifest 缺失、字段缺失或字节不一致均
fail closed。

先配置本机部署：

```bash
cd /root/Steven/physics_video_benchmark
cp baselines/wan22_quantity_embedding/baseline.local.example.json \
  baselines/wan22_quantity_embedding/baseline.local.json
```

在 Git-ignored 的 `baseline.local.json` 中填写 WAN 工程、模型根目录、模型 Python、
GPU 和 Accelerate 配置。Benchmark 编排环境是
`/root/miniconda3/envs/phybench`；实际 WAN 进程使用 local 配置中的 Python。

验证和 dry-run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_quantity_embedding_v1

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --run-id wan22_quantity_embedding_v1_viewa_seed42_dryrun \
  --output-root runs_v2
```

完整训练、推理和评估必须使用新的 run ID：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --run-id wan22_quantity_embedding_v1_viewa_seed42 \
  --output-root runs_v2 \
  --execute
```

不要给需要结果的运行添加 `--stop-after-training`；该选项只保存训练产物并把 inference
标记为 staged。

## 5. 与两个 WAN 对照的公平比较

推荐三臂矩阵：

| Baseline | 结构化 physics | 模型侧形式 |
| --- | --- | --- |
| `..._v3_generic` | 不消费 | 原始 Case prompt |
| `..._v3_physics` | 消费 | literal structured text |
| `..._quantity_embedding_v1` | 消费 | sentinel slot 中的 `z_phys` |

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --matrix-id wan22_viewa_conditioning_seed42 \
  --output-root runs_v2 \
  --execute
```

三臂必须固定 Dataset digest、Task/Case、首帧、train/eval split、seed、WAN base、
LoRA rank/targets、优化器、scene balancing、生成参数和 evaluator。三者必须独立训练，
不可让某一臂复用另一臂的训练后 LoRA。Quantity registry 与 structured-text 模板选择
相同的、经筛选的字段子集，以及相同的单位和精度。

这仍不是严格等参数量消融：quantity 臂增加了 QuantityEncoder 参数，structured-text
臂没有；prompt token 数也因 representation 不同而改变。发布时应同时记录 checkpoint
inventory 和训练成本，不应只报告分数。

## 6. View A 的已知限制

当前 View A 冻结分布为：

| scene | train | ID | OOD1 | 纳入 View A |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 27 | 8 | 5 | 40 |
| free_fall | 7 | 4 | 0 | 11 |
| collision_1d | 11 | 3 | 18 | 32 |
| inclined_plane_slide | 58 | 6 | 16 | 80 |
| uniform_circular_motion | 18 | 2 | 4 | 24 |
| **total** | **121** | **23** | **43** | **187** |

- `selection_policy` 的设计目标是：ID 使用训练已见环境和未见物理数值，OOD1 使用
  训练未见环境和已见物理数值。但当前冻结列表并没有对所有 scene 严格实现这个目标；
- collision 的 18 个 OOD1 Case 中，`striker_initial_velocity` 也都是训练未见值，
  因而属于“环境和物理数值同时变化”的混合分布；
- pendulum 的两个 20° OOD1 Case 使用了训练中未出现的 `initial_angle=20°`；
- circular 的 OOD1 引入训练中不存在的第二物体及 `object_2_orbit_radius` 条件字段，
  不只是背景/颜色变化；
- 27 个另行识别为“数值和环境同时未见”的 Case 未进入 View A，当前 Task 未启用
  OOD2，但这并没有消除上述已进入 OOD1 的混合变化；
- `free_fall` 没有可用 OOD1，因此不能声称五个 scene 都具有 OOD1 证据；
- collision train 仅占本 scene 34.4%，全局 train 占 View A 64.7%，没有达到统一 75%；
- circular ID 只有 2 个 Case，当前官方 Task 也只有一个训练/推理 seed 42。

因此最终表格必须把 `test_ood1` 当作当前 Dataset 的冻结分区名称，不能一概解释为
“只改变环境”。当前结果适合固定 split 下的首轮对照，不足以给出纯因素归因、跨 seed
置信区间或强泛化结论。

## 7. 产物与验收

```text
runs_v2/<run_id>/
├── task_instance/manifest.json
├── adaptations/case_adaptations.jsonl
├── artifacts/wan22/
│   ├── checkpoint.json
│   ├── base_model_assets.json
│   ├── training_spec.json
│   ├── training_sampling_plan.json
│   ├── training_media_audit.jsonl
│   ├── training_quantity_token_audit.jsonl
│   ├── checkpoints/
│   │   ├── *.safetensors
│   │   ├── quantity_encoder_spec.json
│   │   ├── training_args.json
│   │   ├── run.env
│   │   ├── training_sampling_runtime.json
│   │   ├── gradient_audit.json
│   │   └── training_state_latest/
│   └── inference_quantity_token_audits/<job_id>.json
├── jobs/<job_id>.json
├── predictions/
├── predictions.jsonl
└── evaluation/
    ├── case_results.jsonl
    ├── task_result.json
    └── cases/<job_id>/
```

正式成绩的最低验收条件：

```text
predictions.jsonl:
  每个 planned job 恰好有一条 status == "complete" 的 prediction
  每条 status == "evaluated" 的 Case 结果必须对应 complete prediction

evaluation/task_result.json:
  status == "complete"
  coverage == 1.0
  score != null
  integrity_issues == []
  protocol.fingerprint == component_fingerprints.evaluation_protocol
  breakdown / by_scene / score 与合法 case_results 经官方 Task 聚合器重算的结果完全一致

artifacts/wan22/checkpoint.json:
  inventory.tensor_count == 619
  inventory.lora_pair_count == 300
  inventory.lora_tensor_count == 600
  inventory.lora_rank == 32
  inventory.quantity_encoder_tensor_count == 19
  上述 inventory、LoRA topology/shape 与 tensor finite 状态必须从实际
  safetensors bytes 严格重算并与 manifest 一致
  checkpoint size / SHA-256 / baseline identity 必须一致
  若 save_optimizer_state=true，optimizer/scheduler 与所有 rank RNG sidecar 必须齐全
  state manifest 必须与最终 step、world size、checkpoint 文件名一致，RNG 文件名/数量按 rank 核对
  optimizer/scheduler 的 SHA-256 由 checkpoint manifest 锚定；RNG SHA-256 仅记录当前文件摘要，
  没有外部 manifest 摘要锚，也不反序列化验证其语义内容

artifacts/wan22/checkpoints/gradient_audit.json:
  sample_count == expected_total_optimizer_steps
  positive_quantity_gradient_count == sample_count
  每个 sample 的 quantity_gradient_tensor_count == 19
  text_encoder_gradient_tensor_count_max == 0

artifacts/wan22/loss_analysis/loss_summary.json:
  first_step == 1
  last_step == recorded_steps == expected_total_optimizer_steps
  finite_fraction == 1.0
  loss_curve.csv 必须逐步覆盖 1..expected_total_optimizer_steps，并与 summary 统计一致
```

每条训练/推理量值还应能在 token audit 中追溯到 registry fingerprint、SI value、
量纲、type ID、sentinel token ID 和唯一 token span；每个训练 Case 和每个完成推理的
job 都必须恰好有一份与 sealed TaskInstance 输入一致的审计记录。

AtomicRun 进入 `complete`、`inference_incomplete` 或 `failed` 终态后，可用只读汇总器
生成逐 scene、逐 partition、逐 job 结果、官方聚合结果以及训练审计索引。缺失项会保持
`null`/`N/A`，不会被补成 0：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/summarize_quantity_run.py \
  --run-dir runs_v2/<run_id> \
  --output-dir results/<run_id>
```

若执行异常只留下 `state.json: stage=failed` 而尚未来得及写 `run.json`，汇总器会明确
标记并使用 frozen identity/component fingerprints 构造“失败状态身份投影”；该投影
只用于部分故障报告，自身会产生 integrity issue，绝不会被视为正式运行身份或成绩。

汇总器中的 scene×partition、by-scene 和 overall 描述性统计同时给出两种权重：
`job_micro_mean` 对成功评估的推理 job 等权；Case macro 则先在 Case 内对可用 seed
求均值，再对 Case 等权。Benchmark 官方 by-scene 分数是 partition macro，Task
分数再对 scene 做 macro。汇总器使用 Benchmark 自身的 Task 聚合器从合法
`case_results` 独立重算 `breakdown`、`by_scene` 和 `score`，并与冻结结果逐字段核对。
逐 job 输出保留 evaluator、metrics、quality、artifacts（包括 IoU 曲线）和 provenance。

唯一权威的发布开关是
`reporting_status.benchmark_score_publishable`；它只有在 sealed TaskInstance、
全部 planned prediction 完成、Case/prediction 跨记录一致性、协议 fingerprint、
官方聚合以及所有训练证据全部通过时才为 `true`。部分运行仍会列出 coverage、失败
job ID 和 reason code，但其严格官方分数不可发布。若 reevaluate 使用了与 AtomicRun
冻结 fingerprint 不同的修订协议，则必须作为单独标识的 alternate-protocol 结果报告，
不能通过该发布开关。checkpoint、恢复状态、loss、gradient 或训练/推理 token audit
缺失或未通过验收时，汇总器会写入 `integrity_issues`，不会以“无问题”掩盖缺失证据。
当前 TensorBoard loss 是每个 optimizer step 的 rank-0 本地 batch loss，不是八卡
loss 的 all-reduce 均值；它适合检查训练轨迹和有限性，不应解释为全局 batch loss。

## 8. 结果记录模板

截至真正完成上述 `--execute` 运行前，所有分数必须标记为 `PENDING`，不能把 dry-run
的 `planned`、部分运行的 `observed_mean_score` 或 reference self-test 当作模型成绩。

### 8.1 运行身份

| 字段 | 记录值 |
| --- | --- |
| Git commit | `PENDING` |
| Dataset digest | `PENDING` |
| TaskInstance digest | `PENDING` |
| Baseline deployment digest | `PENDING` |
| DiffSynth commit | `PENDING` |
| Base-model asset manifest digest | `PENDING` |
| Checkpoint path / SHA-256 / size | `PENDING` |
| LoRA / QuantityEncoder tensor and parameter counts | `PENDING` |
| GPU 型号、数量与总训练时间 | `PENDING` |
| training / inference seed | `42 / 42` |
| coverage / integrity issues | `PENDING` |

### 8.2 Quantity Baseline 的 View A 分数

从 `evaluation/task_result.json` 的 `breakdown` 和 `by_scene` 原样抄录：

| scene | ID jobs | ID score | OOD1 jobs | OOD1 score | scene macro score |
| --- | ---: | ---: | ---: | ---: | ---: |
| pendulum | 8 | `PENDING` | 5 | `PENDING` | `PENDING` |
| free_fall | 4 | `PENDING` | 0 | N/A | `PENDING` |
| collision_1d | 3 | `PENDING` | 18 | `PENDING` | `PENDING` |
| inclined_plane_slide | 6 | `PENDING` | 16 | `PENDING` | `PENDING` |
| uniform_circular_motion | 2 | `PENDING` | 4 | `PENDING` | `PENDING` |
| **Task macro** | **23** | 不跨 scene 微平均 | **43** | 不跨 scene 微平均 | `PENDING` |

### 8.3 三臂对照

| Baseline | pendulum | free fall | collision | incline | circular | Task score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| structured text | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| quantity embedding | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

每个 scene score 是其所需 partition 的宏平均；Task score 再对五个 scene 宏平均。只有
66/66 正式 jobs 全部成功评估时才填写正式 Task score。失败或缺失时保留
`score=null`，另列 coverage 和失败原因。
