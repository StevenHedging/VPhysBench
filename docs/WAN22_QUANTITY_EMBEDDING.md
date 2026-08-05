# WAN2.2 物理量编码 Baseline

## 1. 实验身份与目标

```text
Baseline ID: wan22_ti2v_5b_lora_r32_quantity_embedding_v1
Current bundle:       1.0.1
Frozen source bundle: 1.0.0
Execution-time HEAD:  918e9f7 (not separately sealed by the run manifest)
Base model:  WAN2.2-TI2V-5B
Source run:  five_scene_finetune_eval_v4 / scene_default_v1
Current Task: five_scene_finetune_eval_v11 (Task schema 4.0) / scene_default_v10
Dataset:     physics_video_six_scene_v11 / View A
```

目标是检验：相对于“不使用结构化物理量”和“把物理量直接写入 prompt”，显式编码数值、
量纲和物理量语义是否能改善生成运动的物理一致性。它仍是 I2V：输入为 Case 首帧、
`case.text.prompt`，并从 `case.physics[annotated=true]` 中消费由 registry 明确筛选的
物理量子集；它不会把全部结构化标注都送入模型。当前V2 registry只选择V11中
`annotated=true`的独立量，并保留value、unit和symbol。当前仅支持`finetune_eval`
Task family。

2026-07-28 完成的 source run 密封了 bundle `1.0.0` 和 baseline digest，并保留当时
的 v4 Task 与 `scene_default_v1` native evaluation identity。执行时仓库 HEAD 为
`918e9f7`，但该 commit 未被 run manifest 单独密封。当前官方Task使用V11 Dataset、
Task schema 4.0与`scene_default_v10`；
该 source run 已另存一份不可变的 `scene_default_v2` alternate reevaluation，没有
回写或伪装成 canonical v1。完整实验身份、训练证据、两套协议和 66 条逐 Case 结果见
[`experiments/WAN22_QUANTITY_EMBEDDING_20260728.md`](experiments/WAN22_QUANTITY_EMBEDDING_20260728.md)。
下文命令引用当前`tasks/official`文件，因此新运行会生成V11 Task identity和bundle 1.0.1
身份，而不是复用历史 v4/v1 source identity。

## 2. 从设想到可训练实现

实现没有从自由文本中用正则猜测 `5 m/s`。这会遇到同一数值多次出现、单位别名、
token 跨度不稳定和文本/结构化标注冲突。Bundle-local
`quantity_registry_v2.json`按scene白名单读取结构化字段，校验：

- `annotated=true`且数值有限、非负；
- 字段名、单位和symbol与registry一致；
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

以下 sampler runtime 约束适用于当前 bundle `1.0.1` 创建的新 run。训练 DataLoader
的 shuffle 显式绑定由 trainer seed 初始化的
`torch.Generator`。sampler seed 同时写入 `training_sampling_plan.json`、
`checkpoints/training_args.json`、`checkpoints/run.env` 与
`checkpoints/training_sampling_runtime.json`，从而区分公共 sampler seed 和
`TRAIN_SEED + rank` 的进程随机数策略。训练开始前还会强制
`len(repeated_dataset) % world_size == 0`，否则直接失败，避免 Accelerate
`even_batches=True` 用 epoch 开头的样本补齐各 rank。runtime audit 会记录 Dataset
长度、每 rank 每 epoch 样本数和 `BatchSamplerShard` 策略。每个 rank 的 RNG sidecar
包含显式 sampler generator state；但因为没有保存当前 DataLoader iterator/
permutation 的位置，也没有自动 resume 入口，这些文件仅是 diagnostic snapshot，
不能宣称为精确可恢复的 “state-complete checkpoint”。

已完成的历史 source run 冻结的是 bundle `1.0.0`：它有 sampling plan、
optimizer/scheduler state 与每 rank RNG sidecar，但没有
`training_sampling_runtime.json`，sidecar 也没有 sampler generator state。该 run
按 versioned legacy acceptance profile 通过；不能把 1.0.1 的 sampler runtime
证据或 state-complete resume 能力追溯归因给它。

每个推理 worker 在加载模型前必须读取 schema-2 `checkpoint.json`，核对 job 中的
checkpoint 路径、文件 size 与 SHA-256；manifest 缺失、字段缺失或字节不一致均
fail closed。真正加载时会拒绝 checkpoint 叶节点 symlink，在平台支持时以
`O_NOFOLLOW` 打开文件，只从同一个 file descriptor 完整读取一次，然后对这份 bytes
计算 SHA-256，并把同一个 bytes 对象交给 `safetensors.torch.load`。校验后不会再按
路径重新打开文件，因此瞬时 swap-and-restore 不能让解析器消费另一份权重。
persistent batch worker 还会逐 job 比较所有只加载一次的配置：
checkpoint、manifest、runtime/model-base、QuantityEncoder 配置、LoRA alpha 及未知的
load-time generation 选项；任一不一致都会在加载模型前拒绝整个 batch，不会静默复用
首个 job 的配置。

同一 bytes 信任边界会分配一份完整的已认证 checkpoint buffer。当前正式 rank-32
checkpoint 为 175,649,752 bytes（167.5 MiB）。由于
`safetensors.torch.load(bytes)` 返回独立的 tensor backing，解析瞬间二者会短暂共存；
一次实际无 GPU 解析测得每 worker 增量峰值 RSS 约 304 MiB，8 个 worker 若同时达到
峰值约 2.38 GiB，而不是只有 167.5 MiB 的输入 buffer。解析返回后会立即显式释放输入
buffer，校验及 LoRA fusion 完成后释放其余临时 CPU state，不增加逐视频生成的常驻
内存。

这里的 SHA-256 是 AtomicRun 内部一致性检查，不是外部真实性锚或数字签名。若某个
主体能同时重写 checkpoint 与同目录的 `checkpoint.json`，它可以生成新的自洽文件
对；需要对抗这种发布者级篡改时，必须由 run 目录之外的可信系统签名或固定 manifest
digest。

训练在 backward 前跨 rank 检查 loss finite，在 optimizer step 前检查
QuantityEncoder gradient finite；最终 safetensors 会验证完整 payload 布局并用浮点
指数位扫描拒绝 NaN/Inf，之后才写 schema-2 checkpoint manifest。

先配置本机部署：

```bash
cd /root/Steven/physics_video_benchmark
test -e baselines/wan22_quantity_embedding/baseline.local.json || \
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
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --run-id wan22_quantity_embedding_v1_viewa_seed42_dryrun \
  --output-root runs_v2
```

完整训练、推理和评估必须使用新的 run ID：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/11.0.0/dataset.json \
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
  --dataset datasets/releases/11.0.0/dataset.json \
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
- 另有 27 个新增 Case 因重复 held-out 条件或未被 View A 的固定采样选中而未纳入；
  其中部分同时含未见物理数值和环境的条件，可留待未来 OOD2，但当前 Task 未启用
  OOD2。这并没有消除上述已进入 OOD1 的混合变化；
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
  inventory.lora_tensor_count == 600
  inventory.quantity_encoder_tensor_count == 19
  严格重算结果中的 lora_pair_count == 300、lora_rank == 32
  上述 inventory、LoRA topology/shape 与 tensor finite 状态始终从实际
  safetensors bytes 严格重算
  checkpoint size / SHA-256 / baseline identity 必须一致
  严格重算结果中的 safetensors_layout_verified == true
  严格重算结果中的 finite_payload_verified == true
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

Checkpoint inventory 的声明协议由冻结的 Baseline 版本决定，而且只在
`run.json`、`frozen/baseline.json` 与 sealed TaskInstance 三处 Baseline ID/版本
完全一致后选择。`1.0.0` 的历史 manifest 必须声明
`tensor_count`、`parameter_count`、`lora_tensor_count`、
`quantity_encoder_tensor_count` 与 `dtype_tensor_counts`；严格解析器新增得到的
pair/rank/topology/layout/finite 字段会记录为 `derived_not_declared`。
`1.0.1` 则必须在 manifest 中完整声明全部十个 hardened 字段。两种协议下，
manifest 已声明的每个字段都必须与实际 bytes 的严格重算值一致；未知版本、三处身份
不一致、字段缺失或值不一致都会 fail closed。

汇总器不会先按路径 hash、再按路径重开 checkpoint。它通过 `O_NOFOLLOW` descriptor
一次读取 immutable buffer，SHA-256、size、header、layout、finite scan 与 inventory
全部绑定这同一份 bytes，并在解析结束后复核 descriptor 和路径身份；因此
swap-and-restore 不能把 A 文件的摘要与 B 文件的 inventory 拼接成可发布证据。

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
  --output-dir results/<run_id>/<summary_id>
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
prediction 的身份规则按冻结 Baseline bundle version 解释：历史 `1.0.0` 未要求
重复输出 `scene_id`，汇总时可由 sealed plan/job/case 唯一确定；若历史记录带有
`scene_id`，其值仍须一致。自 `1.0.1` 起，所有 prediction 都必须显式携带并匹配
`scene_id`；未知版本或三份冻结身份不一致时不会退回宽松规则。Case evaluation
同样始终严格要求 `scene_id`，避免跨 scene 误聚合。
当前 TensorBoard loss 是每个 optimizer step 的 rank-0 本地 batch loss，不是八卡
loss 的 all-reduce 均值；它适合检查训练轨迹和有限性，不应解释为全局 batch loss。

## 8. 2026-07-28 正式实验快照

完整报告及 66 条逐 Case 结果：
[`experiments/WAN22_QUANTITY_EMBEDDING_20260728.md`](experiments/WAN22_QUANTITY_EMBEDDING_20260728.md)。

### 8.1 记录身份

| 字段 | 记录值 |
| --- | --- |
| Frozen bundle | `1.0.0` |
| Execution-time benchmark HEAD（run manifest 未单独密封） | `918e9f7` |
| Hardened summary / v2 evaluator commit | `15b4a0b` |
| Dataset digest | `be5ea8880be3cf8e0d0d316ee025cf02905ef5aeb544f3df5f4b966e5e14782d` |
| TaskInstance digest | `a0f1e35c5c8401e39092e186c0a73f40817a4dfdbc1e36efa4bfb127032826ab` |
| Baseline / deployment digest | `68af559e...e3423 / 5fce28e4...935dca` |
| DiffSynth commit | `fb337fbb90945ff829de69dbd44ded618f73e889` |
| Base-model asset manifest SHA-256 | `5ca3e1387969fcfe68a7336e062c611e9b99ec6504bccb17edcd16c590984eb7` |
| Checkpoint | `step-1450.safetensors` / `d46f96d...b5e` / 175,649,752 bytes |
| Trainable parameters | LoRA 80,609,280 + QuantityEncoder 3,589,888 = 84,199,168 |
| Training | 8 GPU、10 epochs、1450 steps、seed 42 |
| Predictions | 66/66 complete |

AtomicRun 从 frozen 到 baseline stage 完成的间隔为 11,552.825 秒，但它包含准备、训练
和该 bundle 的并行推理，不是纯训练耗时。`train.log` 文件跨度约 10,464 秒，也包含
模型加载和 checkpoint 写入。GPU world size 与 assignment 已冻结；主机报告日核验为
8 × NVIDIA A100-SXM4-40GB，GPU 型号本身没有写入不可变 run manifest。

### 8.2 当前 `scene_default_v2` alternate reevaluation

下表的 `observed` 是部分覆盖描述性均值，不是 strict score：

| scene | ID evaluated/expected | ID observed | OOD1 evaluated/expected | OOD1 observed | scene observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| pendulum | 8/8 | 0.568954 | 5/5 | 0.483335 | 0.526145 |
| free_fall | 3/4 | 0.362257 | 0/0 | N/A | 0.362257 |
| collision_1d | 1/3 | 0.290202 | 4/18 | 0.161618 | 0.225910 |
| inclined_plane_slide | 6/6 | 0.578776 | 15/16 | 0.368440 | 0.473608 |
| uniform_circular_motion | 2/2 | 0.374383 | 2/4 | 0.592492 | 0.483437 |
| **Task** | **20/23** | 不跨 scene 微平均 | **26/43** | 不跨 scene 微平均 | **observed 0.414271** |

总计 46/66 成功评测，coverage `0.696970`，20 个 evaluator error，strict Task score
`null`，`benchmark_score_publishable=false`。canonical v1 为 45/66、
observed `0.413955`、strict `null`；v2 是不同 fingerprint 的并存结果，不能覆盖 v1。

### 8.3 三臂对照状态

| Baseline | pendulum | free fall | collision | incline | circular | Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| structured text | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| quantity embedding | 0.526145 observed | 0.362257 observed | 0.225910 observed | 0.473608 observed | 0.483437 observed | 0.414271 observed；strict `null` |

本次只有 single seed 42。由于 generic 与 structured-text 对照尚未运行，不能声称
quantity-embedding 更优；OOD1 也包含若干环境与物理数值同时变化的混合因素。
