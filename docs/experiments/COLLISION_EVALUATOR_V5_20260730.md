# 碰撞评估器 v5：开放世界实体匹配与任意 N-body 审计

日期：2026-07-30

> 历史快照：文中的全局外置可视化路径记录当时实现。当前实现统一写入所属
> AtomicRun/reevaluation 的 `evaluation/visualizations/`，不要照搬旧路径。

## 1. 结论与发布边界

`scene_default_v5` 已把公共开放世界实体层接入 `collision_1d`，实现：

- Case 结构化 physics 驱动的任意 N 个碰撞实体；
- manifest-directed 角色观测与 residual object discovery；
- 因果实体追踪、带 null hypothesis 的逐帧一对一匹配和身份关联惩罚；
- 对新增、复制、消失、ID swap 和 overflow 的按秒 exposure 门控；
- 连续位置距离与 scene-specific N-body 物理评分；
- reference-exposure anti-abstention 与 residual discovery fail-closed；
- prediction 短视频前缀可评、尾段按 missing exposure 计分；
- 本地哈希产物与 NVMe 外置过程可视化。

协议身份：

```text
protocol_id          = scene_default_v5
protocol_fingerprint = 93703d6afdf8bbdea86b69d8d8a427653c68030bd7660f3801341e3b2b6209f6
```

标准协议配置（`sam2.device=auto`）的碰撞 evaluator 身份：

```text
id          = collision_1d_open_world_nbody
version     = 2.2
fingerprint = 686b705e426f43d29acdc745a95b765fe249bd48628c86f5167f40f851caa156
```

本轮真实审计显式传入 `--device cuda`。设备覆盖进入 evaluator config identity，因此
审计报告记录的运行配置 fingerprint 是：

```text
f6959f4877dd3dce85bb05fe84c0240253a40ebe5aacf3486d66d8a80de7250b
```

协议 fingerprint、标准 evaluator fingerprint 和 CUDA 审计 evaluator fingerprint
是不同 identity，不得混称。正式 Task-facing 主 metric 是
`scene_subject_state_similarity`；碰撞分解同时以
`collision_1d_open_world_similarity` alias 输出。

2.2 fingerprint payload 还显式冻结：

```text
entity_contract = manifest_v1
observer        = open_world_v1.2
physics         = nbody_v1.2
```

v5 目前仍是 shadow protocol：只升级碰撞 scene；单摆、自由落体、斜面下滑和匀速圆周
运动继续使用 v3 evaluator type。它不覆盖官方 `scene_default_v3`，也不覆盖
`scene_default_v4`；不同 fingerprint 下的分数不能混入同一个 leaderboard。

## 2. v4 遗留问题

v4 已把冻结 View B 碰撞 reference 的可观测性提高到 32/32，但其评分契约仍以固定三个
角色为中心：

```text
striker + target_1 + target_2
```

这无法完整处理以下生成错误：

- 画面凭空多出一个或多个球；
- 一个球被复制为多个相似主体；
- 某个主体中途消失；
- 角色身份在遮挡或接触后互换；
- Case 原本就是 2 球、4 球或其他数量；
- 两球不相交但距离很近时，IoU 与相距很远同为 0。

v5 没有在 v4 三角色逻辑上继续叠加特例，而是把基数、存在性和身份关联提升为公共实体
契约，再由碰撞 adapter 提供 N-body 物理内容。

## 3. 公共开放世界实体层

实现位置：

```text
src/physbench/evaluation/common/entities/
├── contracts.py
├── manifest.py
├── observer.py
├── scoring.py
└── timeline.py
```

### 3.1 Case entity manifest

Manifest materializer 从每条 Case 的结构化 physics 生成稳定实体清单。碰撞球记录：

```text
entity_id
role_id
entity_class = ball
lifecycle
condition_anchor.initial_order_index
mass
radius
initial_velocity
```

`N` 由该清单决定，不是 scene 常量。碰撞 evaluator 要求至少两个球，顺序索引必须唯一
覆盖 `0..N-1`，物理量单位和取值必须满足契约。无效 manifest 是 reference/data
问题，返回 `unavailable`，不会由 evaluator 猜测补齐。

Directed SAM2 prompt 数量、reference tracks、prediction expected channels、质量数组及
N-body 状态维度都与 manifest 对齐。因此代码路径支持 2、3、4 或其他合法 N，也不假设
只有一个主动球。

### 3.2 真实秒时间网格

Reference 与 prediction 共用一组物理时间戳。`CommonTimeGrid` 以相邻时间戳中点作为
cell 边界；规则采样时等价于 trapezoidal 权重，所有 cell 权重之和精确等于实际评估
时长。

以下量都按秒积分，而不是按帧计数：

```text
expected existence
prediction existence
missing / false exposure
identity association exposure
overflow exposure
```

这使评分不依赖原视频 FPS 或采样点密度，也让“额外球持续 2 秒”具有明确、可比较的
含义。

### 3.3 开放世界观测

Prediction 使用两条通道：

```text
manifest-directed expected tracks
+ residual motion/Hough proposals
→ per-frame deduplication
→ causal tracking
→ one-to-one entity assignment
```

Residual channel 负责发现 expected channels 没有解释的实体。Track ID 只由当前及历史
观测确定，不以未来帧回填或重命名。超过最大轨迹容量的候选成为可计分的 overflow
exposure，不会抛出 evaluator error。

## 4. 证据等级与分层匹配

视觉 residual 可能是真实额外球，也可能是反光、轨道标记或 motion ghost。v5 为候选
冻结四档证据及正式 exposure 权重：

| Evidence tier | 权重 | 正式行为 |
| --- | ---: | --- |
| `PARTICIPANT` | 1.00 | 参与匹配、presence 和 N-body |
| `INDEPENDENT_SALIENT` | 0.50 | 参与匹配和加权 presence |
| `TENTATIVE` | 0.25 | 参与低等级匹配和保守 presence |
| `AMBIGUOUS` | 0.00 | 仅诊断，不参与正式匹配或惩罚 |

每帧 assignment 分层执行：

1. 先以最高 evidence weight 对仍未匹配的 GT entity 做 Hungarian 一对一匹配；
2. 移除已匹配 GT，再依次让较低等级候选补位；
3. `AMBIGUOUS` 完全不进入正式 Hungarian。

每一层 Hungarian 都包含 null hypothesis。协议中的
`minimum_match_position_similarity = 0.1` 是可接受边的下限；低于阈值的候选不再被
强制匹配，而是同时记为 missing reference 与 extra prediction。审计帧保留
`rejected_candidate_matches`，包括被拒边的位置相似度和阈值，因此“离得很远但仍被
配成同一对象”不能绕过基数惩罚。

这避免弱 Hough 圆先抢占 participant 的 GT 槽。逐帧审计同时保留：

```text
formal residual track IDs
ambiguous candidate track IDs
formal prediction cardinality
participant prediction count
raw prediction candidate count
overflow count
```

碰撞 residual 使用下列校准：

- Hough-only 少于 3 个支持帧时为 `AMBIGUOUS`；
- 持续 Hough-only 最高仍为 `TENTATIVE`，静态持续不能将其升级为 N-body participant；
- motion 至少需要 3 个真正 motion-supported frames；
- motion 累计位移至少达到该轨迹中位半径的 2 倍，否则只为 `TENTATIVE`；
- 只有完整 participant residual 进入 N-body；弱候选仅产生按权重的 presence
  exposure。

Prediction residual discovery 的内部失败采用 fail-closed 空 prediction observation，
并记录稳定 degradation code `prediction_residual_observation_failed`。它不会退回只
看 directed tracks 的高分路径；因此 detector failure 不能被当作“未发现额外物体”的
正证据。

## 5. 完整性门控与 N-body 内容

### 5.1 完整性层

每帧 GT entity 与 prediction track 使用一对一匹配。位置相似度由参考主体尺度与画布
尺度归一化的连续距离核给出，不要求两个 mask 相交；因此“很近但不相交”会明显高于
“相距很远”，不会出现 IoU 的全零悬崖。

完整性层报告 presence、soft localization、association、exposure precision/recall 和
GOSPA decomposition。正式 gate 为：

```text
integrity_gate =
    presence_detection_accuracy × association_accuracy
```

Missing、extra、复制体、ID swap 和 overflow 进入不可稀释的 gate。连续轨迹位置不
进入正式 gate，`soft localization` 也只保留为诊断。

Missing 会同时影响两个语义不同的层：Gate 评价开放世界的基数和身份完整性；N-body
content 评价完整 reference 时间轴上的物理状态。若物体消失，完整性层会记录 missing，
相应物理 component 也必须在缺失时段记 0，否则生成器可以主动删除困难帧，只用少量
幸存帧获得虚高条件分。两层针对的是不同可 hack 维度。

### 5.2 任意 N-body 状态

碰撞 adapter 在 reference 冻结轨道轴上提取：

| Component | N-body 权重 | 内容 |
| --- | ---: | --- |
| `track_position` | 0.30 | 每一帧对应实体的连续位置 |
| `contact_graph` | 0.25 | 所有实体 pair 的接触状态与事件 |
| `velocity` | 0.20 | 逐实体速度 |
| `momentum` | 0.15 | 系统动量变化相对 reference |
| `nonpenetration` | 0.10 | 过度穿透 |

Contact graph 遍历全部 pair，不硬编码球数、相邻对或唯一主动球。确认的额外
participant 也进入 N-body；因为数据没有它的真实质量，目前以 reference 质量中位数
近似。

2.2 的 anti-abstention 规则是：

- partially matched participant 的每个 detection 在 expected 或 residual N-body
  channel 中守恒一次；
- position、velocity、momentum 和 nonpenetration 的正式分数以真实秒 GT/reference
  exposure 为分母，missing 区间贡献 0；同时另报只看共同帧的 `conditional_score`
  供诊断；
- prediction coverage 不再删除 GT contact events；
- prediction 空 contact graph 只有在完整 pair exposure 下才可认证为正确，否则保守
  低分。

### 5.3 最终 Case 组合

Scene content 是：

```text
weighted_geometric_mean(
    nbody_physics: 0.60,
    matched_subject_shape: 0.20,
    matched_subject_appearance: 0.20
)
```

最终分数：

```text
case = integrity_gate × content
```

Task 聚合读取正式主 metric `scene_subject_state_similarity`。Case result 中的
`collision_1d_open_world_similarity` 是碰撞专用 alias，用于公开
`integrity_gate × content` 的完整分解；它不是 v3
`collision_1d_state_similarity` 的同名替代物。

Shape 和 appearance 只比较当帧已匹配实体的 union。完整 reference/prediction 主体
union IoU 继续输出为 `physical_subject_iou_curve.png`，沿用 Jensen 对照图的审计作用，
但不充当正式 content gate。未匹配额外物体已由完整性层惩罚，不再被 appearance 重复
惩罚。

## 6. 媒体与失败语义

Reference 仍必须满足协议最小时长并覆盖 reference-bounded 时间轴；否则为
`unavailable`。

Prediction 短于 reference 时：

1. 已解码前缀正常比较；
2. 视频结束后或解码损坏后的共同时间格标记 `available=false`；
3. 不重复末帧、不插值、不补 GT 首帧；
4. unavailable 尾段通过预期实体 missing exposure 得到有限低分；
5. Case 保持 `evaluated`，不能借 evaluator failure 逃避难例。

Random seek 越界若发生，只返回中性 unavailable cell，不执行越界读取。Prediction
directed observation、residual discovery、N-body 或 subject comparison 的内部局部
失败均有空观测/有限零分 fallback，并记录稳定 degradation code；其中 residual
discovery failure 必须按上一节的 fail-closed 语义处理，不能把 directed-only 当作完整
开放世界观测。Reference 契约错误与真正未知 evaluator 异常仍分别保留为
`unavailable` 和 `error`。

## 7. 最终 r7：GT-self 与 WAN 审计

报告：

```text
/mnt/nvme1/physics_video_benchmark/evaluator_audits/
collision_v5_null_assignment_r7_20260730/audit_report.json
```

该报告共 4 条记录，4/4 `evaluated`、0 `error`、0 `unavailable`：

| Case / variant | Case score | Gate | Content | Missed | False | N-body | Bodies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glass `v04374` GT-self | `0.9880578251` | `0.9880578253` | `0.9999999998` | `0 s` | `0.1484375 s` | `1.0` | 3 |
| steel `v06332` GT-self | `0.9693811074` | `0.9693811075` | `0.9999999999` | `0 s` | `0.3671875 s` | `1.0` | 3 |
| glass `v04374` WAN generic | `0.0250737646` | `0.1395090585` | `0.1797285771` | `7.2734375 s` | `7.140625 s` | `0.0674320397` | 8 |
| steel `v06332` WAN generic | `0.1203513978` | `0.3696488527` | `0.3255830415` | `2.8125 s` | `2.765625 s` | `0.1787889683` | 5 |

两条 self-check 的 `track_position`、`contact_graph`、`velocity`、`momentum` 和
`nonpenetration` 均为 1。总分未精确达到 1 的原因不是主体或 N-body self mismatch，
而是保守 residual exposure：

- glass：短 motion proposals 被降为 tentative，Hough-only 为 ambiguous；
- steel：静态 motion/Hough 伪物体不再成为第四个 N-body body，但仍以 0.25 权重产生
  tentative exposure。

这说明证据校准已阻止弱静态圆形证据污染 N-body 基数，同时保留少量透明的保守惩罚。
它也意味着 v5 当前的 detector/evaluator identity 不满足“像素自比必为数学上的 1”
这一更强性质；该偏差已作为已知限制保留，不能在结果报告中隐藏。

两条 WAN generic prediction 的 SHA-256 分别为：

```text
v04374  4104378620a384c90eab092cfd8fb5f9f15bda08297ea5eddd91fb3ae960789f
v06332  1c86c25cade6c068ed7890ef236b7045fc0572bc2bf5b6b0a7def19fc577999d
```

null assignment 把远处候选转成显式 missed + false exposure；同时 GT exposure
denominator 使位置、速度、动量和非穿透项不能只在少数共同帧上取得条件高分。

## 8. 最终 r8：Quantity 多球/少球反例审计

报告：

```text
/mnt/nvme1/physics_video_benchmark/evaluator_audits/
collision_v5_quantity_r8_20260730/audit_report.json
```

2/2 `evaluated`、0 `error`、0 `unavailable`：

| Case | Prediction SHA-256 | Case score | Gate | Content | Missed | False | N-body | Bodies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glass `v04374` | `56575b25b2b88f01d78f2fca568228434c2acf095bfca34c55a349c96a94cba3` | `0.0039100888` | `0.0218833164` | `0.1786789882` | `8.578125 s` | `14.03125 s` | `0.0690681146` | 11 |
| steel `v06332` | `e4e59609e32d721355bccf35779652dfef6777e6a634a86fe209199477fe8540` | `0.0690306009` | `0.3970986940` | `0.1738373910` | `3.90625 s` | `0.90625 s` | `0.0643423432` | 4 |

Glass prediction 是已知多球反例。它持续出现额外物体/复制体，并且远候选不再被强制
塞入 GT 槽；因此 presence、association、contact graph 和 track position 都保持低分，
真实 extra penalties 没有被 GT-self false-positive 校准一并消除。

Steel prediction 是已知少球反例，累计 `3.90625 s` missing exposure。它仍得到有限
低分并保持 `evaluated`，而不是 evaluator error；reference-exposure denominator 又使
主体缺失不能通过 coverage 漏洞逃避物理评分。

## 9. 外置可视化与复现入口

默认外置根：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_visualizations
```

可用环境变量覆盖：

```text
PHYSBENCH_VISUALIZATION_ROOT
```

仓库入口：

```text
visualizations
  -> /mnt/nvme1/physics_video_benchmark/evaluation_visualizations
```

每个外置 Case bundle 包含：

```text
open_world_nbody_audit.mp4
open_world_nbody_audit.json
```

Run 或 audit 的本地 Case 目录保留：

```text
collision_v5_visualization_manifest.json
per_frame.csv
physical_subject_iou_curve.png
entity_position_curve.png
object_cardinality_timeline.png
```

Manifest 记录外部路径、仓库链接路径、大小和 SHA-256。外置 bundle 是未封印、
best-effort 的诊断资产；写入失败会在 manifest 中记录，但不改变 Case score 或
status。

通用审计入口：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_collision_evaluator_v5.py \
  --device cuda \
  --case-id CASE_ID \
  --self-check \
  --prediction CASE_ID=/absolute/path/to/prediction.mp4 \
  --output /mnt/nvme1/physics_video_benchmark/evaluator_audits/AUDIT_ID
```

## 10. 已知限制与后续验证

- Residual discovery 仍是 collision-specific motion/Hough，不是可直接复用于五个
  scene 的通用视觉基础模型。
- 短暂、静止但真实的额外球可能只获 tentative 权重，存在漏罚或惩罚偏轻风险。
- Motion ghost、反光、轨道结构和印刷圆形仍可能形成弱候选；当前以 evidence tier
  限制其影响，而不是声称完全消除。
- Residual participant 的未知质量以 reference 质量中位数代入 N-body，只是可审计
  近似。
- 最终真实审计只覆盖两个三球 Case；单元和合成测试虽覆盖任意 N 数据结构，但这不
  等价于 2 球、4 球及更大 N 的真实视频验证。
- SAM2 分割和 Hough/motion proposal 的观测质量仍会影响分数。应继续查看 IoU、
  position、cardinality 曲线和外置 audit video，而不能只看总分。
- GT-self 的 tentative false exposure 使最终分数低于 1。发布正式协议前应扩大
  reference self-audit，量化该保守偏差并冻结可接受范围。
- v5 尚未推广到单摆、自由落体、斜面下滑和匀速圆周运动；这四个 scene 仍沿用 v3
  evaluator，不能宣称已具备同样的开放世界防 hack 能力。

在扩大真实 N-body 审计、完成五 scene adapter 并在同一 v5 fingerprint 下重评全部
Baseline 之前，`scene_default_v5` 应继续保持 shadow 状态。
