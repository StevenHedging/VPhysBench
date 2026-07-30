# 对象中心评估架构

日期：2026-07-30

## 1. 目标与边界

五个 scene 的共同问题不是“能否分出一个前景 mask”，而是：

1. Case 规定的每个物理实体是否仍然存在；
2. reference 与 prediction 中的同一物理身份是否被持续对应；
3. 对象位置、形状和外貌是否接近；
4. 是否凭空新增、复制、删除或交换对象；
5. 在实体完整的前提下，scene-specific 物理状态是否正确。

因此对象中心 evaluator 使用以下流程。`scene_default_v5` 已在
`collision_1d` 接入这条链路；其余 scene 目前仍保留为设计目标：

```text
Case + conditioned first frame
  └─ Case Entity Manifest + immutable identity/physics anchors

reference video
  └─ manifest-directed expected-ID segmentation/tracking

prediction video
  ├─ manifest-directed expected-object proposals
  └─ open-world motion/Hough residual discovery
       ↓
  deduplicated detections + causal persistent track IDs
       ↓
  evidence-tiered per-frame Hungarian + explicit null assignment
       ↓  track ID is frozen; no future-trajectory renaming
  per-entity position / shape / appearance comparison
       ├─ entity-set integrity: PresenceDetA + observable AssA
       ├─ localization audit: SoftDetA
       ├─ GOSPA audit: localization / missed / false
       └─ arbitrary-N collision adapter: state and contact graph
       ↓
  weighted geometric case score
```

当前碰撞实现为 `CollisionOpenWorldCaseEvaluator 2.2`，协议内部标记为
`open_world_v1.2` 和 `nbody_v1.2`。它由 opt-in 的 `scene_default_v5` 承载，不修改
已经冻结的 `scene_default_v3` 或 `scene_default_v4`，也不改变 Baseline、Task 或
Dataset 契约。v5 仍是 shadow 协议；它可以真实运行和产出分数，但不能与旧协议结果
混入同一 leaderboard。

## 2. 对象契约

通用层使用：

```text
EntitySpec
  entity_id              持久物理身份
  role_id                striker、bob、sliding_block 等语义角色
  entity_class           ball、block、orbiter 等
  parts                  复合实体的部件
  exchangeability_group  真正不可区分对象的等价组
  lifecycle              persistent、may_exit 等
  anchor                 首帧位置、外貌、尺寸、形状和初始拓扑

ObjectTrack
  track_id
  matched_entity_id
  per-frame position / area / observation confidence
  visibility
```

`entity_id` 与 `role_id` 分开：身份在全视频内不变，角色描述 Case 中的物理功能。
可见性明确区分：

- `visible`：参与视觉和位置比较；
- `occluded`：暂停视觉比较，但允许跨遮挡桥接 ID；
- `out_of_frame`：不按普通漏检处理，由 scene 判断退出事件；
- `not_yet_present`：用于允许进入画面的生命周期；
- `unknown`：不进入正式比较，并在质量字段中报告。

这些状态的裁决权不属于 prediction。Same-case GT 中的 expectedness 由 reference
决定；prediction detector failure 仍是 missing。`occluded` 必须有 reference 或独立
遮挡几何证据，`out_of_frame` 必须有 scene 边界和合法 exit event，不能让预测结果
自行声明状态来移出分母。检测置信度只用于审计或校准，正式曝光不得靠 prediction
置信度任意缩小。

插值点只能用于稳健动力学拟合，不能伪装成实际视觉观测，也不能增加实体曝光。

Dataset 4.0.0 的 Case schema 尚未逐条显式保存完整 entity graph、生命周期或可见性，
`assets.subject_mask` 也没有冻结标注。当前 `Entity Manifest v1` materializer 优先
读取显式 `case.entities`；没有显式声明时，从结构化 `physics`、`appearance` 和
condition anchor 按 scene 的确定性规则物化。Dataset 4.0.0 的 214/214 条 Case 均可
物化；碰撞 v2.2 会实际消费 manifest 的实体数量、ID、初始顺序、质量、半径和初速度，
并把 materializer ID、manifest 内容和 digest 写入结果。任何基数、单位、必需标注或
初始顺序冲突都按 reference invalid 处理，不猜测也不静默写回冻结 release。若后续
增加人工校验的 ID/anchor cache，应作为带独立 digest 的版本化 evaluation asset
发布。

## 3. 身份匹配

语义身份只能由 Case 条件首帧和最早可靠观察窗口建立：

```text
C_ij =
    w_position   × initial anchor distance
  + w_appearance × canonical appearance distance
  + w_shape      × shape/size distance
  + w_relation   × initial topology distance
```

角色不兼容的匹配成本为无穷。通用设计中的初始语义匹配是带 dustbin 的矩形指派：

- 未匹配 expected entity 是初始 missing；
- 未匹配 candidate track 是 residual/extra 候选；
- prediction 中对象数量不必等于 GT；
- 语义 anchor 和物理 ID 在整段视频冻结。

禁止用完整未来 GT 轨迹选择最优 permutation。那会让错误动力学、角色交换和轨迹
穿越通过事后改名获得高分。真正不可区分的实体可以显式声明
`exchangeability_group`，但也只能在初始窗口选择一次全局 permutation，不能逐帧
重新排列。

需要区分两种 assignment：

1. 条件/初始窗口产生的 semantic anchor assignment 冻结，用于初始化角色与追踪器；
2. 每帧只按当前位置产生的 localization assignment，用固定 GT ID 和固定 prediction
   track ID 形成 `EntityMatch`，用于计算 HOTA-style association。

第二步不能改写任何 track ID，也不能用整段未来轨迹选择 permutation。它的作用只是
记录某一帧哪两个固定 ID 在空间上对应；因此追踪器在接触后换了物理对象时，AssA 会
看到 pair 变化。

碰撞 v2.2 已实现第二层：reference 的 expected ID 来自 manifest-directed SAM2
通道，prediction 候选经过去重后只按当前及历史观测建立因果 track ID；每帧按证据
等级从强到弱执行矩形 Hungarian。指派显式包含 null hypothesis，配置
`minimum_match_position_similarity = 0.1`。若一个候选与 GT 的 reference-scaled
Cauchy 位置相似度低于该阈值，就不再为了凑齐基数而强制配对，而是同时保留：

- GT entity 的 missing exposure；
- prediction track 的 extra exposure；
- 含实际距离、相似度、证据等级和拒绝原因的 `rejected_candidate_matches` 审计行。

高等级候选的被拒边不会阻止剩余 GT 与低等级候选形成合法匹配。只有在所有等级完成
后仍两端未匹配的拒绝边才作为正式 null-assignment 诊断输出。

身份并非每帧都可观察。两个外貌不可区分的球在接触合并期、完全遮挡期或低分辨率
重叠期，追踪器的任意选择不能被当作生成模型的 ID switch。Scene adapter 必须由
reference/condition 给出逐帧 `identity_observable`：

- 仍比较对象是否存在、位置与集合基数；
- 暂停该 slot 的 AssA 累积；
- 分离后若外貌/几何足以重新识别，则恢复原 ID；
- 若始终不可识别，只做预先声明 exchangeability group 的逐帧 unordered-set
  localization；这些 match 不进入 AssA，semantic/material ID 仍不被逐帧重命名。

这不是允许事后重命名，而是明确承认观测数据无法支持的身份结论。
Same-case 的逐帧身份可观察性来自 reference timeline；单张 condition 不能独自裁决
未来每一帧。`physics_parent`/`annotation_only` 只能使用 condition anchor 加上
scene-specific、版本化的 detectability policy，仍不得读取 prediction 来降低分母。

额外对象惩罚要求 observation adapter 同时运行：

1. expected-ID 定向追踪；
2. open-world residual discovery。

只传播预先选中的 N 个 mask 无法发现第 N+1 个对象。通用设计把 residual 分为三类：

- participant-class extra：与场景物理主体同类，稳定出现后强惩罚；
- independently-moving salient residual：类别不同但明显成为独立运动主体，先以较低
  权重审计，人工校准后再计分。
- condition-novel static residual：背景配准后新出现、虽静止但显著的对象，仅作审计，
  待误报率校准后再决定是否计分。

三类候选都必须做 merge/split/fragment 去重，并满足以秒为单位的最短持续时间、面积、
运动独立性和置信度约束，避免把反光、手、阴影、细线断片或单帧噪声误判成新实体。
所有 observer 保留的候选都应参与正式指派，未匹配者保留为 residual exposure，不能
因幻觉对象过多让 evaluator 抛错或通过 Top-K 预筛静默消失。

碰撞 v2.2 的实例化是 directed SAM2 加逐帧 motion/Hough residual discovery。候选
超过 causal tracker 的冻结容量时，超出的检测不会被静默丢弃，也不会导致 evaluator
error，而是累积为 `overflow` prediction exposure。

## 4. 连续位置与对象质量

IoU 只在 mask 重叠时有位置梯度，不能区分“刚好不相交”和“相距很远”。通用层以
reference 或 condition 尺度定义位置距离：

```text
r_ref = sqrt(reference_mask_area / pi)
sigma = max(2 × r_ref, 0.02 × reference_frame_diagonal)
d = max(centroid_distance - jitter_tolerance, 0) / sigma

S_position = 1 / (1 + d²)
```

Cauchy 核在有限图像上严格连续且保留重尾：不相交但接近仍有明显正分，距离很远时
继续单调下降而不是统一截成 0。参数是当前 shadow 默认值，正式协议必须通过人工平移
与分割抖动集校准。尺度只能来自 reference 或 conditioned frame，不能使用 prediction
自身的运动跨度，否则整体平移和错误缩放会被规范化掉。

Scene adapter 可以覆盖距离坐标：

- 碰撞：沿轨道和横轨道的各向异性距离；
- 斜面：沿面位移与法向间隙；
- 单摆：角向弧长与摆长误差；
- 圆周：径向误差与角向弧长；
- 自由落体：竖直高度与横向漂移。

对象的形状和外貌与位置分开。建议使用平移对齐后的 canonical shape、面积比、
boundary F、Lab 颜色、纹理和条件首帧外貌。不可比较的组件移除并重新归一化，不能
擅自记为 1。逐 ID visual content 使用加权几何平均：

```text
S_visual = geometric_mean(shape, appearance, size)
```

位置进入 content，不再进入正式完整性门。碰撞 v2.2 将它封装在
`nbody_physics.track_position` 内，并保证位置只在该处正式评分；速度、动量、接触和
非穿透是不同物理量，不再另做一份绝对轨迹分。形状/外貌参与身份 anchor 和 visual
content，但不再同时伪装成“检测是否成功”。Localization-aware SoftDetA 保留为
诊断。

## 5. 新增、消失、复制与 ID 连续性

对所有 reference 授权的可观测 entity-time slot：

```text
M = matched entity-time exposure
R = reference expected-entity exposure
P = all prediction-track exposure, including residual tracks

PresenceDetA = M / (R + P - M)
```

碰撞 `open_world_v1.2` 将候选的正式 exposure 权重冻结为：

| evidence tier | exposure 权重 | 语义 |
| --- | ---: | --- |
| `PARTICIPANT` | 1.00 | 可靠物理参与体 |
| `INDEPENDENT_SALIENT` | 0.50 | 独立显著但证据较弱的实体 |
| `TENTATIVE` | 0.25 | 保守计入的弱候选 |
| `AMBIGUOUS` | 0.00 | 只审计，不参与正式匹配或惩罚 |

确认后的 motion/tentative participant track 会追溯为完整 `PARTICIPANT` exposure，
不存在免费的确认等待窗；仅靠静态圆的持续性则不能升级成 N-body participant。
`overflow` 也按其检测证据计入 `P`，至少按 tentative 权重计费。不得在不同运行中按
视频观感临时改变这些权重。

该定义自然满足：

- 正确且完整的一对一对象集合为 1；
- 对象消失使 `R` 保持而 `M` 降低；
- 新增或复制对象使 `P` 增加而 `M` 不增加；
- 持续十帧的幻觉对象比一帧噪声受到更大惩罚；
- 大对象不会仅凭像素面积淹没小对象，因为基础曝光单位是 entity-time。

对象偏移由上一节的 `S_position` 连续计分，避免与绝对轨迹重复惩罚。另行计算
localization-aware 审计量：

```text
Q = sum(weight_it × localization_quality_it)
SoftDetA = Q / (R + P - Q)
```

身份连续性借鉴 HOTA 的 association 分解，只在身份可观察 slot 上计算。对
reference ID `i` 和 prediction track `j`：

```text
M_ij = 保持该关联的时间曝光
A_ij = M_ij / (E_i + E_j - M_ij)
AssA = 按 matched exposure 加权的 A_ij

S_entity_diagnostic = sqrt(SoftDetA × AssA_observable)
integrity_gate      = PresenceDetA × AssA_observable
```

中途 ID switch、碎裂、消失后用新 ID 重生都会降低 `AssA`。另行输出 GOSPA
（默认 `p=2, c=1, alpha=2`）诊断，明确分解：

- localization；
- missed exposure；
- false exposure。

GOSPA 用于审计错误来源，不重复叠加到正式分数，避免对同一次新增/消失多次惩罚。
实现还必须逐 reference ID、逐 prediction track 检查 matched exposure 守恒；只检查
全局总量会允许一个身份重复消费多条轨迹，并把真正 missing 的身份错误算成满分。

`M/R/P/E_i/E_j` 都是在统一物理时间轴上对 `delta_t` 的积分，不是原始帧数；视频先按
协议重采样到共享时间点和统一画布，再计算距离和曝光。`E_i/E_j` 只包含
association-eligible exposure；若两侧 eligible exposure 都为 0，
`AssA_observable=1`（neutral），但该时段的 group cardinality 仍进入 PresenceDetA。

碰撞 v2.2 对“模型通过不生成难帧来逃避物理评分”采用 anti-abstention 语义：

- shape/appearance 仍只比较成功匹配的主体 mask，extra 由完整性 gate 处理；
- N-body 的 position、velocity、momentum、nonpenetration 使用完整 GT 的真实秒
  reference exposure 作分母，未观测的预期实体时段贡献 0，而不是从分母删除；
- GT contact graph 不按 prediction coverage 裁剪；预测 residual participant 的接触
  事件也保留为可能的 false event；
- “GT 无接触且 prediction 无接触”只有在全部预期 pair exposure 完整时才能认证为
  正确，否则 contact 分保守置 0。

因此 missing 会同时降低“实体集合是否完整”的 gate 和“缺失时段是否提供物理证据”
的 N-body content，这是有意的 anti-abstention，不是把同一个位置距离重复算两次。
每个 participant detection 在 expected slot 或 residual N-body channel 中严格守恒，
partially matched track 的未匹配时段也不能从物理输入中消失。只有 supervisor 本身
不可用时才允许 omit；directed observation 失败仍继续 residual discovery，而
residual discovery 或 open-world comparison 失败则 fail-closed 为等价空 prediction，
返回 `evaluated/degraded_prediction` 的有限保守低分，不能退回 directed-only 满分。

统一时间格使用真实秒 cell 权重。短视频或损坏样本只保留实际可用前缀；尾段填充中性
画布并标记 `available=false`，不复制末帧，预期实体在该尾段继续形成 missing
exposure。

## 6. Reference 能力

新协议必须显式声明三种能力：

| 能力 | 可正式比较 |
| --- | --- |
| `same_case_gt` | 逐 ID 绝对位置、形状、外貌、生命周期、锚点和物理状态 |
| `physics_parent` | canonical physics；对象数量/身份/外貌只与当前 Case 条件首帧比较 |
| `annotation_only` | 仅版本化解析模型或物理约束中可识别的量 |

能力必须落实到 `metric × supervisor` 矩阵，而不是只给 Case 一个粗粒度标签。例如
`physics_parent` 的单摆 OOD：

| 指标 | supervisor |
| --- | --- |
| canonical angle / period / phase | physics-identical parent |
| 当前 bob 数量、外貌、pivot、尺度和拓扑 | 当前 Case 条件首帧 |
| parent raw pixel position、背景、底座和生命周期 | unavailable，禁止使用 |

不同 capability profile 的总分不能通过“删掉不可比项后重新归一化”直接混成同一
排行榜；应固定 profile 分层汇报，或发布经过显式校准的共同维度分数。

Parent canonical dynamics 必须先用当前 condition 的 `t=0`、左右方向、pivot 和尺度做
一次只依赖初始帧的 gauge alignment，例如 sign-normalized angle 与共同 physical
time zero；不能用未来 prediction 选择最优相位或镜像。当前 Case 的未来身份/数量与
拓扑由 `condition anchor + scene persistent/topology contract` 监督，不是说单张首帧
本身包含未来生命周期 GT。

`physics_parent` 不能使用 parent 背景、底座、对象外貌或 raw pixel 位置充当 OOD GT。
`annotation_only` 也不能假装拥有唯一真实视频：例如碰撞缺少可靠恢复系数时，可以评
碰前速度、一维性、接触顺序、非穿透与动量约束，但不能唯一规定碰后轨迹。

OOD 的 condition appearance ROI 必须由条件图自身的 anchor/分割产生，不能复用
prediction frame-0 mask 去裁条件图；否则错误 prediction mask 会同时污染“GT”外貌
区域。

Dataset 4.0.0 还存在一条已确认的可识别性冲突：
`pendulum_ltot0110mm_lrope0100mm_r010mm_a010deg_ood01` 的结构化标注和
physics parent 是 `10 deg`，但 OOD 条件首帧肉眼约为 `31 deg`。下一版 evaluator
不得猜测哪个权威源正确；在数据 release 修正或明确裁决前，该 Case 必须从
physics-parent shadow score 隔离，但保留数据质量诊断。冻结 release 本身不在本次
修改中被静默改写。

## 7. 五场景适配

### 7.1 `collision_1d`

碰撞 v2.2 不再声明固定 `striker + target_1 + target_2`。Case Entity Manifest 给出
`ball_1 ... ball_N` 和 `track_axis` apparatus，要求 `N >= 2`，但不设 N 的场景常量，
也不假设只有一个主动球。legacy Case 的 N 由 `appearance.ball_sequence` 与全部
`ball_i_{mass,radius,initial_velocity}` 的一致对应确定；显式 `case.entities` 则直接
优先。初始顺序、外貌、半径、质量、材质和条件 anchor 用来固定身份契约，活动实体仅
作为结果中的诊断推断，不改变计分对象集合。

Reference 的轨道轴被冻结；位置分逐 ID 计算沿轨道和法向的各向异性连续距离，不使用
N 球 union 质心。接触在所有无序实体 pair 上基于球面间隙：

```text
gap_ij = center_distance - (radius_i + radius_j)
```

可信事件还结合闭合/分离速度、hysteresis、持续窗口与置信度，不能总以最小中心距
制造一个“碰撞帧”。`nbody_v1.2` 比较五个分量：

| component | 权重 | 内容 |
| --- | ---: | --- |
| `track_position` | 0.30 | 全部预期 ID 的沿轨/法向轨迹 |
| `contact_graph` | 0.25 | 全部无序 pair 的接触事件、时间与间隙 |
| `velocity` | 0.20 | 全部预期 ID 的沿轨速度 |
| `momentum` | 0.15 | 多物体系统相对 reference 的动量变化 |
| `nonpenetration` | 0.10 | 全部预期 pair 的过度穿透 |

完整 GT contact graph 不再写死 `striker→target_1→target_2`，也不按 prediction
coverage 删除。Open-world participant residual 的未匹配检测形成独立 N-body
channel，其接触事件保留为潜在 false event；弱于 participant 的候选只进入加权
presence exposure，避免用不可靠静态圆制造物理事件。Residual 的质量暂用 reference
质量中位数，是已记录的近似。短时接触 mask 合并按遮挡问题处理，不立即判定对象
死亡。

若接触期的球在视觉上不可区分，应使用 `contact_group` observation 并暂停该时段的
AssA；分离后只有在外貌或可证实几何连续性足以 re-ID 时才恢复材料身份，否则在
exchangeability group 层评分。像素半径从 reference/condition mask 或标定得到，不能
把米制物理半径直接当作像素 gap。

Collision adapter 的正式 localization 使用连续距离：

```text
delta² =
    (delta_parallel / sigma_parallel)²
  + (delta_normal   / sigma_normal)²

q_location = 1 / (1 + delta²)
```

因此不相交但接近和相距很远不会同得 0。Full-set mask IoU 仍是必需可视化诊断，
shape/appearance 也继续比较 matched mask，但 IoU 不混入 `track_position`。若任一
实体在 frame 0 尚未进入画面，应在最早可靠的碰撞前观测窗口建立 directed prompt；
不能强绑 frame 0，也不能等到接触后再利用未来行为重命名。

### 7.2 `free_fall`

实体是 `falling_body`，静态锚点包括释放位置和地面/画面出口。位置拆成竖直位移与
横漂；坐标系来自 GT 或条件场景，不能完全从 prediction 自拟合。

位置分比较 `y(t)`；物理分只比较速度、加速度、启动时刻、落地/退出事件、向下单调性
和异常反弹，避免把绝对轨迹重复计分。
到达 prediction 自身最大行程的固定比例不能作为唯一 impact 定义，否则错误的短距离
下落也会得到合理事件时间。稳定第二球是 residual FP；GT 落地前消失是 FN；GT 已经
退出画面后不再按普通消失惩罚。

### 7.3 `inclined_plane_slide`

实体是 `sliding_block`；斜面、端点和 track axis 是静态 apparatus/anchor，不计为
额外动态对象。斜面轴优先从条件图或 apparatus 得到，不能分别从 reference/prediction
的滑块轨迹拟合，否则水平运动也可能被规范化成“正确沿面运动”。

位置拆成沿面位移、法向间隙和相对斜面的姿态。物理分比较启动、速度、加速度、
下降时间、接触、穿透、翻滚和离轨。额外滑块是 residual FP；到达底端前消失是 FN；
从真实末端退出则按生命周期事件处理。

### 7.4 `pendulum`

单摆需要真正的复合实体图，不能只用 `parts: tuple[str]`：

```text
static node: pivot/support
dynamic node: bob
structural edge: string/rod
relation: attached(pivot, string), attached(string, bob)
```

`bob` 是主要动态 ID，`string/rod` 是结构边，`pivot/support` 是静态锚点。Bob 位置
使用角向弧长和径向长度误差；string 使用 skeleton/端点连接与覆盖率，不能让支架大
mask 淹没 bob 的位置误差。

位置分比较 bob 的角轨迹与径向摆长；物理分比较角速度/加速度、周期、振幅、阻尼和
能量趋势；anchor/topology 分比较 pivot drift、string tautness 与连接关系。第二个
bob、分叉 string 或复制摆是 FP；bob/string 脱离是拓扑错误。
细线短时不可见只能由 reference/condition detectability 判为遮挡，不能由 prediction
低置信度自行跳过。OOD 背景、底座或 bob 材质只与
当前条件首帧比较外貌；parent 只监督 canonical angular dynamics。

### 7.5 `uniform_circular_motion`

实体是一个或多个 `orbiter`，green disk、rotation center 和 disk radius 是静态锚点。
初始 ID 由条件首帧外貌、半径、相位和结构化 `object_i_orbit_radius` 锁定。不能在看完
完整轨迹后按拟合半径重新排序；那会掩盖内外轨交换和 silver/wood 身份交换。
Rotation center 与 disk radius 也必须由 condition 或 reference 冻结，禁止从
prediction 完整轨迹重新拟合；否则错误偏心圆会通过自选中心获得高分。

逐 ID 位置距离为：

```text
d² = (delta_radius / sigma_radius)²
   + (reference_radius × wrapped_delta_angle / sigma_tangent)²
```

位置分比较逐 ID 半径/角轨迹和对象间相位；物理分比较角速度方向、大小、均匀性与
周期性；anchor/topology 分比较共同旋转中心和 disk 约束。额外 orbiter 是 residual
FP；任一预期对象消失是 FN；多圈 angle unwrap、短时互相遮挡和中心附近对象需要明确
corner-case 测试。`r≈0` 时极角不可识别，必须回退到 Cartesian distance。

### 7.6 指标所有权

为避免 adapter 再次双罚，同一 primitive 只能属于一个正式分量：

| scene | position | physics invariants/events | anchor/topology |
| --- | --- | --- | --- |
| collision | 每 ID 的沿轨/法向坐标轨迹 | 全 pair 接触图、速度、系统动量、非穿透 | 固定 reference 轨道轴 |
| free fall | `x(t), y(t)` | 速度、加速度、启动、impact/exit、反弹 | ground/exit anchor |
| inclined plane | 沿面/法向坐标与相对姿态 | 速度、加速度、启动、下降、接触/离轨 | 固定斜面轴、端点、透视标定 |
| pendulum | bob 的角向/径向轨迹 | 角速度、周期、振幅、阻尼、能量趋势 | pivot—string—bob 连接与支撑稳定 |
| circular | 每 ID 的半径、角度与相对相位 | 角速度、方向、均匀性、周期性 | 固定中心、disk、对象基数 |

同一位置 primitive 不应同时出现在多个正式分量。碰撞 v2.2 在实现上把 position 与
其余四个 scene-specific 量共同封装成 `nbody_physics`，但内部仍是互斥的五分量；
physics 的导数可从轨迹估计，不能再复制一份绝对轨迹分。Topology 不重复比较 node
的绝对坐标，只比较锚点漂移、关系和事件结构。

## 8. Case 分数与失败语义

对象完整性不能只是普通加权平均中的一项，否则其惩罚会被高 physics/appearance
稀释。最终使用两层组合：

```text
content = geometric_mean(
    per-ID position,
    per-ID shape,
    per-ID appearance,
    scene physics invariants/events,
    anchor/topology consistency
)

S_case = integrity_gate × content
```

碰撞 v2.2 的具体内容组合为：

```text
content = weighted_geometric_mean(
    nbody_physics: 0.60,
    matched_subject_shape: 0.20,
    matched_subject_appearance: 0.20
)
```

例如 N 个球都完全正确但全程多出一个同权 participant 时：

```text
PresenceDetA = N / (N + 1)
AssA_observable = 1
S_case <= N / (N + 1)
```

该上限不会因为其余内容组件为 1 而被抬到接近满分。不可比较的 content 组件移除并
重归一化；但不同 reference capability 必须分层，不能靠任意 omission 制造更高分。
全部核心曝光缺失或显式 critical failure 可以得到有限 0 分；部分缺失按第 5 节同时
影响完整性和相应 N-body evidence coverage，不能通过删分母获益。
`sqrt(SoftDetA×AssA_observable)` 仍可作为 HOTA-style 可读诊断，但不用于稀释式的
最终组合。

状态继续沿用现有责任边界：

- `evaluated/normal`：完整观测；
- `evaluated/degraded_prediction`：prediction 解码、发现或物理提取失败，返回有限低分；
- `unavailable/reference_invalid`：GT、条件图或 reference anchor 不足；
- `error/internal`：代码 invariant 或未捕获异常。

所有正式 similarity/score 必须有限、位于 `[0,1]`、schema 固定，并分别报告
reference/prediction 观测质量。GOSPA 原始距离不受 `[0,1]` 限制，但其 normalized
diagnostic 必须有限。可视化至少包含逐 ID overlay、匹配表、birth/death/switch 日志、
位置/形状曲线、身份可观察性时间线和 GOSPA 分解。

## 9. 落地状态与验证门

公共实体内核已经实现：

```text
src/physbench/evaluation/common/entities/
  contracts.py   EntitySpec、ObjectTrack、visibility、分通道真实秒 exposure
  manifest.py    Case Entity Manifest v1 的显式解析与确定性 legacy materializer
  timeline.py    公共物理时间格与 cell 权重
  matching.py    通用 initial-anchor rectangular assignment
  observer.py    证据分层、因果 track、null assignment、overflow、开放世界比较
  scoring.py     continuous distance、PresenceDetA、SoftDetA、observable AssA、
                 GOSPA、gated geometric score

src/physbench/evaluation/scenes/collision/
  open_world.py       directed SAM2 + motion/Hough residual discovery
  nbody.py            role-free arbitrary-N kinematics/contact scoring
  v5_evaluator.py     collision evaluator 2.2 integration
  v5_visualization.py open-world/N-body 过程可视化
```

`scene_default_v5` 已实际把上述内核接入 `collision_1d`，其运行身份是：

```text
evaluator: collision_1d_open_world_nbody 2.2
observer:  open_world_v1.2
physics:   nbody_v1.2
```

单元测试覆盖 2/4/N manifest、任意 N pair、近/远距离单调性、null assignment、
missing、短/长 extra、duplicate、ID switch、身份不可观察窗口、候选 overflow、
逐 ID/逐 track exposure 守恒、partially matched participant 守恒、短视频尾段、
contact anti-abstention、fail-closed residual、零分门控和有限输出。专项测试为
52/52，全仓测试为 346/346。

最终真实视频审计没有触发回滚：

| 样本 | Case score | evaluator error |
| --- | ---: | ---: |
| `v04374` GT-self | 0.9880578251 | 0 |
| `v06332` GT-self | 0.9693811074 | 0 |
| `v04374` WAN | 0.0250737646 | 0 |
| `v06332` WAN | 0.1203513978 | 0 |
| `v04374` quantity，多球反例 | 0.0039100888 | 0 |
| `v06332` quantity，少球反例 | 0.0690306009 | 0 |

GT-self 的 N-body 均为 1，低于满分来自保守 tentative false exposure；已知多球/少球
反例在 null assignment、完整 GT exposure 和 fail-closed 语义下进一步降分。r7/r8
审计分别为 4/4、2/2 `evaluated`，均为 0 error、0 unavailable。

每个碰撞 Case 会写出 `per_frame.csv`、主体 IoU 曲线、实体位置曲线和对象基数时间线；
大型 overlay、匹配/拒绝日志、contact graph 和 N-body dashboard 外置到
`/mnt/nvme1/physics_video_benchmark/evaluation_visualizations`，仓库
`visualizations` 链接可直接访问。

当前成熟度仍定义为 **可运行、已真实审计的 shadow collision protocol**，不等同于
正式 leaderboard 协议。正式发布前仍需：

1. 扩大真实 2/4/N 球、复制、遮挡、交换、无接触、穿透和时间截断审计；目前真实审计
   的 GT Case 仍是三球；
2. 继续校准 motion/Hough 对静态额外球、反光、轨道圆形结构和 motion ghost 的误报/
   漏报；
3. 完成接触期 identity observability/contact-group 的可靠监督；
4. 把 Case Entity Manifest 与双通道 open-world observer 逐 scene 迁移到其余四个
   evaluator；第 7.2–7.5 节目前仍是设计，不应误报为已实现；
5. 冻结新 fingerprint 并用同一协议重评所有 baseline 后再形成 leaderboard。

## 10. 方法来源

本设计吸收但不直接照搬以下指标：

- Luiten et al., *HOTA: A Higher Order Metric for Evaluating
  Multi-object Tracking*, IJCV 2021：检测与身份关联分解；
- Rahmathullah et al., *Generalized Optimal Sub-Pattern Assignment Metric*,
  FUSION 2017：定位、漏检和误检的可分解集合距离；
- Perazzi et al., *A Benchmark Dataset and Evaluation Methodology for Video
  Object Segmentation*, CVPR 2016：DAVIS region J 与 contour F；
- 轨迹预测中的 ADE/FDE：连续位置误差的诊断思路。

Benchmark 的正式分数仍需由自身物理 Case、reference capability 与扰动审计校准，
不能直接把通用 MOT/VOS 指标当作物理视频质量。
