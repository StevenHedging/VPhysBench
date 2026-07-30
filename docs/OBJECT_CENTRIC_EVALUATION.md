# 对象中心评估架构

日期：2026-07-30

## 1. 目标与边界

五个 scene 的共同问题不是“能否分出一个前景 mask”，而是：

1. Case 规定的每个物理实体是否仍然存在；
2. reference 与 prediction 中的同一物理身份是否被持续对应；
3. 对象位置、形状和外貌是否接近；
4. 是否凭空新增、复制、删除或交换对象；
5. 在实体完整的前提下，scene-specific 物理状态是否正确。

因此下一代 evaluator 使用以下流程：

```text
Case + conditioned first frame
  └─ expected entity graph + immutable identity anchors

reference / prediction video
  ├─ expected-ID directed tracking
  └─ open-world residual discovery
       ↓
  persistent object tracks + visibility states
       ↓
  initial-window rectangular assignment + dustbins
       ↓  identity is frozen; no future-trajectory rematching
  per-entity position / shape / appearance comparison
       ├─ entity-set integrity: PresenceDetA + observable AssA
       ├─ localization audit: SoftDetA
       ├─ GOSPA audit: localization / missed / false
       └─ scene adapter: physical state and topology
       ↓
  weighted geometric case score
```

这个架构是新协议的 shadow 基础，不修改已经冻结的
`scene_default_v3` 或 `scene_default_v4`，也不改变 Baseline、Task 或 Dataset
契约。只有在扰动测试和真实 prediction 人工排序审计通过后，才能发布新的正式协议。

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

当前 Dataset 4.0.0 的 Case schema 尚未显式保存 entity graph、生命周期或可见性，
`assets.subject_mask` 也没有冻结标注。Shadow 阶段由 scene adapter 根据现有
`physics`、`appearance` 和条件首帧生成 `EntitySpec`；不能把在线推断结果静默写回
冻结 release。若后续增加人工校验的 ID/anchor cache，应作为带独立 digest 的版本化
evaluation asset 发布。

## 3. 身份匹配

跨视频匹配只能使用 Case 条件首帧和最早可靠观察窗口：

```text
C_ij =
    w_position   × initial anchor distance
  + w_appearance × canonical appearance distance
  + w_shape      × shape/size distance
  + w_relation   × initial topology distance
```

角色不兼容的匹配成本为无穷。匹配是带 dustbin 的矩形指派：

- 未匹配 expected entity 是初始 missing；
- 未匹配 candidate track 是 residual/extra 候选；
- prediction 中对象数量不必等于 GT；
- 匹配完成后在整段视频冻结。

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

只传播预先选中的 N 个 mask 无法发现第 N+1 个对象。Residual discovery 采用两条
通道：

- participant-class extra：与场景物理主体同类，稳定出现后强惩罚；
- independently-moving salient residual：类别不同但明显成为独立运动主体，先以较低
  权重审计，人工校准后再计分。
- condition-novel static residual：背景配准后新出现、虽静止但显著的对象，仅作审计，
  待误报率校准后再决定是否计分。

两者都必须做 merge/split/fragment 去重，并满足以秒为单位的最短持续时间、面积、
运动独立性和置信度约束，避免把反光、手、阴影、细线断片或单帧噪声误判成新实体。
精确身份指派的复杂度应只对 expected entity 数量指数、对候选轨迹数量线性；因此
候选再多也全部参与最优指派，未匹配者全部保留为 residual exposure，不能因幻觉对象
过多让 evaluator 抛错或通过 Top-K 预筛改变最优角色匹配。

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

位置作为独立 `S_position` 进入 content；不再进入正式完整性门，也不在 scene
physics 中重复比较绝对轨迹。形状/外貌参与初始身份 assignment 和 visual content，
但不再同时伪装成“检测是否成功”。Localization-aware SoftDetA 保留为诊断。

## 5. 新增、消失、复制与 ID 连续性

对所有 reference 授权的可观测 entity-time slot：

```text
M = matched entity-time exposure
R = reference expected-entity exposure
P = all prediction-track exposure, including residual tracks

PresenceDetA = M / (R + P - M)
```

Residual 的 exposure 权重必须冻结在协议里：

```text
P =
    P_expected_tracks
  + 1.0 × P_participant_class_extra
  + lambda_salient × P_independently_moving_salient

audit-only: lambda_salient = 0
score-enabled: lambda_salient 经人工误报校准后冻结在 (0, 1]
```

静态 novel residual 在当前阶段只报告，不进入 `P`。不得在不同运行中按视频观感临时
改变权重。

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

逐 ID position/shape/appearance 只在成功匹配曝光上条件聚合，missing/extra 只由 gate
惩罚一次。只有 supervisor 本身不可用时才允许 omit；prediction 提取失败必须返回
`degraded_prediction` 的有限低分，不能借 omit 抬高分数。只有全部核心曝光缺失或显式
critical failure 才把整个 content 置 0，单个对象部分缺失不重复归零。

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

实体：

```text
striker, target_1, target_2 + track_axis anchor
```

初始身份由左右顺序、`appearance.ball_sequence`、半径、质量、材质和条件首帧锁定。
主位置分逐角色计算沿轨道/法向距离，不再使用三球 union 质心。接触基于球面间隙：

```text
gap_ij = center_distance - (radius_i + radius_j)
```

可信接触还需联合相对速度符号变化和持续窗口，不能总以最小中心距制造一个“碰撞帧”。
位置分比较逐 ID 轨迹；物理分比较碰前后速度、动量、reference-relative 恢复系数、
一维运动约束与非穿透；拓扑/事件分比较
`striker→target_1→target_2` 接触图和顺序。Open-world discovery 寻找第四球、复制球
和持续 residual。短时接触 mask 合并按遮挡处理，不立即判定对象死亡。
若接触期的球在视觉上不可区分，应使用 `contact_group` observation 并暂停该时段的
AssA；分离后只有在外貌或可证实几何连续性足以 re-ID 时才恢复材料身份，否则在
exchangeability group 层评分。像素半径从 reference/condition mask 或标定得到，不能
把米制物理半径直接当作像素 gap。

Collision adapter 的 localization 可以保留少量 IoU 精细项，但主项必须是连续距离：

```text
delta² =
    (delta_parallel / sigma_parallel)²
  + (delta_normal   / sigma_normal)²

q_location = 0.75 / (1 + delta²) + 0.25 × per-ID IoU
```

IoU 不再决定“不相交即统一为 0”，同时仍能区分真正重合的实例边界。若 striker 在
frame 0 尚未进入画面，应在“最早共同可观测且仍处于碰撞前”的窗口冻结 ID；不能强绑
frame 0，也不能等到接触后再利用未来行为重命名。

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
| collision | 每 ID 的沿轨/法向坐标轨迹 | 速度、动量、恢复、非穿透、一维性 | 固定轨道、两级接触图与顺序 |
| free fall | `x(t), y(t)` | 速度、加速度、启动、impact/exit、反弹 | ground/exit anchor |
| inclined plane | 沿面/法向坐标与相对姿态 | 速度、加速度、启动、下降、接触/离轨 | 固定斜面轴、端点、透视标定 |
| pendulum | bob 的角向/径向轨迹 | 角速度、周期、振幅、阻尼、能量趋势 | pivot—string—bob 连接与支撑稳定 |
| circular | 每 ID 的半径、角度与相对相位 | 角速度、方向、均匀性、周期性 | 固定中心、disk、对象基数 |

Position 不再进入 physics；physics 的导数可从轨迹估计，但其得分只比较表中不变量和
事件。Topology 不重复比较 node 的绝对坐标，只比较锚点漂移、关系和事件结构。

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

例如三球都完全正确但全程多出第四球时：

```text
PresenceDetA = 3 / 4
AssA_observable = 1
S_case <= 0.75
```

该上限不会因为其余内容组件为 1 而被抬到接近满分。不可比较的 content 组件移除并
重归一化；但不同 reference capability 必须分层，不能靠任意 omission 制造更高分。
全部核心曝光缺失或显式 critical failure 可以得到有限 0 分；单个对象部分缺失只由
gate 惩罚。`sqrt(SoftDetA×AssA_observable)` 仍可作为
HOTA-style 可读诊断，但不用于稀释式的最终组合。

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

已实现但尚未接入正式 scene evaluator 的 shadow 内核：

```text
src/physbench/evaluation/common/entities/
  contracts.py   EntitySpec、ObjectTrack、visibility、reference capability
  matching.py    initial-window rectangular assignment + dustbins
  scoring.py     continuous distance、PresenceDetA、SoftDetA、observable AssA、
                 GOSPA、geometric score
```

单元测试覆盖 identity、近/远距离单调性、missing、短/长 extra、duplicate、ID switch、
身份不可观察窗口、矩形匹配、dustbin、候选 overflow、逐 ID 曝光守恒、零分门控和
有限输出。

正式发布前还必须完成：

1. 为每个 scene 实现可靠的双通道 open-world residual discovery；
2. 用 reference、真实 baseline prediction 与人工扰动集校准尺度；
3. 验证平移、缩放、冻结、删除、复制、第四对象、ID swap、无接触、穿透和时间扭曲
   都按预期单调降分；
4. 对高分、低分和模型排序反例人工盲审；
5. 以新 fingerprint 重评所有 baseline 后才能形成 leaderboard。

建议迁移顺序是 circular motion、collision、inclined plane、free fall、pendulum。
圆周运动全部有同 Case GT，最适合作为对象层 control；碰撞是当前优先场景，但必须先
完成第四球 residual detector 和接触 ambiguity group。复合单摆在实体图和上述 OOD
冲突解决前只做 audit，不产出新的统一总分。

因此当前成熟度定义为 **audit-only shadow core**：允许产出 tracks、matches、
PresenceDetA/SoftDetA/AssA/GOSPA 和可视化，不进入正式 leaderboard。Circular 和
collision adapter 可率先接入 audit instrumentation；只有通过反事实单调性和真实视频
人工盲审，才启用 shadow score。Pendulum 暂时阻断。

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
