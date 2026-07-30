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
       ├─ entity-set integrity: SoftDetA + AssA
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

额外对象惩罚要求 observation adapter 同时运行：

1. expected-ID 定向追踪；
2. open-world residual discovery。

只传播预先选中的 N 个 mask 无法发现第 N+1 个对象。Residual 必须满足
scene-specific 类别、最小面积和最短持续时间约束，避免把反光、阴影、细线断片或
单帧噪声误判成新物理实体。

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

位置只进入下一节的连续 localization quality；不在 visual content 中再次重复。
形状/外貌参与初始身份 assignment 和 visual content，但不再同时伪装成“检测是否
成功”，避免同一错误被 SoftDetA 与 content 重复计算。

## 5. 新增、消失、复制与 ID 连续性

对所有 reference 可观测 entity-time slot 累积连续 localization/objectness：

```text
Q = sum(weight_it × localization_quality_it)
R = reference expected-entity exposure
P = all prediction-track exposure, including residual tracks

SoftDetA = Q / (R + P - Q)
```

该定义自然满足：

- 正确且完整的一对一对象集合为 1；
- 对象偏移时连续下降；
- 对象消失使 `R` 保持而 `Q` 降低；
- 新增或复制对象使 `P` 增加而 `Q` 不增加；
- 持续十帧的幻觉对象比一帧噪声受到更大惩罚；
- 大对象不会仅凭像素面积淹没小对象，因为基础曝光单位是 entity-time。

身份连续性借鉴 HOTA 的 association 分解。对 reference ID `i` 和 prediction track
`j`：

```text
M_ij = 保持该关联的时间曝光
A_ij = M_ij / (E_i + E_j - M_ij)
AssA = 按 matched exposure 加权的 A_ij

S_entity_diagnostic = sqrt(SoftDetA × AssA)
integrity_gate      = SoftDetA × AssA
```

中途 ID switch、碎裂、消失后用新 ID 重生都会降低 `AssA`。另行输出 GOSPA
（默认 `p=2, c=1, alpha=2`）诊断，明确分解：

- localization；
- missed exposure；
- false exposure。

GOSPA 用于审计错误来源，不重复叠加到正式分数，避免对同一次新增/消失多次惩罚。

## 6. Reference 能力

新协议必须显式声明三种能力：

| 能力 | 可正式比较 |
| --- | --- |
| `same_case_gt` | 逐 ID 绝对位置、形状、外貌、生命周期、锚点和物理状态 |
| `physics_parent` | canonical physics；对象数量/身份/外貌只与当前 Case 条件首帧比较 |
| `annotation_only` | 仅版本化解析模型或物理约束中可识别的量 |

`physics_parent` 不能使用 parent 背景、底座、对象外貌或 raw pixel 位置充当 OOD GT。
`annotation_only` 也不能假装拥有唯一真实视频：例如碰撞缺少可靠恢复系数时，可以评
碰前速度、一维性、接触顺序、非穿透与动量约束，但不能唯一规定碰后轨迹。

OOD 的 condition appearance ROI 必须由条件图自身的 anchor/分割产生，不能复用
prediction frame-0 mask 去裁条件图；否则错误 prediction mask 会同时污染“GT”外貌
区域。

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
物理分比较逐 ID 轨迹、`striker→target_1→target_2` 接触图与顺序、碰前后速度、动量、
reference-relative 恢复系数、一维性、横漂与穿透。Open-world discovery 寻找第四球、
复制球和持续 residual。短时接触 mask 合并按遮挡处理，不立即判定对象死亡。

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

物理分比较 `y(t)`、速度、加速度、启动时刻、落地/退出事件、向下单调性和异常反弹。
到达 prediction 自身最大行程的固定比例不能作为唯一 impact 定义，否则错误的短距离
下落也会得到合理事件时间。稳定第二球是 residual FP；GT 落地前消失是 FN；GT 已经
退出画面后不再按普通消失惩罚。

### 7.3 `inclined_plane_slide`

实体是 `sliding_block`；斜面、端点和 track axis 是静态 apparatus/anchor，不计为
额外动态对象。斜面轴优先从条件图或 apparatus 得到，不能分别从 reference/prediction
的滑块轨迹拟合，否则水平运动也可能被规范化成“正确沿面运动”。

位置拆成沿面位移、法向间隙和相对斜面的姿态。物理分比较启动、`s(t)`、速度、加速度、
下降时间、接触、穿透、翻滚和离轨。额外滑块是 residual FP；到达底端前消失是 FN；
从真实末端退出则按生命周期事件处理。

### 7.4 `pendulum`

使用复合实体图：

```text
pivot/support -- string/rod -- bob
```

`bob` 是主要动态 ID，`string/rod` 是结构边，`pivot/support` 是静态锚点。Bob 位置
使用角向弧长和径向长度误差；string 使用 skeleton/端点连接与覆盖率，不能让支架大
mask 淹没 bob 的位置误差。

物理分比较角轨迹、周期、相位、振幅、阻尼、摆长稳定、pivot drift、string tautness
和连接拓扑。第二个 bob、分叉 string 或复制摆是 FP；bob/string 脱离是拓扑错误。
细线短时不可见按遮挡/低置信度处理，而不是直接判死。OOD 背景、底座或 bob 材质只与
当前条件首帧比较外貌；parent 只监督 canonical angular dynamics。

### 7.5 `uniform_circular_motion`

实体是一个或多个 `orbiter`，green disk、rotation center 和 disk radius 是静态锚点。
初始 ID 由条件首帧外貌、半径、相位和结构化 `object_i_orbit_radius` 锁定。不能在看完
完整轨迹后按拟合半径重新排序；那会掩盖内外轨交换和 silver/wood 身份交换。

逐 ID 位置距离为：

```text
d² = (delta_radius / sigma_radius)²
   + (reference_radius × wrapped_delta_angle / sigma_tangent)²
```

物理分比较角轨迹、角速度方向和大小、圆度、中心稳定、半径配置、对象间相位与角速度
一致性。额外 orbiter 是 residual FP；任一预期对象消失是 FN；多圈 angle unwrap、
短时互相遮挡和中心附近对象需要明确 corner-case 测试。

## 8. Case 分数与失败语义

对象完整性不能只是普通加权平均中的一项，否则其惩罚会被高 physics/appearance
稀释。最终使用两层组合：

```text
content = geometric_mean(
    scene_physics,
    per-ID shape/appearance,
    anchor/topology consistency
)

S_case = integrity_gate × content
```

例如三球都完全正确但全程多出第四球时：

```text
SoftDetA = 3 / 4
AssA = 1
S_case <= 0.75
```

该上限不会因为其余内容组件为 1 而被抬到接近满分。不可比较的 content 组件移除并
重归一化；核心实体明确缺失可以得到有限 0 分。`sqrt(SoftDetA×AssA)` 仍可作为
HOTA-style 可读诊断，但不用于稀释式的最终组合。

状态继续沿用现有责任边界：

- `evaluated/normal`：完整观测；
- `evaluated/degraded_prediction`：prediction 解码、发现或物理提取失败，返回有限低分；
- `unavailable/reference_invalid`：GT、条件图或 reference anchor 不足；
- `error/internal`：代码 invariant 或未捕获异常。

所有正常输出必须有限、位于 `[0,1]`、schema 固定，并分别报告 reference/prediction
观测质量。可视化至少包含逐 ID overlay、匹配表、birth/death/switch 日志、位置/形状
曲线和 GOSPA 分解。

## 9. 落地状态与验证门

已实现但尚未接入正式 scene evaluator 的 shadow 内核：

```text
src/physbench/evaluation/common/entities/
  contracts.py   EntitySpec、ObjectTrack、visibility、reference capability
  matching.py    initial-window rectangular assignment + dustbins
  scoring.py     continuous distance、SoftDetA、AssA、GOSPA、geometric score
```

单元测试覆盖 identity、近/远距离单调性、missing、短/长 extra、duplicate、ID switch、
矩形匹配、dustbin、零分门控和有限输出。

正式发布前还必须完成：

1. 为每个 scene 实现可靠的 open-world residual discovery；
2. 用 reference、真实 baseline prediction 与人工扰动集校准尺度；
3. 验证平移、缩放、冻结、删除、复制、第四对象、ID swap、无接触、穿透和时间扭曲
   都按预期单调降分；
4. 对高分、低分和模型排序反例人工盲审；
5. 以新 fingerprint 重评所有 baseline 后才能形成 leaderboard。

建议迁移顺序是 collision、circular motion、inclined plane、free fall、pendulum。
collision 和 circular motion 已有多实例观测，最适合先验证通用对象层；复合单摆需要
额外的部件图和拓扑观测。

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
