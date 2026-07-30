# 五场景评估协议

## 1. 边界与输入

正式评估属于 Benchmark。TaskEvaluator 消费：

```text
CanonicalTaskPlan + frozen cases + predictions.jsonl + evaluation protocol
```

当前官方协议：

```text
configs/evaluation/protocols/scene_default_v3.json
```

官方 v6 Task 使用 `scene_default_v3`，五个 scene evaluator 都使用实现版本 `1.3`。
v3 是独立协议，不覆盖 v5 run 的 canonical v2 结果。历史协议继续冻结：

- v2：v5 Task，evaluator `1.2`，reference-bounded 时间轴与顺序前向解码；
- v1：v4 Task，evaluator `1.1`，random-seek；单摆使用固定 0–5 秒时间轴。

v2 让单摆与其余 scene 一样采用 reference-bounded 时间轴，并为所有 scene 显式固定
顺序前向解码。协议、evaluator 版本、解码策略和完整 config 都进入 identity，因此
v1/v2/v3 结果不能混合。v3 沿用 v2 的媒体规范，但增加主体评分与新的失败责任语义。

碰撞评估器的下一版审计协议是：

```text
configs/evaluation/protocols/scene_default_v4.json
```

v4 只把 `collision_1d` 升级到 evaluator `1.4`，其余四个 scene 仍为 `1.3`。它增加
多帧角色发现、双向且互斥的 SAM2 实例传播、观测可靠性惩罚和外置过程可视化。v4
目前不替换官方 v6 Task 所固定的 v3；使用 v4 必须创建新的 reevaluation variant，
不能覆盖或混用已有 v3 分数。

`plan.jobs` 是主表。缺失 prediction、重复 prediction、失败生成或未知 case 都必须产生
一个显式 case result。

## 2. Case 状态

v3 的状态按故障责任划分：

| 状态 | 含义 | score |
| --- | --- | --- |
| `evaluated` | 得到正常分数，或 prediction 侧失败的保守退化分数 | `[0,1]` |
| `unavailable` | GT/reference 资产、时长或 reference 观测不可用 | `null` |
| `unsupported` | 协议未实现该 scene | `null` |
| `error` | 数据契约冲突、重复记录或真正的 evaluator 内部异常 | `null` |

prediction 缺记录、生成失败、视频缺失/损坏/过短、分割或跟踪失败均返回：

```text
status = evaluated
score = 0
quality.degraded = true
reason_code = 稳定原因码
```

这类 Case 进入 coverage，不能靠评估失败逃避难例。reference 侧问题则为
`unavailable`，不惩罚 Baseline；未知内部异常保留为 `error`，不会被低分掩盖。v1/v2
仍保持原先的历史状态语义。

## 3. 评分契约

v3 主 metric 为：

```text
scene_subject_state_similarity
```

有同 Case GT 时：

```text
subject = 0.50 × position + 0.20 × shape + 0.30 × appearance
case    = 0.60 × subject + 0.40 × scene_physics_state
```

其中：

- `position`：0.60 × 质心距离相似度 + 0.40 × 原画布 mask IoU；
- `shape`：0.60 × canonical crop mask IoU + 0.40 × boundary F；
- `appearance`：0.45 × mask 内 Lab 颜色直方图交集
  + 0.35 × canonical crop SSIM + 0.20 × 梯度方向纹理相似度。

质心距离按画布对角线归一化并使用指数核；canonical crop 保持主体宽高比、居中并缩放
到固定画布，因此形状项不会被绝对位置和尺度重复主导。所有逐帧值写入
`subject_components.csv`。

没有同 Case GT、只能使用 parent physics reference 时：

```text
case = 0.70 × parent-relative physics state
     + 0.30 × generated-subject appearance vs Case conditioned first frame
```

parent 视频只提供动力学 GT，不参与 OOD 外貌或像素位置评分。条件首帧从 Case 的
`assets.first_frame` 读取；prediction 第 0 帧 mask 仅作为不可变条件图上的主体 ROI，
mask 缺失或错误会得到保守低分。由于不存在真实 continuation，后续绝对位置和形状没有
可识别 GT，不会伪造这两项。

Scene score 是 reference 与 prediction 的相似度，不是 prediction 的绝对质量分。
所有正常 comparison 必须满足：

```text
identity:     S(x, x) = 1
range:        0 <= S(x, y) <= 1
sensitivity: 受评分维度发生差异时 S(x, y) < 1
```

每个组件使用 reference/prediction 差值：

```text
scaled_error = |prediction - reference| / protocol_scale
similarity   = exp(-scaled_error)
```

相对误差组件使用 reference 值和稳定的最小分母。`[0,1]` 比率使用
`1 - |prediction-reference|`。

scene physics state 仍使用 v2 的 reference-relative 组件。以下量不能单边扣
prediction：

- reference 自身的支点漂移或摆长波动；
- reference 自身的横向漂移；
- reference 自身的斜面离轨和姿态波动；
- reference 自身的非圆度或非匀速残差；
- reference 自身的动量残差或非一维漂移。

它们仍被完整报告为绝对诊断。正式 similarity 只比较 prediction 相对 reference
是否变差或发生变化。若将来需要纯物理模型分，应使用独立 `physics_model` evaluator，
不能混入 reference similarity。

## 4. 媒体归一化

Reference 与 prediction 可以有不同分辨率、FPS 和帧数，但必须覆盖相同物理区间。

各版共有的步骤：

1. probe source metadata；
2. 建立物理时间戳；
3. 用 `np.rint(time × source_fps)` 选择最近 source frame；
4. 按协议的 decode policy 解码；
5. 保持宽高比 resize；
6. letterbox 到 scene canvas；
7. 保存 source indices 和空间变换。

不会补 GT 首帧，不会重复末帧掩盖时长不足。

- v1 `legacy_random_seek`：按请求顺序对每个 index 执行一次
  `CAP_PROP_POS_FRAMES` seek，保留历史 decoder 行为。
- v2/v3 `sequential_forward`：从帧 0 解到最大目标 index，再按原时间戳顺序重组，避免
  对长视频重复 seek。

两种策略使用相同的 `np.rint` source index，也都保留非单调请求和重复 index；但不能
声称跨 codec 像素或失败语义等价。真实长 GOP HEVC 上，random seek 与从帧 0 顺序解码
可能得到不同像素，未请求中间帧损坏也只会阻断顺序解码。正因如此，两种策略属于不同
evaluator identity。

| scene | v2/v3 timeline | decode | 最小时长 | 最大时长 | canvas |
| --- | --- | --- | ---: | ---: | --- |
| pendulum | reference-bounded，16 Hz | sequential | 4.8 s | 5 s | 480 × 832 |
| free_fall | reference-bounded，32 Hz | sequential | 0.2 s | 1 s | 480 × 832 |
| inclined_plane_slide | reference-bounded，16 Hz | sequential | 1 s | 5 s | 640 × 480 |
| uniform_circular_motion | reference-bounded，8 Hz | sequential | 3 s | 5 s | 640 × 480 |
| collision_1d | reference-bounded，16 Hz | sequential | 1.5 s | 5 s | 640 × 360 |

所有源视频最低要求为 8 FPS。时间轴由 reference 的实际末帧时间决定，但不会超过
协议的最大时长。采样点保持协议 FPS，因此最后一个采样点是不晚于 reference 边界的
最大网格点。prediction 必须覆盖同一个时间区间；时长不足不会补帧或重复末帧：v3
返回 `evaluated/0/degraded`，v1/v2 保持历史 `unavailable`。

`scene_default_v4` 仅把碰撞画布提高到 `960 × 540`，以稳定小球边缘和接触阶段的实例
分割；时间轴、顺序解码、最小时长和最大时长保持不变。

单摆 v2/v3 的 4.8 秒下限覆盖当前冻结数据集中最短的可信 reference，同时仍提供足够的
周期观测区间。需要复现旧运行时必须显式使用 `scene_default_v1`，其固定 5 秒要求
不会被 v2 行为静默改写。

## 5. Reference 模式

优先级：

```text
same_case_reference
→ parent_physics_reference
→ unavailable
```

`parent_physics_reference` 只允许：

- case 存在 `parent_case_id`；
- parent 位于冻结 case catalog；
- structured physics 完全一致；
- parent reference 存在。

它是动力学 reference，不是同外观视觉 GT。v3 的外貌项使用 OOD Case 自己的条件首帧，
不使用 parent 像素。协议保留 `physics_model` 和 `reference_free` 模式，但当前五场景
正式动力学分数都使用可信视频 reference。

因此，没有同外观 GT 的 OOD1 case 仍可评估动力学：只要它与 parent 的 structured
physics 完全一致，就使用 parent 的真实运动作为物理 reference。背景、底座、颜色或
物体外观差异可能使 parent/生成原图 IoU 偏低，所以这条 IoU 只保留为诊断，不进入
正式分数。

## 6. Mask 与 IoU 语义

v3 中 Mask 同时承担两类职责：

1. 为各 scene 的物理状态轨迹提供观测；
2. 在同 Case GT 模式中形成 position/shape/appearance 的主体 ROI。

原始 `physical_subject_mask_iou` 曲线继续保留，既是 position 的一个子项，也是便于
人工审计的 Jensen 风格对照图；在 parent 模式中它只是跨外观诊断，不计分。v1/v2 中
Mask IoU 仍然只是诊断。

逐帧 IoU：

```text
IoU = intersection / union
```

当 reference 和 prediction mask 都为空时，IoU 是“未观测”：

- CSV 写 `null/空值`；
- 曲线显示断点；
- mean/min/max 只聚合 observed frames；
- `observed_frame_ratio` 明确报告覆盖率。

空/空既不是 0，也不是伪造的 1。跟踪质量由独立 valid ratio 门控。

## 7. 单摆

### 7.1 主体观测

单摆 evaluator 的观测链路为：

```text
独立运动区域 proposal
→ SAM2 分割完整单摆主体
→ mask 顶部估计支点
→ mask 底部估计摆球
→ 计算支点—摆球向量
→ 得到摆长和摆角时间序列
```

reference 和 prediction 首先分别生成自己的运动 proposal。如果 prediction 无法产生
可靠 proposal，只允许使用 reference 的几何 prompt 作为 SAM2 fallback；两侧的分割
传播、轨迹提取和状态拟合仍然独立执行。

每帧从 mask 提取：

```text
pivot_xy = 支点位置
bob_xy   = 摆球位置
length   = ||bob_xy - pivot_xy||
angle    = atan2(horizontal displacement, vertical displacement)
```

周期通过摆角序列的自相关估计，允许范围为 0.3–2.5 s。振幅定义为摆角相对其中位数
绝对偏移的 95% 分位数。

### 7.2 正式 metric

主 metric：

```text
pendulum_state_similarity
```

| component | 权重 | 含义 |
| --- | ---: | --- |
| `angle_trajectory` | 0.60 | 完整摆角轨迹 |
| `period` | 0.20 | 摆动周期 |
| `amplitude` | 0.10 | 摆角振幅 |
| `structural_consistency` | 0.10 | 支点漂移和摆长稳定性 |

摆角轨迹：

```text
angle_rmse = RMSE(theta_prediction(t), theta_reference(t))
angle_scale = max(reference_amplitude, 5 degrees)
angle_trajectory_score = exp(-angle_rmse / angle_scale)
```

周期：

```text
period_error = |T_prediction - T_reference| / T_reference
period_score = exp(-period_error)
```

reference 有可信周期而 prediction 无法估计周期时，`period=0`。reference 自身无法
可靠估计周期时，该组件从当前 case 移除，其余权重重新归一化。

振幅：

```text
amplitude_error =
    |A_prediction - A_reference| / max(A_reference, 1 degree)

amplitude_score = exp(-amplitude_error)
```

结构一致性使用：

```text
pivot_drift_ratio = 支点漂移标准差 / 中位摆长
length_cv         = 摆长标准差 / 摆长均值
```

比较 prediction 与 reference 的差值，而不是单边惩罚 prediction：

```text
pivot drift scale = 0.03
length CV scale   = 0.05

structural_consistency =
    0.5 × pivot_drift_similarity
  + 0.5 × length_cv_similarity
```

### 7.3 质量门控

- 有效 mask 比例至少为 0.90；
- 主体 mask 至少 20 像素，同时至少占画布 `0.00005`；
- 主体 mask 不得超过画布的 25%；
- 至少有 3 个有效观测点；
- 必须能估计有效支点、摆球和摆长。

不满足门控时不会基于不可信轨迹伪造正常分数：v3 prediction 侧返回
`evaluated/0/degraded`，reference 侧返回 `unavailable`；v1/v2 保持历史 `error`。

### 7.4 诊断和产物

诊断 metric：

```text
physical_subject_mask_iou
```

它报告 `mean/minimum/maximum/observed_frame_ratio`，但不进入
`pendulum_state_similarity`。同时保留 reference/prediction 的振幅、周期、摆角
RMSE、支点漂移、摆长变异系数和有效 mask 比例。

产物：

```text
per_frame.csv
physical_subject_iou_curve.png
result.json
```

`per_frame.csv` 包含逐帧时间、摆角、mask 面积、tracking valid 和主体 IoU。
IoU 曲线沿用 Jensen 风格的物理主体对照形式。

## 8. 自由落体

### 8.1 主体观测

```text
独立运动区域 proposal
→ SAM2 分割下落小球
→ 提取 mask 质心
→ 得到二维质心轨迹
→ 计算归一化竖直位移
→ 二次拟合 y(t)
```

竖直位移和归一化轨迹定义为：

```text
vertical(t) = y(t) - y(0)
span = max(|vertical(t)|)
normalized_vertical(t) = vertical(t) / span
```

使用二次多项式拟合：

```text
y(t) = c2 t² + c1 t + c0
acceleration_px_s2 = 2 × c2
normalized_acceleration_s2 = acceleration_px_s2 / span
```

`impact_time` 定义为主体第一次到达最大竖直行程 95% 的时间。

### 8.2 正式 metric

主 metric：

```text
free_fall_state_similarity
```

| component | 权重 | scale | 含义 |
| --- | ---: | ---: | --- |
| `vertical_trajectory` | 0.50 | 0.15 | 归一化竖直轨迹 |
| `normalized_acceleration` | 0.25 | 0.35 | 归一化加速度 |
| `impact_time` | 0.15 | 0.15 | 95% 行程到达时间 |
| `motion_constraints` | 0.10 | 组合项 | 横向漂移和向下单调性 |

轨迹：

```text
trajectory_error =
    RMSE(y_normalized_prediction(t), y_normalized_reference(t))

vertical_trajectory_score = exp(-trajectory_error / 0.15)
```

加速度：

```text
acceleration_error =
    |a_normalized_prediction - a_normalized_reference|
    / max(|a_normalized_reference|, epsilon)

normalized_acceleration_score = exp(-acceleration_error / 0.35)
```

到达时间：

```text
impact_error =
    |t_prediction - t_reference| / evaluation_duration

impact_time_score = exp(-impact_error / 0.15)
```

运动约束包括：

```text
horizontal_drift_ratio = horizontal_range / vertical_span
downward_progress_ratio = 没有明显反向上升的相邻帧比例
```

向下单调性允许 2% 行程跨度的测量噪声。横向漂移的 reference-relative scale 为
`0.08`，最终：

```text
motion_constraints =
    0.5 × horizontal_drift_similarity
  + 0.5 × downward_progress_similarity
```

### 8.3 质量门控

- 有效 mask 比例至少为 0.70；
- 主体 mask 至少 12 像素；
- mask 不得超过画布的 8%；
- 竖直运动跨度至少 8 像素。

### 8.4 诊断和产物

诊断 metrics：

```text
physical_subject_mask_iou
reference_physics_diagnostic
```

如果 case 有 `physics.initial_height`，evaluator 使用 reference 的像素行程做尺度标定，
报告 reference 拟合加速度的 SI 值。它只用于 reference calibration diagnostic，不会
用该标定猜测 prediction 的像素比例，也不进入正式分数。

reference 和 prediction 的二次拟合残差都会写入状态 metric，用于诊断轨迹是否接近
抛物线。

产物：

```text
per_frame.csv
physical_subject_iou_curve.png
vertical_trajectory_curve.png
result.json
```

## 9. 斜面下滑

### 9.1 主体观测

```text
独立运动区域 proposal
→ SAM2 分割滑块
→ 提取 mask 质心
→ 对质心轨迹执行 PCA
→ reference 和 prediction 分别拟合运动轴
→ 转换为 plane-local 坐标
```

plane-local 状态为：

```text
s(t) = 沿拟合斜面运动轴的位移
d(t) = 垂直运动轴的偏移
```

reference 和 prediction 独立拟合自己的轴，因此不要求斜面在两段视频中具有相同像素
位置、尺度或构图。

沿斜面位移归一化为：

```text
span = max(s) - min(s)
s_normalized(t) = s(t) / span
```

加速度只使用首次到达 90% 行程之前的片段拟合，避免末端碰撞或停止污染下滑阶段：

```text
s(t) = c2 t² + c1 t + c0
acceleration_px_s2 = 2 × c2
normalized_acceleration_s2 = acceleration_px_s2 / span
```

这里的 `initial_velocity` 是拟合量，不强制使用可能不精确的首帧速度标签。

### 9.2 正式 metric

主 metric：

```text
inclined_plane_state_similarity
```

| component | 权重 | scale | 含义 |
| --- | ---: | ---: | --- |
| `along_plane_trajectory` | 0.50 | 0.18 | 沿斜面的归一化轨迹 |
| `normalized_acceleration` | 0.25 | 0.40 | 归一化加速度 |
| `descent_time` | 0.15 | 0.18 | 95% 行程到达时间 |
| `contact_and_pose_constraints` | 0.10 | 组合项 | 离轨、倒退和姿态稳定性 |

轨迹：

```text
trajectory_error =
    RMSE(s_normalized_prediction(t), s_normalized_reference(t))

along_plane_trajectory_score = exp(-trajectory_error / 0.18)
```

加速度：

```text
acceleration_error =
    |a_normalized_prediction - a_normalized_reference|
    / max(|a_normalized_reference|, epsilon)

normalized_acceleration_score = exp(-acceleration_error / 0.40)
```

`descent_time` 是首次到达最大沿斜面行程 95% 的时间：

```text
time_error =
    |t_prediction - t_reference| / evaluation_duration

descent_time_score = exp(-time_error / 0.18)
```

接触和姿态约束：

```text
cross_track_std_ratio = std(d(t)) / span
monotonic_progress_ratio = 没有明显反向滑动的相邻帧比例
orientation_std_deg = mask 最小外接矩形姿态角的标准差
```

单调性允许 1.5% 行程跨度的噪声。子项组合为：

```text
contact_and_pose_constraints =
    0.45 × cross_track_similarity
  + 0.35 × monotonic_progress_similarity
  + 0.20 × orientation_stability_similarity
```

对应 reference-relative scale：

```text
cross-track scale = 0.05
orientation scale = 12 degrees
```

### 9.3 质量门控

- 有效 mask 比例至少为 0.75；
- 主体 mask 至少 20 像素；
- mask 不得超过画布的 15%；
- 沿斜面运动跨度至少 16 像素。

### 9.4 原图和 rectified IoU

斜面场景输出：

```text
physical_subject_mask_iou
plane_rectified_mask_iou
```

第一项直接在统一画布中计算。第二项分别根据 reference 和 prediction 的运动轴与行程
尺度，将 mask 变换到 canonical strip 后计算：

- 原图 IoU 同时反映画面位置、构图、尺度和主体外观；
- rectified IoU 更接近去除视角、位置和尺度后的主体重合；
- 两项都是诊断，不进入正式 case score。

### 9.5 物理标注诊断和产物

如果 case 提供 `theoretical_acceleration` 和 `calibration_length`，系统报告：

- 标注理论加速度；
- reference 视频拟合得到的 SI 加速度；
- `initial_velocity_treatment=fitted_not_trusted_from_label`。

理论加速度目前不进入正式 similarity。若视频的时间或长度标定不足以支持绝对比较，
系统不会强行生成一个理论物理分。

产物：

```text
per_frame.csv
physical_subject_iou_curve.png
along_plane_trajectory_curve.png
result.json
```

## 10. 匀速圆周运动

### 10.1 主体观测

当前圆周运动数据有稳定的绿色圆盘结构，因此该 evaluator 不使用 SAM2：

```text
HSV 定位绿色圆盘
→ 提取圆盘内部非绿色前景
→ 连通域候选
→ 跨帧连续性实例匹配
→ 圆拟合
→ 按轨道半径排序
```

预期对象数量来自 `case.appearance.object_count`，支持单物体和多物体。对每个物体：

1. 提取二维中心轨迹；
2. 拟合圆心与半径；
3. 计算逐帧极角；
4. 对角度 unwrap；
5. 减去首帧角度；
6. 对相对角轨迹做线性拟合。

初始相位没有可信标注，所以使用：

```text
relative_angle(t) = unwrap(theta(t)) - theta(0)
```

这样保留旋转方向、角位移和角速度，同时避免只因初始相位不同而扣分。

### 10.2 正式 metric

主 metric：

```text
uniform_circular_motion_state_similarity
```

| component | 权重 | scale | 含义 |
| --- | ---: | ---: | --- |
| `angular_trajectory` | 0.50 | 0.35 rad | 相对角轨迹 |
| `angular_velocity` | 0.25 | 0.25 | 拟合角速度 |
| `orbit_geometry` | 0.15 | 组合项 | 圆度和多轨道半径配置 |
| `uniform_motion` | 0.10 | 0.20 rad | 匀角速度拟合残差 |

相对角轨迹：

```text
angle_rmse =
    RMSE(relative_angle_prediction(t), relative_angle_reference(t))

angle_trajectory_score = exp(-angle_rmse / 0.35)
```

多物体时先为每个物体计算，再对物体取平均。

角速度由线性拟合得到：

```text
relative_angle(t) ≈ omega × t + b
omega_error =
    |omega_prediction - omega_reference|
    / max(|omega_reference|, epsilon)

angular_velocity_score = exp(-omega_error / 0.25)
```

轨道几何：

```text
radial_cv =
    轨迹点到拟合圆心的半径标准差 / 拟合半径
```

prediction 与 reference 的 `radial_cv` 差异 scale 为 `0.08`。多物体半径分别除以
当前视频中的最大轨道半径，得到 `normalized_radii`；两侧半径配置的 RMSE scale 为
`0.12`。最终：

```text
orbit_geometry =
    0.5 × mean_object_circularity_similarity
  + 0.5 × radius_configuration_similarity
```

匀速性使用相对角轨迹对线性角速度模型的 RMSE：

```text
angular_fit_rmse =
    RMSE(relative_angle(t), omega × t + b)
```

正式分数比较 prediction 与 reference 的拟合残差差异，scale 为 `0.20 rad`，不是
单边惩罚 prediction 的绝对残差。

### 10.3 质量门控

- 绿色圆盘至少占画布的 5%；
- 必须持续观测到预期数量的物体；
- 每个实例的有效跟踪比例至少为 0.90；
- 单个候选物体面积不得超过画布的 3%。

### 10.4 原图和 rectified IoU

输出：

```text
physical_subject_mask_iou
orbit_rectified_mask_iou
```

原图 IoU 使用所有运动物体 mask 的 union。rectified IoU 分别使用 reference 和
prediction 拟合的外层轨道圆心与半径，将两侧 mask 变换到统一轨道坐标后计算，从而
减弱轨道中心、整体尺度和构图差异。两项都不进入正式 case score。

### 10.5 标注诊断和产物

如果 case 有 `angular_velocity_rad_s`，它会作为 annotation diagnostic 保存；正式
角速度分数仍然比较 prediction 与真实 reference 的拟合结果。

产物：

```text
per_frame.csv
physical_subject_iou_curve.png
angular_trajectory_curve.png
result.json
```

## 11. 一维碰撞

11.1–11.6 记录 v1–v3 的冻结逻辑。v4 不修改这些历史协议，而是在 11.7–11.9 所述的
独立 evaluator `1.4` 中替换碰撞主体观测并增加可靠性与可视化。

### 11.1 主体观测

当前场景固定跟踪三个角色：

```text
striker
target_1
target_2
```

观测链路：

```text
frame 0 轨道区域提取彩色组件
→ 识别左侧 striker
→ 识别相邻 target pair
→ 为三个实例建立 SAM2 prompt
→ 在同一 SAM2 state 中传播三个实例 mask
→ 提取三个实例的质心轨迹
```

reference 和 prediction 分别执行 frame-0 对象定位。三个物体的全部轨迹点用于 PCA
拟合共同轨道轴，然后投影为：

```text
s_i(t) = 第 i 个物体沿轨道轴的位置
d_i(t) = 第 i 个物体垂直轨道轴的位置
```

每个物体的位置减去自己的初始位置，再除以整个场景的轨迹跨度：

```text
normalized_position_i(t) =
    (s_i(t) - s_i(0)) / scene_span
```

### 11.2 事件和速度

接触事件定义为 striker 与 `target_1` 距离最小的帧。接触前后分别选取时间窗口，对
三个实例的归一化位置做线性拟合：

```text
pre_velocity_normalized_s
post_velocity_normalized_s
```

窗口长度为总帧数的 20%，且至少包含 3 帧。

### 11.3 正式 metric

主 metric：

```text
collision_1d_state_similarity
```

| component | 权重 | scale | 含义 |
| --- | ---: | ---: | --- |
| `instance_trajectories` | 0.45 | 0.12 | 三个实例的位置轨迹 |
| `contact_event_time` | 0.15 | 0.10 | 碰撞事件时刻 |
| `pre_post_velocities` | 0.20 | 0.25 | 碰撞前后三球速度 |
| `collision_physics` | 0.15 | 组合项 | 动量残差和恢复系数 |
| `one_dimensional_constraint` | 0.05 | 0.03 | 一维轨道约束 |

实例轨迹：

```text
trajectory_error =
    mean(
        NRMSE(object_1),
        NRMSE(object_2),
        NRMSE(object_3)
    )

instance_trajectories_score = exp(-trajectory_error / 0.12)
```

角色从 frame 0 开始按 `striker/target_1/target_2` 保持，不以 union mask 代替实例身份。

事件时刻：

```text
event_error =
    |t_event_prediction - t_event_reference|
    / evaluation_duration

contact_event_time_score = exp(-event_error / 0.10)
```

前后速度：

```text
velocity_rmse =
    RMSE(concat(v_pre, v_post)_prediction,
         concat(v_pre, v_post)_reference)

velocity_error =
    velocity_rmse / max(|reference velocities|)

pre_post_velocities_score = exp(-velocity_error / 0.25)
```

### 11.4 碰撞物理

使用 case 的三个质量标注：

```text
ball_1_mass
ball_2_mass
ball_3_mass
```

计算：

```text
momentum_before = Σ(m_i × v_i_before)
momentum_after  = Σ(m_i × v_i_after)

momentum_residual =
    |momentum_after - momentum_before|
    / Σ|m_i × v_i_before|
```

正式比较 prediction 与 reference 的动量残差差异，scale 为 `0.20`。

有效恢复系数由视频轨迹估计：

```text
effective_restitution =
    separation_speed / closing_speed
```

当前数据没有可信恢复系数标签，所以不会根据材质猜值。reference 可以估计恢复系数时：

```text
collision_physics =
    0.6 × momentum_residual_similarity
  + 0.4 × restitution_similarity
```

恢复系数相对误差 scale 为 `0.25`。reference 无法估计恢复系数时，只使用动量残差
相似度；reference 有而 prediction 无法估计时，恢复系数子项为 0。

一维约束：

```text
cross_track_std_ratio =
    三个物体 cross-track std 的均值 / scene_span
```

比较 prediction 与 reference 的差值，scale 为 `0.03`。

### 11.5 质量门控

- frame 0 必须可靠定位三个角色；
- 三个实例的有效跟踪比例均至少为 0.80；
- 每个 mask 至少 12 像素；
- 单个 mask 不得超过画布的 3%；
- 整体运动跨度至少 20 像素。

### 11.6 IoU 诊断和产物

输出：

```text
physical_subject_mask_iou
matched_instance_mask_iou
```

第一项对三个实例 mask 的 union 计算。第二项分别比较
`striker/target_1/target_2`，报告总体均值、每实例均值和每实例 observed frame
ratio，用于诊断身份交换或跟踪丢失。实例两侧 mask 同时为空时仍按“未观测”处理。

产物：

```text
per_frame.csv
physical_subject_iou_curve.png
striker_trajectory_curve.png
result.json
```

### 11.7 v4 多帧角色观测

v4 不再假设三个球能在 frame 0 通过高饱和度颜色组件被发现。reference 与 prediction
各自执行：

```text
在视频前 60% 的最多 12 个候选帧中搜索
→ 在 960 × 540 轨道带内做多阈值 Hough 圆检测
→ 按“左侧 striker + 相邻 target_1/target_2 + 共线轨道”选择三元组
→ Hough 全部失败时使用 temporal-median motion proposal
→ 在共同 seed frame 建立三个带固定语义角色的 SAM2 prompt
→ 从 seed 向前和向后传播
→ 每个像素只分配给最高正 SAM2 logit 的一个实例
→ 以角色 seed 和质心连续性保留每帧单一连通分量
```

seed 可以晚于 frame 0，所以入镜较晚、首帧部分遮挡或首帧颜色不显著不再直接使
evaluator 失败。三个角色始终是 `striker/target_1/target_2`；不会通过自由 Hungarian
重排把身份交换误判为高分。接触阶段的实例重叠以最高正 logit 决定唯一归属，避免同一
像素同时进入多个球 mask。

共享 SAM2 适配器的 `exclusive_masks` 默认仍为 `false`。只有 v4 碰撞显式启用互斥
分配；v1–v3 的 frame-0、独立二值阈值和单向传播行为保持冻结。

### 11.8 v4 质量与可靠性

v4 的 reference 每个角色最少需要 20% 的有效观测帧，每个 mask 至少 12 像素、不得
超过画布的 2%，整体运动跨度至少 30 像素。20% 是对冻结 View B 全部 32 个真实
reference 完整运行后设置的 hard gate；该批次最低角色有效率为 `0.2567567568`。

prediction 观测失败仍按 v3 责任语义得到有限零分，不转化为 evaluator error。对于能
形成轨迹但比 reference 丢失更多观测的 prediction，定义：

```text
coverage_role =
    clip(prediction_valid_ratio_role / reference_valid_ratio_role, 0, 1)

observation_reliability = min(coverage_striker, coverage_target_1,
                              coverage_target_2)

collision_state_score =
    raw_collision_state_score × observation_reliability
```

这样避免旧式单一阈值 cliff，同时不允许低覆盖 prediction 仅凭少数幸运帧取得高物理
分。正常主体位置、形状、外貌评分与 v3 相同；可靠性只作用于碰撞 state 子分。

### 11.9 v4 过程可视化

每个成功或可保守退化的 v4 碰撞 Case 都会 best-effort 生成：

```text
collision_observation.mp4
collision_instance_similarity.png
collision_event_timeline.png
collision_tracks.csv
collision_observation.json
```

视频为 2×2 面板：reference 角色 mask/质心/轨迹、prediction 对应视图、union mask
重合图，以及逐帧 IoU、seed、接触帧和角色状态 dashboard。两张图分别展示三角色及
union IoU 曲线、三角色归一化轨迹和 reference/prediction 接触时刻。

大文件写入 `PHYSBENCH_VISUALIZATION_ROOT`，默认是：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_visualizations
```

仓库顶层 `visualizations` 是该目录的本机链接。正式 Case artifact 目录只保存
`collision_visualization_manifest.json`，其中记录外部绝对路径、仓库链接路径、
文件大小、SHA-256 和 evaluator config digest。外置可视化属于未封印诊断，不进入
AtomicRun/reevaluation 的 sealed artifact manifest；生成失败也不会改变 Case 分数或
状态。

## 12. 产物

```text
runs_v2/<run_id>/evaluation/
├── manifest.json
├── case_results.jsonl
├── task_result.json
├── case_metrics.jsonl              # compatibility projection
├── summary.json                    # compatibility projection
└── cases/<job_id>/
    ├── result.json
    ├── per_frame.csv
    ├── subject_components.csv       # v3
    ├── physical_subject_iou_curve.png
    ├── subject_similarity_curve.png # v3
    ├── collision_visualization_manifest.json # v4 collision
    └── <scene_state_curve>.png
```

v4 碰撞的大型过程视频、图和轨迹表不放入上述 run 目录；本地 manifest 通过 SHA-256
把它们关联到 `visualizations/scene_default_v4/...`。

不是每个 scene 都有额外 state curve。当前精确映射为：

| scene | state curve |
| --- | --- |
| pendulum | 无；摆角逐帧值在 `per_frame.csv` |
| free_fall | `vertical_trajectory_curve.png` |
| inclined_plane_slide | `along_plane_trajectory_curve.png` |
| uniform_circular_motion | `angular_trajectory_curve.png` |
| collision_1d | `striker_trajectory_curve.png` |

每个 case 的 `result.json` 和 `case_results.jsonl` 记录：

- evaluator ID、版本和 fingerprint；
- case status、reason code 和 score；
- 正式 component scores、权重和中间误差；
- reference/prediction 物理观测量；
- mask 和 tracking 质量；
- reference 模式；
- reference/prediction 视频路径和 SHA-256；
- 实际使用的 source frame indices；
- 空间 resize/letterbox transform；
- 分割、跟踪和 rectification provenance；
- CSV 和曲线 artifact 路径。

`task_result.json` 记录 coverage、状态计数、partition/scene breakdown、严格 Task score
和部分观察分数。v3 还记录 `robustness`：退化零分数目与比例、reference unavailable
数、真正 evaluator error 数以及退化原因码分布。

## 13. Task 聚合

### 13.1 主表和完整性

TaskEvaluator 以 frozen canonical plan 的 `jobs` 为唯一主表，而不是以
`predictions.jsonl` 中实际出现的记录为主表。因此：

- 缺失 prediction 会产生显式 case result；v3 为退化零分，v1/v2 为 `unavailable`；
- 重复 prediction 会产生显式 `error`；
- 未知 job 的 prediction 会写入 `integrity_issues`；
- 失败 job 不会从分母中静默消失。

### 13.2 Partition 分数

`train_seen` 是训练集记忆诊断，不进入正式聚合。

每个正式 partition 只有 coverage 为 1 时才有正式分数：

```text
partition_score = mean(all evaluated case scores)
```

任意 case 缺失或失败时：

```text
partition_score = null
```

系统同时报告 `observed_mean_score`，它只汇总已经成功评估的部分，不得用作正式排名。

### 13.3 Scene 分数

`finetune_eval` 对 scene 内所需 partition 做宏平均：

```text
scene_score =
    macro_mean(ID score, OOD1 score, ...)
```

这避免 case 数较多的 partition 主导结果。

`direct_eval` 对 scene 中全部正式 case 求均值：

```text
scene_score = mean(all official case scores in this scene)
```

direct task 的 group 仍会保留 breakdown，但只是诊断维度，不改变 scene 的正式权重。

### 13.4 Task 总分

Task 对选中的 scene 做宏平均：

```text
task_score =
    macro_mean(scene_1_score, ..., scene_N_score)
```

只有每个正式 scene 都具有完整分数时，Task 才有正式总分。任意 scene 不完整时：

```text
task_score = null
status = partial
```

此时 `observed_mean_score` 仍可帮助检查部分运行，但它不是可比较的 Benchmark 总分。

## 14. Metrics 总表

### 14.1 正式 scene metrics

v3 五个 scene 的主 metric 都是 `scene_subject_state_similarity`。它组合共同的
`physical_subject_similarity` 与下表的 scene state 子 metric：

| scene | scene state 子 metric | components |
| --- | --- | --- |
| pendulum | `pendulum_state_similarity` | angle trajectory、period、amplitude、structural consistency |
| free_fall | `free_fall_state_similarity` | vertical trajectory、normalized acceleration、impact time、motion constraints |
| inclined_plane_slide | `inclined_plane_state_similarity` | along-plane trajectory、normalized acceleration、descent time、contact and pose constraints |
| uniform_circular_motion | `uniform_circular_motion_state_similarity` | angular trajectory、angular velocity、orbit geometry、uniform motion |
| collision_1d | `collision_1d_state_similarity` | instance trajectories、contact event time、pre/post velocities、collision physics、one-dimensional constraint |

v1/v2 仍以下表 scene state metric 作为各自历史主 metric。

### 14.2 共同诊断 metric

五个 scene 都有：

```text
physical_subject_mask_iou
```

它包含：

```text
mean
minimum
maximum
observed_frame_ratio
role = position_component_and_required_diagnostic  # v3 same-case
     | parent_reference_visual_diagnostic_not_scored
     | diagnostic_not_primary_score                # v1/v2
```

### 14.3 Scene-specific diagnostics

```text
free_fall:
    reference_physics_diagnostic

inclined_plane_slide:
    plane_rectified_mask_iou
    reference_physics_diagnostic

uniform_circular_motion:
    orbit_rectified_mask_iou
    annotation_diagnostic

collision_1d:
    matched_instance_mask_iou
    annotation_policy
```

单摆的振幅、周期、支点漂移和摆长稳定性已经作为主 metric 的详细字段保存。

### 14.4 与旧 metrics 框架的边界

仓库仍保留早期的：

```text
common_sense
prediction
visual_judgment
```

其默认配置位于 `configs/metrics/default.json`，目前都是 disabled placeholder。当前
当前官方 AtomicRun 的正式五场景分数来自 `scene_default_v3` 下的 scene-specific evaluator，
不是这三个旧 placeholder metric。

### 14.5 论文依据与工程取舍

v3 的主体层采用可审计、无额外训练的组合指标：

- DAVIS 的 region similarity \(J\) 与 contour accuracy \(F\) 说明区域重合和边界质量
  应分开观察；v3 的 shape 因此组合 canonical mask IoU 与 tolerance-aware boundary F。
- SSIM 提供亮度、对比度和结构联合比较的经典构造；v3 在主体 canonical crop 的共同
  mask 上使用稳定的 SSIM-style 项，并与 Lab 颜色和梯度纹理互补。
- LPIPS 说明深特征距离通常比单纯像素距离更符合感知判断，但它依赖固定 backbone、
  权重和预处理，并可能给小型实验物体带来域偏差。v3 暂不把 LPIPS 作为硬依赖，避免
  模型下载或设备问题使 Case 失败；以后只能作为版本化协议中的新增组件。
- SAM 2 为当前视频 mask 传播提供主体层；CoTracker 与 TAP-Vid 提供点跟踪和遮挡
  benchmark 的相关思路。后续若增加 fallback tracker，必须记录 backend identity，
  并发布新 evaluator 版本，不能静默改变 v3。

参考：

1. Perazzi et al., *A Benchmark Dataset and Evaluation Methodology for Video
   Object Segmentation*, CVPR 2016 (DAVIS).
2. Wang et al., *Image Quality Assessment: From Error Visibility to Structural
   Similarity*, IEEE TIP 2004.
3. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric*, CVPR 2018 (LPIPS).
4. Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024.
5. Karaev et al., *CoTracker: It is Better to Track Together*, ECCV 2024.
6. Doersch et al., *TAP-Vid: A Benchmark for Tracking Any Point in a Video*,
   NeurIPS 2022.

## 15. 回归验证

每个 scene scorer 必须有两类测试：

1. 非理想 reference 与自身比较，精确得到 1；
2. 只扰动一个被评分状态，结果严格小于 1。

真实视频自比还要验证：

- reference 与 prediction 独立走完整媒体和观测管线；
- case score 为 1；
- observed mask IoU 为 1；
- 未观测帧不会污染 IoU 均值；
- evaluator fingerprint 与协议一致。

主要回归测试位于：

```text
tests/test_scene_evaluation.py
tests/test_metrics.py
tests/test_evaluation_protocol_v3.py
tests/test_evaluation_protocol_v4.py
```

实现或修改 evaluator 后至少执行：

```bash
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -p 'test_scene_evaluation.py' -v

PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v
```

v4 还提供真实 reference 可观测性审计：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_collision_evaluator_v4.py \
  --device cuda \
  --output visualizations/scene_default_v4/\
collision_reference_observability_audit.json
```

该脚本只观测冻结 reference，不生成或修改 prediction。当前 View B 结果为 32/32
observable；完整验证记录见
[`docs/experiments/COLLISION_EVALUATOR_V4_20260730.md`](experiments/COLLISION_EVALUATOR_V4_20260730.md)。

## 16. AtomicRun 并存式重评

已封印 AtomicRun 的 canonical evaluation 不允许原地覆盖。重评必须显式指定目标协议和
本次评估 ID：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate \
  --run-dir runs_v2/RUN_ID \
  --protocol-id scene_default_v3 \
  --evaluation-id protocol-v3-audit-001
```

输出目录由协议内容指纹确定：

```text
runs_v2/RUN_ID/reevaluations/
└── scene_default_v3/
    └── <protocol_sha256>/
        └── protocol-v3-audit-001/
            ├── reevaluation.json
            ├── protocol.json
            ├── source_integrity.json
            ├── reference_assets.json
            ├── artifact_manifest.json
            ├── report.md
            └── evaluation/
```

同一 `(protocol_id, protocol_sha256, evaluation_id)` 使用独占创建，重复执行会失败，不会
覆盖旧变体。协议 ID 和 evaluation ID 必须是 path-safe identifier。AtomicRun 内部的
canonical source tree 和 reevaluation 目标路径不允许 symlink、路径穿越或逃逸；
显式传入的 `run-dir` 会先 resolve。Dataset reference 可以使用解析后仍位于
`asset_root` 内部的 symlink，但不能逃逸。

### 16.1 Preflight 信任边界

创建变体前必须全部通过：

1. `task_instance/manifest.json` 的 seal 与 digest；
2. `plan.json` 与 sealed canonical plan 完全一致；
3. `frozen/task.json` digest 与 sealed Task identity 一致；
4. 从 `source.asset_root/releases/<release>/dataset.json` 重新加载原始 release，
   重算 digest 并与 sealed Dataset identity 对齐；frozen descriptor、cases、views 和
   asset lock 必须与该 release 完全一致。原始 release 不可用时拒绝重评；
5. `predictions.jsonl` 覆盖全部 frozen jobs，identity、状态和 run-local artifact
   与 `artifacts/prediction_artifacts.json` 一致；
6. canonical native protocol 的实际指纹与
   `component_fingerprints.evaluation_protocol` 一致；
7. native `case_results.jsonl` 的每条记录通过 `CaseEvaluationResult` contract，与
   `evaluation/cases/<job>/result.json` 及 compatibility projection 完全一致，并由
   当前 Task 聚合器重算得到同一个 `task_result.json`；
8. 目标协议实际会评估的 complete prediction 所消费的 same-case 或 OOD parent
   physics reference，其路径、角色、case binding、大小和 SHA-256 必须与已认证的
   asset lock 一致。

Native evaluation 可以是严格意义上的 `partial`，例如 prediction 都是 `planned`；但它
不能缺 prediction 或 case-result 记录。目标 v3 对 `planned`、`failed` 等非-complete
prediction 给显式退化零分；v1/v2 保持 `unavailable`。非-complete prediction 或目标
协议标为 unsupported 的 job 不会提前读取其 reference，避免把不会被 evaluator 消费的
资产变成额外 blocker。

### 16.2 Provenance 与失败审计

每个变体固定保存：

- 目标 protocol 原文快照与 fingerprint；
- canonical AtomicRun 源文件 size/SHA-256 manifest；
- sealed Dataset release digest 的重算结果及四份 frozen 文档 digest；
- 实际 reference 及 OOD parent binding；
- evaluator source tree hash、Git commit 和限定路径 dirty status；
- Python 与相关 package 版本；
- 本地可解析的 SAM2 snapshot、revision 与 weight SHA-256；
- 变体内全部常规文件的 artifact manifest。

Dataset 输入有 sealed digest 作为真实性锚。Prediction 和 native evaluation 是运行时
输出，历史 AtomicRun 没有外部签名；preflight 能证明其 artifact/明细/聚合在当前
canonical run 内部自洽，不能把它们描述成“外部已签名”。这个边界会明确写入
`source_integrity.json` 的 `authenticity_boundary`。

若 evaluator 中途失败，变体目录仍保留，`reevaluation.json` 标记
`workflow_status=failed` 并记录异常类型与消息；失败前已写出的 case/summary 产物也会
进入 artifact manifest。失败 ID 同样不能重用。

以下 canonical 内容在该流程中始终只读：

```text
evaluation/
run.json
report.md
state.json
component_fingerprints.json
```

因此同一 AtomicRun 的 v1、v2、v3 或未来协议结果可以并存审计，但不得跨 fingerprint
混合聚合。

旧的 Python 符号 `physbench.orchestration.reevaluate_atomic` 仅作为 fail-closed
兼容守卫保留：调用它会报错并指向 `reevaluate_atomic_variant`，不会再写 canonical
目录。只有没有 AtomicRun v2 markers 且 `run.json.schema_version=1.0` 的历史 run
可以继续通过 `physbench.runner.reevaluate_run` 或 CLI legacy 分支原地重建其旧评估
产物。

2026-07-29 的七组冻结预测批量审计、逐 scene/partition 结果和完整性核验见：

```text
docs/experiments/EVALUATION_V3_20260729.md
```

## 17. 实现位置

正式协议与 Task 聚合：

```text
configs/evaluation/protocols/scene_default_v3.json
configs/evaluation/protocols/scene_default_v2.json  # frozen v5
configs/evaluation/protocols/scene_default_v1.json  # frozen legacy
src/physbench/evaluation/task_evaluator.py
src/physbench/evaluation/common/
src/physbench/orchestration/evaluation_variants.py
```

五个 scene evaluator：

```text
src/physbench/evaluation/scenes/pendulum/
src/physbench/evaluation/scenes/free_fall/
src/physbench/evaluation/scenes/inclined_plane/
src/physbench/evaluation/scenes/circular_motion/
src/physbench/evaluation/scenes/collision/
```
