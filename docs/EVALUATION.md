# 五场景评估协议

## 1. 边界与输入

正式评估属于 Benchmark。TaskEvaluator 消费：

```text
CanonicalTaskPlan + frozen cases + predictions.jsonl + evaluation protocol
```

当前协议：

```text
configs/evaluation/protocols/scene_default_v1.json
```

协议阈值版本仍是 `scene_default_v1`；五个 scene evaluator 的当前实现版本是 `1.1`。
`1.1` 将 reference-relative similarity 与 reference 自身的绝对物理诊断分离，并把
实现版本写入 evaluator 指纹，修复前后的结果不能混合。

`plan.jobs` 是主表。缺失 prediction、重复 prediction、失败生成或未知 case 都必须产生
一个显式 case result。

## 2. Case 状态

| 状态 | 含义 | score |
| --- | --- | --- |
| `evaluated` | 成功得到可信物理状态相似度 | `[0,1]` |
| `unavailable` | prediction、时长或可信 reference 不可用 | `null` |
| `unsupported` | 协议未实现该 scene | `null` |
| `error` | 解码、分割、跟踪或质量校验失败 | `null` |

错误和不可用不能转换为零分。

## 3. 评分契约

Scene score 是 reference 与 prediction 的相似度，不是 prediction 的绝对质量分。
所有 evaluator 必须满足：

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

以下量不能再单边扣 prediction：

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

公共步骤：

1. probe source metadata；
2. 建立物理时间戳；
3. 分别按时间戳取最近 source frame；
4. 保持宽高比 resize；
5. letterbox 到 scene canvas；
6. 保存 source indices 和空间变换。

不会补 GT 首帧，不会重复末帧掩盖时长不足。

| scene | timeline | canvas |
| --- | --- | --- |
| pendulum | 固定 0–5 s，16 Hz，81 点 | 480 × 832 |
| free_fall | reference-bounded，32 Hz，最多 1 s | 480 × 832 |
| inclined_plane_slide | reference-bounded，16 Hz，最多 5 s | 640 × 480 |
| uniform_circular_motion | reference-bounded，8 Hz，最多 5 s | 640 × 480 |
| collision_1d | reference-bounded，16 Hz，最多 5 s | 640 × 360 |

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

它是动力学 reference，不是同外观视觉 GT。协议保留 `physics_model` 和
`reference_free` 模式，但当前五场景正式分数都使用可信视频 reference。

## 6. Mask 与 IoU 语义

Mask 是状态观测和诊断层，不是正式 case score。

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

观测：

- 独立运动 proposal；
- SAM2 reference/prediction 分割；
- mask 顶部估计支点；
- mask 底部估计摆球；
- 计算摆长与摆角。

正式组件：

- 摆角轨迹；
- 周期；
- 振幅；
- reference-relative 支点漂移与摆长波动。

IoU 曲线保留 Jensen 的物理主体对照形式，但不是正式分数。

## 8. 自由落体

观测：

- SAM2 单球 mask；
- 质心轨迹；
- `y(t)` 二次拟合；
- 95% 行程到达时间。

正式组件：

- 归一化竖直轨迹；
- 归一化加速度；
- impact time；
- reference-relative 横向漂移与向下单调性。

初始高度只用于 reference SI 加速度诊断，不用于猜测 prediction 像素比例。

## 9. 斜面下滑

观测：

- SAM2 滑块 mask；
- 质心 PCA 拟合斜面运动轴；
- plane-local `s(t), d(t)`；
- mask 最小外接矩形姿态；
- 到达 90% 行程前拟合加速度。

正式组件：

- 沿斜面归一化轨迹；
- 归一化加速度；
- descent time；
- reference-relative 接触漂移、单调性和姿态稳定性。

输出原图 IoU 与独立轴/尺度 rectified IoU。`initial_velocity` 作为拟合量。
理论加速度只作为标注诊断；若 reference 时间或尺度标定不支持绝对比较，不强行进入分数。

## 10. 匀速圆周运动

观测：

- 定位绿色圆盘；
- 提取盘内非绿色运动物体；
- 连续性实例匹配；
- 单/双物体圆拟合；
- 按轨道半径排序。

初始角未标注，因此使用：

```text
relative_angle(t) = unwrap(theta(t)) - theta(0)
```

正式组件：

- 相对角轨迹；
- 角速度；
- reference-relative 圆度；
- 多物体半径配置；
- reference-relative 匀速拟合残差。

输出原图 union IoU 和中心/半径 rectified IoU。

## 11. 一维碰撞

观测：

- frame 0 轨道色彩组件定位 striker 与相邻 target pair；
- 同一 SAM2 state 传播三个实例；
- PCA 拟合共同轨道轴；
- 计算标量实例轨迹和接触时刻；
- 接触前后窗口拟合速度。

正式组件：

- 三实例轨迹；
- 接触时刻；
- 前后速度；
- reference-relative 动量残差；
- reference 估计恢复系数相似度；
- reference-relative 一维轨道约束。

数据没有可信恢复系数标签，因此不按材质猜值。输出 union IoU 和逐实例 IoU；
实例空/空帧按未观测处理。

## 12. 产物

```text
evaluation/
├── manifest.json
├── case_results.jsonl
├── task_result.json
└── cases/<job_id>/
    ├── result.json
    ├── per_frame.csv
    ├── physical_subject_iou_curve.png
    └── <scene_state_curve>.png
```

Case result 包含 evaluator 指纹、score、组件、绝对诊断、质量、artifact 和完整采样
provenance。

## 13. Task 聚合

- 排除 `train_seen` 预览 job；
- partition 只有 coverage 为 1 时才有正式分数；
- `finetune_eval` 在 scene 内对 ID/OOD1 做宏平均；
- `direct_eval` 在 scene 内对所有 group case 求均值；
- Task 对 scene 做宏平均；
- 任一正式 scene 不完整时 Task `score=null`；
- `observed_mean_score` 明确标记为部分观察结果。

## 14. 回归验证

每个 scene scorer 必须有两类测试：

1. 非理想 reference 与自身比较，精确得到 1；
2. 只扰动一个被评分状态，结果严格小于 1。

真实视频自比还要验证：

- reference 与 prediction 独立走完整媒体和观测管线；
- case score 为 1；
- observed mask IoU 为 1；
- 未观测帧不会污染 IoU 均值；
- evaluator fingerprint 与协议一致。
