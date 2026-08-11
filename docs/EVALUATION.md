# Evaluation v1

发行版只有一个公开协议：
`configs/evaluation/protocols/scene_default_v1.json`。规范位于
`schemas/evaluation_protocol.schema.json`。历史协议编号不是发布接口；当前 v1 已合并并
固定重构前最新实现的全部评分语义。

## Public evaluator routing

| Scene | Public type | Current implementation |
|---|---|---|
| `pendulum` | `pendulum_v1` | fail-closed frozen subject identity + open-world tracking |
| `collision_1d` | `collision_1d_v1` | fail-closed N-body identity and collision scoring |
| `inclined_plane_slide` | `inclined_plane_slide_v1` | causal open-world block tracking |
| `uniform_circular_motion` | `uniform_circular_motion_v1` | apparatus-frozen orbit tracking |
| `parabolic_motion` | `parabolic_motion_v1` | fail-closed projectile identity and strict composition |

所有公开 evaluator 的 `evaluator_version` 都是 `1.0`。源码中保留少量版本化辅助模块，
因为当前算法复用它们；它们不是可选公共协议，也不能通过 registry 选择。

## Shared contract

每个 scene evaluator 接收一个冻结 job、对应 Case、完整 case catalog、prediction、只读
asset root、输出目录和该 scene 的协议配置。它输出：

- 明确的 `evaluated`、`unavailable`、`protocol_error` 或 `error` 状态；
- scene-local expert score 与可审计组件；
- evaluator ID、版本和配置 fingerprint；
- 可选 visualization 与 identity/observation 诊断。

Evaluator 是唯一允许读取 reference video、reference masks 和评分注释的组件。Baseline
投影会排除这些字段，并审计媒体摘要，防止 V2V 输入别名到参考媒体。

## Five scene scores

- Pendulum 比较角轨迹、周期、振幅、结构一致性和条件主体身份；冻结首帧主体身份，预测
  不能用后来出现的相似物体替换目标。
- Collision 在冻结参与者身份后比较实例轨迹、接触时刻、碰撞前后速度、动量/恢复系数
  和一维约束；意外参与者或身份丢失会降分并留下诊断。
- Inclined plane 比较沿斜面轨迹、归一化加速度、下降时间、接触/姿态，并使用条件因果的
  open-world observation。
- Uniform circular motion 冻结转盘几何与相位基准，比较角轨迹、角速度、轨道几何、
  匀速性和对象完整性。
- Parabolic motion 冻结抛射体主体身份，比较轨迹、水平速度、重力加速度、飞行时间和
  运动约束；physics 与 identity 采用严格乘法组合，任一关键门失败都不能被另一项补偿。

## CSTI

CSTI 使用 `exact_full_tube_edt`：在 scene 原生分析分辨率上构造参考与预测实体 tube，排除
配置指定的初始条件帧，按物理重叠时间轴计算带空间/时间容差的完整 tube 一致性。case
层按 GT entity 求均值，Task 层作为独立维度汇总；CSTI 不替代 scene expert score。

## Failure semantics

协议的预测侧策略是 fail-closed but finite：缺失记录、无效媒体或预测 observation 失败
产生零分 case，并计入 `degraded_evaluated_jobs` 与 reason counts。这样提交缺失不会通过
降低分母获益。

Reference 缺失或不可认证时结果为 `unavailable`；内部 evaluator 异常为 `error`。这些
状态不会转换成成功分数。所有 case 结果都必须对应 canonical job，重复、越界或身份不符
的记录会被拒绝。

## Aggregation

Task 汇总按 canonical `scene_ids` 计算五场景宏平均，同时报告逐场景、partition、状态、
coverage、observed mean、CSTI 和 robustness 诊断。报告策略来自 Task v1；Evaluator 不得
扩大或缩小评估集合。
