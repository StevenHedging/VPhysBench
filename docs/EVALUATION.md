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
| `vertical_spring_oscillator` | `vertical_spring_oscillator_v1` | frozen steel-ball identity + vertical spring dynamics/topology |

所有公开 evaluator 的 `evaluator_version` 都是 `1.0`。源码中保留少量版本化辅助模块，
因为当前算法复用它们；它们不是可选公共协议，也不能通过 registry 选择。

协议和两个官方 Task v1 都选择上述六个 scene evaluator；Task 文件仍是正式 job 集的唯一
选择权威，registry 不能自行增删评估 case。

## Shared contract

每个 scene evaluator 接收一个冻结 job、对应 Case、完整 case catalog、prediction、只读
asset root、输出目录和该 scene 的协议配置。它输出：

- 明确的 `evaluated`、`unavailable`、`protocol_error` 或 `error` 状态；
- scene-local expert score 与可审计组件；
- evaluator ID、版本和配置 fingerprint；
- 可选 visualization 与 identity/observation 诊断。

Evaluator 是唯一允许读取 reference video、reference masks 和评分注释的组件。Baseline
投影会排除这些字段，并审计媒体摘要，防止 V2V 输入别名到参考媒体。

## Scene expert scores

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
- Vertical spring oscillator 冻结条件帧中的钢球身份，在 reference 与 prediction 的共同
  物理时间轴和同一无 padding 画布上比较六项 dynamics：vertical trajectory、period、
  amplitude envelope、equilibrium/release phase、vertical-axis confinement 和
  oscillation evidence。period 同时保留 prediction-vs-reference 的经验相似度，并仅惩罚
  prediction 相对 reference 新增的理想弹簧周期偏差；更接近理论不能覆盖经验周期不符。
  vertical-axis confinement 同时检查新增横向跨度，以及首帧归零后按共同时间样本比较的
  横向轨迹 RMSE；后者按 reference 垂直振幅和 `horizontal_trajectory_scale` 归一，因此
  相同横向跨度不能掩盖不同的横向运动历史。
  若共同 overlap timeline 只追加了一个短于网格步长的终点，spring dynamics 使用剔除该
  唯一末点的严格均匀 prefix；subject、topology 与 CSTI 仍使用完整共同 timeline，二者的
  sample counts 和被排除 index 会写入 audit/provenance。内部或多处 cadence 异常继续
  fail-closed。
  subject appearance/shape/position 与当前帧 spring topology 作为另外两个内容组件；
  topology 同时是乘法 integrity factor，因此移除或断开的弹簧不能靠钢球运动补偿。
  reference topology 还必须达到协议的 `minimum_reference_score`；reference 无弹簧支持是
  `unavailable`，而 prediction 无支持仍是已评估零分，不能倒置缺陷归属。

## CSTI

CSTI 使用 `exact_full_tube_edt`。它的精确输入是：冻结 entity manifest 中的 GT
`entity_id`、`role_id` 和 `same_case_gt` reference capability；scene evaluator 已验证并对齐
到同一物理时间轴、同一原生分析画布的 reference/prediction mask tubes；以及只有通过冻结
身份门后才写入的 matched prediction track IDs。当前公开 v1 只在完整 tube 评分前排除
第 0 个样本（条件首帧）；其余共同时间样本均参与评分。prediction unavailable 样本保持
空 mask，不会从 reference 补帧。

空间容差不再按画布尺寸取固定比例，而是逐 GT entity 只从 reference Tube 自适应计算：
对排除条件首帧后的共同规则时间轴上每个非空 reference mask 的面积 `A_t` 计算面积等效直径
`d_t = 2 * sqrt(A_t / pi)`，取各帧中位数 `d_ref`，再以
`r = 0.5 * d_ref` 作为 x/y 同尺度的 soft-support 半径。EDT 的三维采样尺度因此为
`(delta_t / 0.025 s, 1 / r, 1 / r)`。该半径不读取 prediction mask 的尺寸，避免模型通过
放大预测主体来放宽自身容差；面积等效直径也不会被 mask 的孤立远端像素像 bounding box
那样显著放大。每个对象的 `d_ref`、有效 `r` 与非空 reference 帧数都会写入 CSTI audit。
运行时 API 仍可读取显式提供的旧 canvas-fraction 配置，但它不属于当前公开 v1 schema。
既有结果继续绑定其原 protocol fingerprint，不会被当前配置静默重解释；要获得新分数必须
以当前 fingerprint 重新执行或导入预测。

case 层按全部 GT entities 求均值，Task 层作为独立维度汇总；CSTI 不替代 scene expert
score。

## Failure semantics

协议的预测侧策略是 fail-closed but finite：缺失记录、无效媒体或预测 observation 失败
产生零分 case，并计入 `degraded_evaluated_jobs` 与 reason counts。这样提交缺失不会通过
降低分母获益。

Reference 缺失或不可认证时结果为 `unavailable`；内部 evaluator 异常为 `error`。这些
状态不会转换成成功分数。所有 case 结果都必须对应 canonical job，重复、越界或身份不符
的记录会被拒绝。

Spring evaluator 还保持明确的失败来源：reference mask/identity/trace 缺陷是
`unavailable`，prediction identity/observation 缺陷按 robust contract 为已评估零分，
依赖、模型、配置或实现内部故障不能伪装成任一视频的评分结果。

### Reference evaluability preflight

正式评测前必须运行 `make evaluation-preflight`。该命令按官方 Task 物化完整 job 集，
将每个 Case 的 canonical reference video 作为已知合法 prediction 交给同一公开 evaluator，
并要求每个 job 均返回 `evaluated`、有限的主分数，以及有限或明确
`not_applicable` 的 CSTI。任一 `unavailable`、`error`、重复/缺失结果或非有限分数都会让
命令非零退出，并记录在 `reference_preflight.json`。

该预检是 Dataset/evaluator 的发布门，不是 Baseline 分数。它用于保证 reference 侧缺陷在
昂贵推理开始前暴露；绝不能通过把 reference 异常改记为 prediction 零分来通过预检。

## Artifacts and audit

成功的 scene evaluator 可以写入逐帧 CSV、诊断曲线和 JSON audit；artifact 写入失败只进入
`quality.artifact_failures` 与 `provenance.artifact_failures`，不能改变分数或状态。Spring
结果具体发布 vertical-spring per-frame CSV、trajectory curve、subject-components CSV、
subject-similarity curve 和 audit JSON。其 provenance 记录媒体摘要、共同 timeline、双方
无 padding spatial transform、冻结 anchor/NPZ 摘要、SAM2 prompt/传播、identity 决策、
topology、CSTI 绑定和是否复用完全相同的 sampled frames。
冻结 anchor 的 manifest/NPZ 路径按 `dataset_relative_asset_reference_v1` 发布；本机解析出的
绝对路径只供 evaluator 内部读取，不进入 result 或 audit provenance。

## Aggregation

Task 汇总按 canonical `scene_ids` 计算六场景宏平均，同时报告逐场景、partition、状态、
coverage、observed mean、CSTI 和 robustness 诊断。报告策略来自 Task v1；Evaluator 不得
扩大或缩小评估集合。
