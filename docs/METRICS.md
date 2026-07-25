# Metrics & Evaluation

## CommonSenseMetrics

不依赖 GT，判断物体/支架是否稳定、运动是否连续、是否出现违背任务常识的拓扑变化。当前由 `manual_scores.common_sense` 或未来 VLM judge 插件提供 `[0,1]` 分数；没有 judge 时状态为 `unavailable`。

## PredictionMetrics

要求 `assets.physics_reference_video`。目标流程为：时间对齐 → scene 专属主体分割 → 非物理区域 mask → 轨迹/形状重合与物理量误差。当前接口会检查 GT、预测视频和 segmenter/evaluator 是否齐备；算法插件缺失时不伪造分数。

v2 AtomicRun 已实现五场景 scene-aware Task evaluator。正式 case 分数来自 scene-local
物理状态，而不是直接把像素 IoU 当作物理正确性：

- 单摆：摆角轨迹、周期、振幅、支点与摆长稳定性；
- 自由落体：竖直轨迹、归一化加速度、触地时间与横向漂移；
- 斜面下滑：斜面局部轨迹、加速度、下降时间、接触与姿态稳定性；
- 匀速圆周：相对角轨迹、角速度、圆轨道与多物体半径配置；
- 一维碰撞：三实例轨迹、接触时刻、前后速度、动量与参考轨迹估计的恢复系数。

所有成功 case 都输出逐帧 CSV 和 Jensen 风格物理主体 IoU 曲线；斜面与圆周另有
几何归一化 IoU，一维碰撞另有 matched instance IoU。详见
[Scene-aware Evaluation](EVALUATION_ARCHITECTURE.md)。

## VisualJudgmentMetrics

仅 `has_real_reference_video=true` 时适用。计划包含 SSIM/LPIPS 或 VLM 感知评分，以及材质渲染一致性。合成首帧但没有同外观实拍延续的 OOD1 case 自动为 `not_applicable`。

## Scene 专属配置

`configs/scenes/*.json` 定义常识检查、主体、关键物理量和材质属性。
`configs/evaluation/protocols/*.json` 声明 scene evaluator、时间窗、观测质量阈值和
评分权重。`scene_default_v1` 已覆盖五个正式 scene；未知或显式关闭的 scene 仍返回
`unsupported`，不会静默记作零分。

## 聚合

默认权重：CommonSense 0.25、Prediction 0.45、VisualJudgment 0.30。只在适用指标间归一化；默认要求所有适用维度均已评估，否则 final score 为 null，并始终报告 coverage。

上述三维权重仍用于 v1 runner。v2 AtomicRun 的 Task evaluator 使用严格 coverage：
`finetune_eval` 先分别聚合 ID/OOD1，再做 scene 宏平均；`direct_eval` 在 scene 内聚合
全部 case；Task 最后对 scene 做宏平均。任一正式 case 未评估时 Task score 为 null，
同时提供明确标记为部分结果的 `observed_mean_score`。
