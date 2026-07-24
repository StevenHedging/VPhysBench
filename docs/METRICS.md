# Metrics & Evaluation

## CommonSenseMetrics

不依赖 GT，判断物体/支架是否稳定、运动是否连续、是否出现违背任务常识的拓扑变化。当前由 `manual_scores.common_sense` 或未来 VLM judge 插件提供 `[0,1]` 分数；没有 judge 时状态为 `unavailable`。

## PredictionMetrics

要求 `assets.physics_reference_video`。目标流程为：时间对齐 → scene 专属主体分割 → 非物理区域 mask → 轨迹/形状重合与物理量误差。当前接口会检查 GT、预测视频和 segmenter/evaluator 是否齐备；算法插件缺失时不伪造分数。

## VisualJudgmentMetrics

仅 `has_real_reference_video=true` 时适用。计划包含 SSIM/LPIPS 或 VLM 感知评分，以及材质渲染一致性。合成首帧但没有同外观实拍延续的 OOD1 case 自动为 `not_applicable`。

## Scene 专属配置

`configs/scenes/*.json` 定义常识检查、主体、关键物理量和材质属性。后续每实现一个算法，按 scene ID 注册 evaluator 即可。

## 聚合

默认权重：CommonSense 0.25、Prediction 0.45、VisualJudgment 0.30。只在适用指标间归一化；默认要求所有适用维度均已评估，否则 final score 为 null，并始终报告 coverage。

