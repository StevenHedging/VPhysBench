# Benchmark protocol

## Dataset identity

- Dataset ID：`physics_video_seven_scene_v13`
- Release：`13.0.0`
- Cases：916
- Views：`view_a`、`view_b`
- Hub binding：`datasets/huggingface.json`

每个 Case 拥有规范 prompt、结构化物理量、输入资产、参考资产和 split 注释。Baseline
只能读取其 input policy 授权的投影；参考媒体与 evaluator 注释不会进入推理输入。

## Five scored scenes

1. `pendulum`
2. `collision_1d`
3. `inclined_plane_slide`
4. `uniform_circular_motion`
5. `parabolic_motion`

## Two preview scenes

1. `push_bottle`
2. `vertical_spring_oscillator`

`vertical_spring_oscillator` 在 finetune Task v1 中参与训练，但没有公开 evaluator，
不进入正式评估 job、五场景聚合或 leaderboard 分数。`push_bottle` 的数据完整保留，
但不被任何官方 Task v1 的训练或评估集合选择。

## Official Tasks

- `five_scene_direct_eval_v1`：五场景 658 个直接评估 job，无训练；
- `six_scene_train_five_scene_eval_v1`：六场景 679 个训练 case，五场景 76 个 ID
  评估 job。

两者都使用 `scene_default_v1`。Task 拥有数据选择、种子和报告策略；Baseline 拥有输入
适配、是否使用物理信息、模型训练和推理实现。

## Media contract

Managed I2V/V2V job 会封印允许使用的媒体通道、输出画布、FPS、物理时间零点、帧数规则
和 run 内输出路径。无法解码、违反媒体契约或越权读取 evaluator/source 视频的预测会在
场景评分前失败。

## Scoring, degradation and coverage

Evaluation v1 对五场景运行对象级物理评分，并计算独立 CSTI 时空一致性维度。以下预测侧
失败按“已评估零分”处理，同时写入 degradation reason：

- 缺失或未完成的 prediction record；
- prediction media 无效；
- prediction observation 失败。

参考资产失败仍标记为 unavailable，evaluator 内部错误仍标记为 error；这两类不会伪装成
有效零分。Task 汇总以 canonical jobs 为分母，记录 coverage、status counts、逐场景结果、
expert 分数、CSTI 和退化诊断。只有 canonical identity、计划、预测与协议摘要一致的产物
才能作为正式结果。

评分实现细节见 [EVALUATION.md](EVALUATION.md)。
