# Benchmark protocol

## Dataset identity

- Dataset ID：`physics_video_seven_scene_v13`
- Release：`13.0.0`
- Cases：916
- Views：`view_a`、`view_b`
- Hub binding：`datasets/huggingface.json`

每个 Case 拥有规范 prompt、结构化物理量、输入资产、参考资产和 split 注释。Baseline
只能读取其 input policy 授权的投影；参考媒体与 evaluator 注释不会进入推理输入。

## Six scored scenes

1. `pendulum`
2. `collision_1d`
3. `inclined_plane_slide`
4. `uniform_circular_motion`
5. `parabolic_motion`
6. `vertical_spring_oscillator`

## One Dataset-only unsupported scene

1. `push_bottle`

`vertical_spring_oscillator` 已由公开的 `vertical_spring_oscillator_v1` evaluator 正式
评分，并同时进入两个官方 Task 的评估集合。`push_bottle` 数据完整保留，但不被任何官方
Task v1 的训练或评估集合选择，也没有公开 scene evaluator。

## Official Tasks

- `six_scene_direct_eval_v1`：六场景 775 个直接评估 job，无训练；
- `six_scene_train_six_scene_eval_v1`：六场景 679 个训练 case，六场景 96 个 ID
  评估 job。

两者都使用 `scene_default_v1`。Task 拥有数据选择、种子和报告策略；Baseline 拥有输入
适配、是否使用物理信息、模型训练和推理实现。

`scene_default_v1` 的 registry 公开解析上述六个正式评分场景。协议可解析范围与 Task
选择范围仍彼此独立；只有发布新的已封印 Task 文件才会改变 scene IDs 或 canonical jobs。

## Media contract

Managed I2V job 会封印允许使用的媒体通道、输出画布、FPS、物理时间零点、帧数规则和
run 内输出路径。V2V 接口是预留扩展点，当前官方 Dataset Case 尚未提供独立裁剪的条件
前缀视频，因此官方 benchmark 不支持 managed V2V。reference/source/physics-reference
video 都不得替代该输入资产。无法解码、违反媒体契约或越权读取 evaluator/source 视频
的预测会在场景评分前失败。

## Scoring, degradation and coverage

官方 Task v1 对六场景运行对象级物理评分，并计算独立 CSTI 时空一致性维度。以下由
robustness policy 覆盖的预测侧失败按“已评估零分”处理，同时写入 degradation reason：

- 缺失或未完成的 prediction record；
- evaluator 侧的 prediction 解码失败；
- prediction observation 失败。

输出视频违反 **sealed media/record contract**（例如画布、FPS 或帧数不符），或 runtime
明确产生 `protocol_error` prediction record 时，Task 结果仍记录 `protocol_error`；它不会
被改写为已评估零分，也不计入完整 coverage。prediction 清单的身份/覆盖不匹配、重复或
额外记录、以及嵌入 media contract 不匹配则由 **pre-evaluation validation** 直接拒绝；
这种运行不会产生 publishable Task result。

参考资产失败仍标记为 unavailable，evaluator 内部错误仍标记为 error；这两类不会伪装成
有效零分。Task 汇总以 canonical jobs 为分母，记录 coverage、status counts、逐场景结果、
expert 分数、CSTI 和退化诊断。只有 canonical identity、计划、预测与协议摘要一致的产物
才能作为正式结果。

评分实现细节见 [EVALUATION.md](EVALUATION.md)。
