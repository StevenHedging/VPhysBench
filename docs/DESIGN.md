# Benchmark 设计规划

## 1. 术语与边界

- **scene**：一大类物理实验，如单摆、自由落体。
- **case**：一条原子数据；case ID 全局唯一。
- **ID**：物理参数变化仍位于该 scene 定义的任务域内。
- **OOD1**：物理任务不变、外观或成像因素变化。合成样本只合成首帧，不伪造真实参考视频。
- **OOD2**：跨物理任务域泛化。当前没有可信 case 和算法约定，框架明确拒绝执行。

## 2. 两种数据视图

### 视图 A：训练/微调与泛化评测

每个 scene 的 case 被显式标注为：

```text
scene
├── train
├── test_id
└── test_ood1
    ├── background
    ├── material
    ├── support / shape / viewpoint ...
    └── provenance.parent_case_id -> 对应 ID case
```

OOD1 case 必须记录变化因子和父 case。合成 OOD1 必须复用父 case 的物理参数；校验器检查这一约束。父 case 的真实视频可以作为
`physics_reference_video` 供运动预测比较，但不能冒充同外观实拍，因此
`has_real_reference_video=false`，VisualJudgment 自动不适用。

### 视图 B：零训练直接评测

每个 scene 独立打乱，再以轮转方式放入 `group_1..n`。打乱种子经 SHA-256 与 scene ID 组合，避免 Python 进程哈希导致不可复现；各组大小差不超过 1。
group 只负责均衡组织和报告，不暗含 train/test 语义。

## 3. 两类 Task

1. `finetune_and_eval`：使用视图 A，可选择多个 scene；训练 `train`，同时评测 `test_id`、`test_ood1`。baseline 可声明 train 或 finetune 能力。
2. `zero_shot_eval`：使用视图 B，不允许训练；可按 scene、group 或显式 case ID 筛选。

Task planner 只生成不可歧义的训练列表与推理 job，不把模型命令硬编码到 Benchmark。

## 4. Baseline 抽象

case 提供语义等价的输入视图：

- `t2v`：文本描述/提示词；
- `i2v`：首帧 + 文本；
- `ti2v`：首帧 + 文本 + 结构化物理参数；
- 后续模型可注册新的 view，不修改 case 的原始资产。

baseline 配置声明 `input_view`、训练能力和 adapter。内置 `dummy` 用于无模型自测；`command` adapter 为外部训练/推理程序生成 JSON job，可在明确允许时执行。

## 5. 指标与缺失处理

三个维度分别返回 `evaluated / unavailable / not_applicable / error`，绝不把“没有算法/GT”记作 0 分：

- CommonSense：不要求 GT；当前支持 VLM/人工分数注入接口。
- Prediction：要求 `physics_reference_video`，并预留 segmenter/mask 与 scene evaluator。
- VisualJudgment：仅在 `has_real_reference_video=true` 且具有真实参考视频时适用。

默认聚合策略 `require_all_applicable`：任一适用指标未评估时最终分为空；可选
`reweight_available` 仅用于探索性报告。报告同时给出 metric coverage，防止高分低覆盖率被误读。

## 6. 运行可复现性

每个 run 保存：

```text
runs/<run_id>/
├── run.json                 # 时间、版本、数据 SHA-256、状态
├── frozen_task.json
├── frozen_baseline.json
├── plan.json                # train cases + inference jobs
├── predictions.jsonl        # 模型产物契约
├── case_metrics.jsonl
├── summary.json
└── report.md
```

实际执行器可断点续跑；case/job ID 决定输出路径，避免不同任务互相覆盖。

## 7. 自评

该结构贴近当前意图，因为它同时满足：scene/case 语义、A/B 双视图、调推与零训练两模式、OOD1 合成首帧溯源、多输入 baseline、逐 scene 指标扩展和无 GT 样本的严格门控。真实数据、分割器、VLM 与基线权重缺失不会阻止结构验证，也不会产生伪分数。

已知后续决策点：OOD1 的官方合成协议、各 scene 的物理主体 mask 定义、时间对齐策略、三维权重和 OOD2 的科学任务构造。它们均被保留在配置/插件层，不需要重构数据格式。

