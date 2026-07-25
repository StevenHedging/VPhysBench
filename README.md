# Physics Video Benchmark

> 新实验建议使用 v2 的 `Dataset × AtomicTask × Baseline` 架构。设计和命令见
> [Benchmark v2 架构](docs/ARCHITECTURE_V2.md)；Baseline 的任务编译边界见
> [TaskBuilder 架构](docs/TASK_BUILDER_ARCHITECTURE.md)；模型输入适配契约见
> [统一 Data Adapter 架构](docs/DATA_ADAPTER_ARCHITECTURE.md)。所有权威数据统一位于
> `datasets/`；原有 `configs/`、v1 release 和 `runs/` 作为兼容层与历史记录保留。

一个面向物理视频生成模型的“训推一体 / 调推一体”Benchmark 骨架。它把数据、
划分、任务、模型输入适配和评测解耦，支持：

- **视图 A**：多 scene 训练或微调后，评测 `test_id` 和 `test_ood1`。
- **视图 B**：不训练，按 scene 内确定性随机、尽量等量的 `group_1..n` 直接评测。
- **多输入 baseline**：T2V、I2V/TI2V 以及外部命令式模型均通过统一适配层接入。
- **Baseline-owned TaskBuilder**：每个 Baseline 将不可变的 `DatasetSnapshot + TaskSpec`
  编译为本模型可执行、带 SHA-256 封印的 `BaselineTaskInstance`；训练器和预测器只消费该实例。
- **三类评测**：CommonSense、Prediction、VisualJudgment；当前算法接口和适用性门控已实现，模型相关实现为占位插件。
- **可复现运行**：每次运行冻结 task、baseline、数据指纹、job plan、预测清单和报告。
- **条件受控消融**：case/视频与模型输入解耦；当前 v2 Task 提供 `generic` 和
  `physics`，由 Baseline 的 TaskBuilder 调用其内部 DataAdapter 生成输入；第一类任务
  分别训练独立 Adapter，并共享数据划分、首帧、seed 与生成超参。

OOD2 暂不开放。配置若请求 OOD2，校验器会明确报错，避免把不合理的跨任务测试混入结果。

## 快速自测

项目只使用 Python 标准库，不需要安装依赖：

```bash
cd /root/Steven/physics_video_benchmark
make test
make smoke
```

当前五场景 release（仍使用 v2 数据契约）常用命令：

```bash
# 只编译，不训练或推理；输出可审计的 sealed task instance
PYTHONPATH=src python3 -m physbench task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/wan22_physics_task_instance.json

# atomic-run 总是先执行同一个 TaskBuilder 编译步骤；不加 --execute 只做计划/暂存
PYTHONPATH=src python3 -m physbench atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

v1 兼容命令：

```bash
PYTHONPATH=src python3 -m physbench validate --manifest examples/fixtures/cases.jsonl
PYTHONPATH=src python3 -m physbench split --view A --manifest examples/fixtures/cases.jsonl --output /tmp/view_a.json
PYTHONPATH=src python3 -m physbench split --view B --manifest examples/fixtures/cases.jsonl --groups 2 --seed 42 --output /tmp/view_b.json
PYTHONPATH=src python3 -m physbench run --task examples/fixtures/task_view_a.json --baseline configs/baselines/dummy_i2v.json --manifest examples/fixtures/cases.jsonl --split examples/fixtures/view_a.json --output-root runs
```

真实数据到位后，只需按 [数据契约](docs/DATA_CONTRACT.md) 写 `cases.jsonl`，无需修改核心代码。
提示词模板、最终解析文本及其 SHA-256 会随每个 run 冻结。
两类提示词的训练/推理展开规则见 [提示词条件设计](docs/PROMPT_CONDITIONING.md)。

已有 WAN2.2-TI2V-5B + LoRA 已作为专用 baseline 接入，包含视图 A 联合微调、视图 B 冻结 LoRA、首帧映射和不同视频规格的模型侧只读适配。见 [WAN2.2 + LoRA 文档](docs/WAN22_LORA_BASELINE.md)。

真实单摆、碰撞、自由落体、斜面下滑和匀速圆周运动数据的命名解释、时间尺度、
官方划分与审计方式见 [真实数据导入记录](docs/REAL_DATA_IMPORT.md)。

## 项目结构

```text
physics_video_benchmark/
├── datasets/             # 唯一数据根：权威资产、来源审计与版本化 release
├── tasks/                # v2：四类原子 Task 与 OOD2 recipe
├── baselines/            # v2：静态 Baseline bundle、内置 TaskBuilder 与模型侧适配配置
├── configs/              # scene、task、baseline、metric 配置
├── docs/                 # 设计、数据、评测和接入文档
├── examples/fixtures/    # 不依赖真实视频的端到端测试数据
├── schemas/              # JSON Schema 数据契约
├── src/physbench/        # CLI、划分、任务、适配、评测、报告
├── tests/                # 标准库 unittest
├── runs/                 # v1 历史运行
└── runs_v2/              # v2 AtomicRun 产物
```
