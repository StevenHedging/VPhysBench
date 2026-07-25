# Physics Video Benchmark

Physics Video Benchmark 是一个面向物理视频生成模型的五场景、训推一体评测框架。
当前唯一正式数据快照是
`datasets/physics_video/releases/3.0.0/dataset.json`，包含：

- 单摆 `pendulum`
- 自由落体 `free_fall`
- 一维对心碰撞 `collision_1d`
- 斜面下滑 `inclined_plane_slide`
- 匀速圆周运动 `uniform_circular_motion`

系统把数据、任务、模型适配、生成和评估分成明确边界：

```text
DatasetSnapshot
  + Atomic TaskSpec
  + Baseline bundle
        │
        ▼
Baseline-owned TaskBuilder
        │
        ▼
sealed BaselineTaskInstance
        │
        ├── optional fine-tuning
        ├── generation
        └── benchmark-owned scene evaluation
```

## 核心能力

- View A：训练或微调后评测数值 ID 与环境 OOD1。
- View B：不训练，按确定性分组直接评测全部 case。
- `finetune_eval/direct_eval × generic/physics` 四种原子任务。
- Baseline 私有 DataAdapter，支持模型原生文本、首帧、时空规格和物理信息注入。
- 五个 scene-local evaluator，以物理状态相似度作为正式分数。
- Jensen 风格物理主体 IoU 曲线，以及场景专属几何或实例诊断。
- Task 级严格 coverage：缺失 case 不会被静默计零，也不会被部分均值掩盖。
- 可审计的 Dataset、TaskInstance、prediction、evaluator 和 run 指纹。

## 环境

Benchmark 的正式虚拟环境是：

```text
/root/miniconda3/envs/phybench
```

核心规划与数据测试可用普通 Python。视频 evaluator 需要：

```bash
cd /root/Steven/physics_video_benchmark
/root/miniconda3/envs/phybench/bin/pip install -e ".[scene-evaluation]"
/root/miniconda3/envs/phybench/bin/pip install -e /root/Jensen/Eval/sam2-main
```

SAM2 当前来自 `/root/Jensen/Eval/sam2-main`。未编译 `_C` 后处理扩展时会出现警告，
官方 fallback 仍可运行，但部署环境应优先完成官方扩展安装。

## 快速验证

```bash
# 核心测试；缺少 scene-evaluation extras 时相关测试会明确 skip
make test

# 正式环境中的完整测试
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v

# 数据资产逐字节验收
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-asset-hashes
```

## 编译和运行任务

只编译 sealed TaskInstance：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/wan22_physics_task_instance.json
```

创建 AtomicRun；不加 `--execute` 只冻结计划和输入：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

对已有 AtomicRun 重新评估：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/<run_id>
```

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [数据集与划分](docs/DATASET.md)
- [TaskBuilder 与 Baseline 接入](docs/TASKS.md)
- [DataAdapter 与条件隔离](docs/DATA_ADAPTER.md)
- [五场景评估协议](docs/EVALUATION.md)
- [WAN2.2 + LoRA baseline](docs/WAN22.md)
- [运行、验证与故障排查](docs/OPERATIONS.md)

## 仓库结构

```text
physics_video_benchmark/
├── datasets/                 # 唯一权威数据根
├── tasks/official/           # 四类五场景原子任务
├── baselines/wan22_lora/     # Baseline bundle、TaskBuilder 配置和 profiles
├── configs/evaluation/       # Scene evaluator 协议
├── schemas/v2/               # 当前公共 JSON Schema
├── src/physbench/            # 数据、任务、编排、评估和 CLI
├── tests/                    # 核心与 scene evaluator 回归测试
├── docs/                     # 当前架构与操作文档
└── runs_v2/                  # AtomicRun 输出
```

权威数据资产只能写入 `datasets/`。Baseline 重采样、缩放、抽帧、特征和模型缓存必须
写入内容寻址 cache 或 run 目录，不能回写 Dataset。
