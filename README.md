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
- 三档自注册 Baseline：submission、managed（默认）与高级 command。
- 标准 I2V/T2V 由公共 compiler 接管；新模型通常只需 manifest 和薄 driver。
- 便携实现指纹与机器部署指纹分离，代码、profile、checkpoint 均可追踪。
- 五个 scene-local evaluator，以物理状态相似度作为正式分数。
- Jensen 风格物理主体 IoU 曲线，以及场景专属几何或实例诊断。
- Task 级严格 coverage：缺失 case 不会被静默计零，也不会被部分均值掩盖。
- 可审计的 Dataset、TaskInstance、prediction、evaluator 和 run 指纹。
- AtomicRun 自包含：除模型代码、权重和可重建 cache 外，预测、日志、输入快照与
  评测产物必须保存在 `runs_v2/<run_id>/` 内。

当前自动发现的 Baseline：

| baseline ID | 模型身份 | 支持任务 | 可比性 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3` | WAN2.2 + View A LoRA | `finetune_eval`, `direct_eval` | 按正式任务执行 |
| `cosmos3_nano_i2v` | Cosmos3-Nano base snapshot | `direct_eval` | base pretrained |
| `wan22_g15_sparse_motion_r32_e20` | G15 step-2840 frozen LoRA | `direct_eval` | 仅诊断；训练源与 v3 重叠 |

G15 的 source-aware audit 记录了 176/214 个见过的源 case，因此全量分数不能与无泄漏
Baseline 横向排名。详见 Bundle 内的
`baselines/wan22_g15_sparse_motion/provenance/benchmark_overlap_v3.json`。

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

列出并验证自动发现的 Baseline：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate cosmos3_nano_i2v

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20
```

只编译 sealed TaskInstance：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_physics.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output /tmp/wan22_physics_task_instance.json
```

创建 AtomicRun；不加 `--execute` 只冻结计划和输入：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_generic.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output-root runs_v2
```

创建新 Baseline：

```bash
# 标准 I2V：生成 schema v4 managed Bundle
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_i2v --backend managed-i2v

# 已有视频：生成 output-only submission Bundle
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_outputs --backend submission
```

managed 与 submission 复用 Benchmark 的 canonical plan、五阶段 DataAdapter、TaskInstance
seal、prediction 组装和 run-local 归档；复杂训练仍可使用 v3 command 接口。

对已有 AtomicRun 重新评估：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/<run_id>
```

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [数据集与划分](docs/DATASET.md)
- [自定义 Baseline 集成指南](docs/BASELINE_INTEGRATION.md)
- [TaskBuilder 与 Baseline 接入](docs/TASKS.md)
- [DataAdapter 与条件隔离](docs/DATA_ADAPTER.md)
- [五场景评估协议](docs/EVALUATION.md)
- [WAN2.2、View A LoRA 与 G15 baseline](docs/WAN22.md)
- [Cosmos3-Nano I2V baseline](docs/COSMOS3.md)
- [运行、验证与故障排查](docs/OPERATIONS.md)

## 仓库结构

```text
physics_video_benchmark/
├── datasets/                 # 唯一权威数据根
├── tasks/official/           # 四类五场景原子任务
├── baselines/                # 三个自注册 Bundle、driver/endpoint 与本机模板
├── configs/evaluation/       # Scene evaluator 协议
├── schemas/v2/               # Dataset、Task、TaskInstance JSON Schema
├── schemas/v3/               # 高级 command Bundle Schema
├── schemas/v4/               # managed/submission Bundle Schema
├── src/physbench/            # 数据、任务、managed runtime、评估和 CLI
├── tests/                    # 核心与 scene evaluator 回归测试
├── docs/                     # 当前架构与操作文档
└── runs_v2/                  # AtomicRun 输出
```

权威数据资产只能写入 `datasets/`。Baseline 重采样、缩放、抽帧、特征和模型缓存必须
写入内容寻址 cache 或 run 目录，不能回写 Dataset。
