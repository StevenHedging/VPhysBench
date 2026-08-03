# Physics Video Benchmark

Physics Video Benchmark 是一个面向物理视频生成模型的六场景、训推一体评测框架。当前
Dataset release 是
`datasets/physics_video/releases/6.0.0/dataset.json`，包含604个case和1,650个
锁定资产：

- 单摆 `pendulum`
- 自由落体 `free_fall`
- 一维对心碰撞 `collision_1d`
- 斜面下滑 `inclined_plane_slide`
- 匀速圆周运动 `uniform_circular_motion`
- 平抛运动 `parabolic_motion`

当前六场景Task为`six_scene_finetune_eval.json`和`six_scene_direct_eval.json`。现有
`five_scene_*` Task与4.0.0元数据只用于解释明确引用旧Dataset ID/digest的历史结果，
不会被当前运行入口自动选择。

## 设计原则

数据、任务和模型输入策略各有唯一所有者：

| 对象 | 负责 |
| --- | --- |
| Dataset / Case | 原始 `text.prompt`、首帧、结构化物理与环境标注、参考资产、View |
| Task | `family`、数据选择、seed、诊断报告策略、评估协议 |
| Baseline | 模型身份、I2V/V2V 等输入范式、是否使用物理信息、物理表示与 adapter |
| Evaluator | 参考解析、时空对齐、scene-local 物理评分与 Task 汇总 |

因此 Task 不再区分“带/不带物理注入”。所有 Baseline 都接收同一种可条件化 Case，
再由固定的 `input_policy.physics.usage` 决定 `ignored`、`optional` 或 `required`。
例如 WAN generic 与 WAN physics 是两个 Baseline identity；前者原样使用
`case.text.prompt`，后者在 Baseline-owned adapter 中追加经审计的结构化物理量。

```text
DatasetSnapshot + model-agnostic TaskSpec
                    │
                    ▼
            CanonicalTaskPlan
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
generic Baseline          physics Baseline
       │                         │
       └── sealed BaselineTaskInstance
                          │
                          ▼
                     AtomicRun
                          │
                          ▼
                 scene-local evaluation
```

同一 Task 对多个 Baseline 的 canonical plan 必须完全相同；差异只能来自各 Baseline
的模型、adapter、训练或推理实现。

## 当前 Baseline

Registry 会发现每个 Bundle 目录下的 `baseline.json` 与 `*.baseline.json`：

| Baseline ID | 物理策略 | 支持 Task family | 说明 |
| --- | --- | --- | --- |
| `wan22_ti2v_5b_lora_r32_v3_generic` | `ignored` | `finetune_eval`, `direct_eval` | WAN2.2 + LoRA |
| `wan22_ti2v_5b_lora_r32_v3_physics` | `required` / `structured_text` | `finetune_eval`, `direct_eval` | 同模型，追加结构化物理文本 |
| `wan22_ti2v_5b_lora_r32_quantity_embedding_v1` | `required` / `quantity_token_embedding_v1` | `finetune_eval` | WAN2.2 + LoRA，SI 数值/量纲编码 |
| `cosmos3_nano_i2v_generic` | `ignored` | `direct_eval` | Cosmos3-Nano base |
| `cosmos3_nano_i2v_physics` | `required` / `structured_text` | `direct_eval` | 同模型，追加结构化物理文本 |
| `wan22_g15_sparse_motion_r32_e20_generic` | `ignored` | `direct_eval` | G15 step-2840，诊断型 |
| `wan22_g15_sparse_motion_r32_e20_physics` | `required` / `structured_text` | `direct_eval` | G15 step-2840，诊断型 |

G15 的训练源与 Dataset 底层 trial 有重叠，不能进入无泄漏排名。审计见
`baselines/wan22_g15_sparse_motion/provenance/benchmark_overlap_v3.json`；4.0.0 没有
改变 3.0.0 的 case 或媒体集合，所以该 source-aware 结论仍成立。

## 环境与验证

Benchmark 环境：

```text
/root/miniconda3/envs/phybench
```

```bash
cd /root/Steven/physics_video_benchmark

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  -m unittest tests.test_six_scene_dataset_v6 -v

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --check-asset-hashes
```

这是当前release的正式数据/Task回归入口。`make legacy-test`会额外运行历史Dataset测试；
其中部分测试需要已经退出当前资产布局的旧路径，不属于6.0.0发布门槛。

Scene evaluator 需要额外安装：

```bash
/root/miniconda3/envs/phybench/bin/pip install -e ".[scene-evaluation]"
/root/miniconda3/envs/phybench/bin/pip install -e /root/Jensen/Eval/sam2-main
```

## 快速开始

下面三条命令是现有五场景Baseline在当前6.0.0 Dataset上的兼容性smoke，使用
`tasks/smoke/five_scene_direct_eval_v6.json`排除尚未支持的平抛scene。要运行完整六场景
Task，Baseline必须先声明并实现`parabolic_motion`支持。

发现并验证 Baseline：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3_generic
```

编译一份 sealed TaskInstance：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --task tasks/smoke/five_scene_direct_eval_v6.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --output /tmp/wan22_generic_task_instance.json
```

创建 AtomicRun；不加 `--execute` 时只冻结并展开计划：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --task tasks/smoke/five_scene_direct_eval_v6.json \
  --baseline cosmos3_nano_i2v_generic \
  --output-root runs_v2
```

在同一 Task 上成对比较两个 Baseline：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --task tasks/smoke/five_scene_direct_eval_v6.json \
  --baseline cosmos3_nano_i2v_generic \
  --baseline cosmos3_nano_i2v_physics \
  --matrix-id cosmos3_generic_vs_physics \
  --output-root runs_v2
```

加 `--execute` 才会启动模型。每个矩阵元素仍是独立
`runs_v2/<matrix_id>__<baseline_id>/` AtomicRun。

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [Dataset、Case 与划分](docs/DATASET.md)
- [原始视频与XLSX导入规范](docs/DATASET_INGESTION.md)
- [新实验情景Evaluator接入指南](docs/NEW_SCENE_EVALUATOR.md)
- [Task 与运行矩阵](docs/TASKS.md)
- [DataAdapter 与输入策略](docs/DATA_ADAPTER.md)
- [自定义 Baseline 集成](docs/BASELINE_INTEGRATION.md)
- [场景评估协议](docs/EVALUATION.md)
- [WAN2.2 Baseline](docs/WAN22.md)
- [WAN2.2 物理量编码 Baseline](docs/WAN22_QUANTITY_EMBEDDING.md)
- [WAN2.2 物理量编码五场景实验报告](docs/experiments/WAN22_QUANTITY_EMBEDDING_20260728.md)
- [碰撞评估器 v4 与可观测性审计](docs/experiments/COLLISION_EVALUATOR_V4_20260730.md)
- [Cosmos3-Nano Baseline](docs/COSMOS3.md)
- [运行、验证与故障排查](docs/OPERATIONS.md)

## 仓库结构

```text
physics_video_benchmark/
├── datasets/                 # 唯一权威数据根；6.0.0是当前release
├── tasks/official/           # direct_eval 与 finetune_eval 两份模型无关 Task
├── baselines/                # schema v5 Bundle、adapter/driver 与本机配置模板
├── configs/evaluation/       # scene evaluator 协议
├── schemas/v3/               # 历史Dataset、Case、Task及当前TaskInstance
├── schemas/v4/               # 当前Dataset、Case和Task
├── schemas/v5/               # Baseline Bundle
├── src/physbench/            # planner、runtime、评估与 CLI
├── tests/                    # 回归测试
├── docs/                     # 架构和操作文档
└── runs_v2/                  # 当前 AtomicRun 输出
```

权威数据资产只能写入 `datasets/`。缩放、抽帧、特征、模型缓存和预测必须进入内容寻址
cache 或当前 run，不能回写 Dataset。
