# Physics Video Benchmark

Physics Video Benchmark 是一个面向物理视频生成模型的六场景、训推一体评测框架。当前
Dataset release 是
`datasets/releases/11.0.0/dataset.json`，包含799个case和6,038个
锁定资产：

- 单摆 `pendulum`
- 一维对心碰撞 `collision_1d`
- 斜面下滑 `inclined_plane_slide`
- 匀速圆周运动 `uniform_circular_motion`
- 平抛运动 `parabolic_motion`
- 推水瓶 `push_bottle`

当前官方Task为`five_scene_finetune_eval.json`和`five_scene_direct_eval.json`，均指向
11.0.0 Dataset。推水瓶已进入Dataset，但专用评估器尚未完成，因此暂不进入这两份正式
计分Task。

11.0.0为每个有效Case新增
`datasets/assets/<scene>/<case>/physics.v11.json`，并通过
`assets.physics_annotation`绑定到Case和`assets.lock.json`。内联`case.physics`仍是
Baseline和Evaluator的兼容运行时API，Loader会强制校验两者完全一致。Release目录只
保留`README.md`、`dataset.json`、`release.json`、`cases.jsonl`、
`assets.lock.json`、`scenes/`和`views/`；迁移与验证证据位于
`datasets/provenance/releases/11.0.0/`。每个quantity包含稳定`symbol`；独立量的符号
必须出现在无数值prompt中。所有标量保存为非负大小，运动方向由prompt表达。10.0.0及其
`physics.json`保持不可变，以便复现历史结果。

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

PYTHONPATH=src:tests:. /root/miniconda3/envs/phybench/bin/python \
  -m unittest tests.test_six_scene_dataset_v8 -v

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/11.0.0/dataset.json \
  --check-asset-hashes
```

这是当前release的正式数据/Task回归入口。`make legacy-test`会额外运行历史Dataset测试；
其中部分测试需要已经退出当前资产布局的旧路径，不属于11.0.0发布门槛。

Scene evaluator 需要额外安装：

```bash
/root/miniconda3/envs/phybench/bin/pip install -e ".[scene-evaluation]"
/root/miniconda3/envs/phybench/bin/pip install -e /root/Jensen/Eval/sam2-main
```

## 快速开始

下面三条命令使用当前11.0.0 Dataset和已有评估器的五场景官方Task。推水瓶需等专用
评估器和协议接入后再加入正式Task。

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
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --output /tmp/wan22_generic_task_instance.json
```

创建 AtomicRun；不加 `--execute` 时只冻结并展开计划：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --output-root runs_v2
```

在同一 Task 上成对比较两个 Baseline：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
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
├── datasets/                 # 唯一权威数据根；11.0.0是当前release
├── tasks/official/           # direct_eval 与 finetune_eval 两份模型无关 Task
├── baselines/                # schema v5 Bundle、adapter/driver 与本机配置模板
├── configs/evaluation/       # scene evaluator 协议
├── schemas/v3/               # 历史Dataset、Case、Task及当前TaskInstance
├── schemas/v4/               # 当前Task与历史Dataset/Case
├── schemas/v5/               # 当前Dataset/Case与Baseline Bundle
├── src/physbench/            # planner、runtime、评估与 CLI
├── tests/                    # 回归测试
├── docs/                     # 架构和操作文档
└── runs_v2/                  # 当前 AtomicRun 输出
```

权威数据资产只能写入 `datasets/`。缩放、抽帧、特征、模型缓存和预测必须进入内容寻址
cache 或当前 run，不能回写 Dataset。
