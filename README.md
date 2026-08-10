# VPhysBench

Physics Video Benchmark 是一个面向物理视频生成模型的七场景、训推一体评测框架。当前
Dataset release 是
`datasets/releases/13.0.0/dataset.json`，包含916个case：

- 单摆 `pendulum`
- 一维对心碰撞 `collision_1d`
- 斜面下滑 `inclined_plane_slide`
- 匀速圆周运动 `uniform_circular_motion`
- 平抛运动 `parabolic_motion`
- 推水瓶 `push_bottle`
- 竖直弹簧振子 `vertical_spring_oscillator`

当前13.0.0 Dataset提供两组五场景官方Task：原有的
`five_scene_finetune_eval.json`/`five_scene_direct_eval.json`冻结
`scene_default_v10`专家评分；新增的同名`*_csti.json`变体冻结
`scene_default_v11`，在保留专家评分的同时独立报告CSTI轨迹评分。推水瓶和竖直弹簧
振子已进入Dataset，但专用评估器尚未完成，因此暂不进入这些正式计分Task。

| Task | 协议 | 输出维度 |
| --- | --- | --- |
| `five_scene_finetune_eval.json` / `five_scene_direct_eval.json` | `scene_default_v10` | 专家评分 |
| `five_scene_finetune_eval_csti.json` / `five_scene_direct_eval_csti.json` | `scene_default_v11` | 专家评分 + 独立CSTI |

v11结果中的顶层`score`仍是专家评分；CSTI位于
`task_result.json.dimensions.csti`，不会与专家分数混合成新的总分。

13.0.0在每个Case资产目录中只保存一份`caption.json`和一份`physics.json`，并由
`cases.jsonl`中的`assets.caption`与`assets.physics_annotation`引用。Loader读取这两个
Case-local成员后，向Baseline和Evaluator提供兼容的`case.text`与`case.physics`运行时
接口。Release目录只保留`dataset.json`、`cases.jsonl`、`scenes/`和`views/`；迁移与验证证据位于
`datasets/provenance/releases/13.0.0/`。每个quantity包含稳定`symbol`；独立量的符号
必须出现在无数值prompt中。标量使用`value/unit/symbol`，推水瓶外力使用带显式时间戳的
`samples/time_unit/unit/symbol`；所有正式数值保存为非负大小，运动方向由prompt表达。V1–V12
不再保留为活动运行目录；历史结果依靠Git历史和provenance追溯。

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
./.venv
```

```bash
cd VPhysBench

PYTHONPATH=src:tests:. python \
  -m unittest tests.test_current_dataset tests.test_single_current_physics_v13 -v

PYTHONPATH=src python -m physbench \
  validate-dataset \
  --dataset datasets/releases/13.0.0/dataset.json \
  --check-assets

PYTHONPATH=src python scripts/validate_dataset_v13.py
```

这是当前release的正式数据/Task回归入口。`make test`运行当前Dataset门禁，完整测试使用
`make full-test`；仓库不再提供依赖旧Release目录的运行门禁。

Scene evaluator 需要额外安装：

```bash
python -m pip install -e ".[scene-evaluation]"
python -m pip install -e ../sam2
```

## 快速开始

下面三条快速开始命令使用冻结的v10专家评分Task。需要独立CSTI维度时，把Task路径
替换为对应的`*_csti.json`文件；CSTI使用一次正式full-Tube 3D EDT和四个诊断prefix，
批量运行前应先阅读
[`CSTI_REFERENCE_PERFORMANCE_20260807.md`](docs/experiments/CSTI_REFERENCE_PERFORMANCE_20260807.md)
并规划CPU与内存。推水瓶和竖直弹簧振子需等专用评估器和协议接入后再加入正式Task。

发现并验证 Baseline：

```bash
PYTHONPATH=src python -m physbench \
  baseline list

PYTHONPATH=src python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3_generic
```

编译一份 sealed TaskInstance：

```bash
PYTHONPATH=src python -m physbench \
  task-build \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --output results/wan22_generic_task_instance.json
```

创建 AtomicRun；不加 `--execute` 时只冻结并展开计划：

```bash
PYTHONPATH=src python -m physbench \
  atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --output-root run
```

在同一 Task 上成对比较两个 Baseline：

```bash
PYTHONPATH=src python -m physbench \
  matrix-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --baseline cosmos3_nano_i2v_physics \
  --matrix-id cosmos3_generic_vs_physics \
  --output-root run
```

加 `--execute` 才会启动模型。每个矩阵元素仍是独立
`run/<matrix_id>__<baseline_id>/` AtomicRun。

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [数据集说明与新 Scene 导入手册](datasets/README.md)
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
VPhysBench/
├── datasets/                 # 唯一权威数据根；13.0.0是当前release
├── tasks/official/           # direct_eval 与 finetune_eval 两份模型无关 Task
├── baselines/                # schema v5 Bundle、adapter/driver 与本机配置模板
├── configs/evaluation/       # scene evaluator 协议
├── schemas/v3/               # 历史Dataset、Case、Task及当前TaskInstance
├── schemas/v4/               # 当前Task与历史Dataset/Case
├── schemas/v5/               # 当前Dataset/Case与Baseline Bundle
├── src/physbench/            # planner、runtime、评估与 CLI
├── tests/                    # 回归测试
├── docs/                     # 架构和操作文档
└── run/                      # 当前 AtomicRun 输出
```

权威数据资产只能写入 `datasets/`。缩放、抽帧、特征、模型缓存和预测必须进入内容寻址
cache 或当前 run，不能回写 Dataset。
