# VPhysBench

VPhysBench 是面向物理视频生成模型的训练与评测框架。日期分支
`2026-08-08` 是供协作者使用的干净近似发行版：它保留 Dataset、Task、Evaluator、
通用 baseline 接口和 submission 导入能力，但不集成任何生成算法、模型配置或权重。

Dataset 13.0.0 包含 916 个 case、7 个场景。其中 5 个场景进入正式计分 Task：

- `pendulum`
- `collision_1d`
- `inclined_plane_slide`
- `uniform_circular_motion`
- `parabolic_motion`

`push_bottle` 和 `vertical_spring_oscillator` 是 preview/data-only 场景，不进入正式总分。

## 快速开始

需要 Python 3.11、Git、ffmpeg/ffprobe，以及对私有 Hugging Face Dataset 的访问权。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hub]"

hf auth login
physbench dataset pull
physbench doctor --level evaluation

make smoke-interface
physbench baseline list
```

干净 checkout 中 `baseline list` 应输出空数组。创建自己的 I2V 接入：

```bash
physbench baseline init my_model --backend managed-i2v
physbench baseline validate my_model
```

脚手架位于 `baselines/my_model/`。把其中的通用命令替换为自己的推理入口后，先运行
一个 case：

```bash
physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline baselines/my_model \
  --case-id circular_r1_silver02cm_img_0370 \
  --run-id my_model_smoke \
  --output-root run \
  --execute
```

所有预测、日志和评测结果都属于 `run/<run_id>/`；生成视频位于
`run/<run_id>/predictions/`。不要把运行结果写入 Dataset。

## 仓库边界

```text
datasets/          Dataset 元数据与固定 Hugging Face 绑定
tasks/official/    模型无关的正式 Task
baselines/         用户算法接入目录；初始只有 README
configs/           场景与评测协议
schemas/           Dataset、Task、Baseline 和 Run 契约
src/physbench/     CLI、运行时和 evaluator
examples/          未注册的协议夹具
tests/             CPU 元数据、接口和 evaluator 回归
run/               唯一运行输出根；初始只有 README
```

## 文档

- [安装与首个运行](docs/GETTING_STARTED.md)
- [接入自定义 I2V/V2V 算法](docs/CUSTOM_BASELINE_QUICKSTART.md)
- [导入已有预测视频](docs/SUBMISSION_QUICKSTART.md)
- [Benchmark 协议](docs/BENCHMARK_PROTOCOL.md)
- [Run 目录契约](docs/RUN_LAYOUT.md)
- [复现实验](docs/REPRODUCIBILITY.md)
- [完整 baseline 接口](docs/BASELINE_INTEGRATION.md)
- [Evaluator 细节](docs/EVALUATION.md)

`examples/dummy_i2v_command.py` 只验证命令和视频协议，不是参考算法，其分数没有研究意义。
