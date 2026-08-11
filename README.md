# VPhysBench

VPhysBench 是面向物理视频生成模型的训练与评测框架。日期分支
`2026-08-11` 是干净发行版：它保留冻结的 Dataset、Task v1、Evaluation v1、
通用 Baseline 接口和运行时，但不集成任何具体生成算法、已注册 Baseline、模型配置、
权重或运行结果。

Dataset 13.0.0 包含 916 个 case、7 个场景。Task v1 定义两个可复现工作负载：

- `five_scene_direct_eval_v1`：直接评估五个计分场景的全部 658 个 case；
- `six_scene_train_five_scene_eval_v1`：使用六场景 679 个训练 case，随后在五个
  计分场景的 76 个留出 case 上评估。

Dataset 中仍保留 `push_bottle` 的全部 127 个训练 case 和 14 个 test case，但两个
官方 Task v1 都不会选择它们。

两个 Task 都绑定唯一公开协议 `scene_default_v1`。

## 快速开始

需要 Python 3.11、Git、ffmpeg/ffprobe，以及对私有 Hugging Face Dataset 的访问权。
`.[hub]` 只提供 Hub 下载和轻量接口；真实场景评测和 `atomic-run` 还需要
`.[scene-evaluation]`。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hub]"

# Hub 下载、元数据检查和接口 smoke 只需要 .[hub]
physbench doctor --level metadata
make smoke-interface

hf auth login
physbench dataset pull
physbench baseline list
```

`dataset pull` 只请求冻结 revision 中的 distribution v1 manifest 和 13 个
校验分片，不再逐个解析数千个媒体文件。分片缓存在
`datasets/.vphysbench/cache/<revision>/`；下载中断或遇到 Hub 限流后直接重跑即可，
已通过 SHA-256 的分片会复用。全部分片、文件和 Dataset digest 验证完成前，
正式 `datasets/assets/` 不会切换。

干净 checkout 中 `baseline list` 应输出空数组。创建自己的 I2V 接入：

```bash
physbench baseline init my_model --backend managed-i2v
physbench baseline validate my_model
```

Baseline 可以在自己的目录内实现训练和推理脚本，但不能重写官方 Task 的数据选择、
种子或评分协议。基准先冻结 canonical plan，随后 Baseline 只负责适配、编译和执行。

先运行一个 case。`atomic-run` 会执行真实 evaluator，因此在运行前安装 evaluator
stack，并在下载完整私有 Dataset 后确认 evaluation readiness：

```bash
python -m pip install -e ".[scene-evaluation]"
physbench doctor --level evaluation
```

然后运行：

```bash
physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_v1.json \
  --baseline baselines/my_model \
  --case-id circular_r1_silver02cm_img_0370 \
  --run-id my_model_smoke \
  --output-root run \
  --execute
```

所有预测、日志和评测结果都属于 `run/<run_id>/`；不要把运行结果写入 Dataset。

## 仓库边界

```text
datasets/          Dataset 元数据与固定 Hugging Face 绑定
tasks/official/    两个模型无关的 Task v1
baselines/         用户接入目录；发行版初始只有 README
configs/           唯一公开 Evaluation v1 与场景配置
schemas/           Dataset、Task、Baseline、Evaluation 和 Run 契约
src/physbench/     规划、通用运行时与 evaluator
examples/          非 Baseline 的接口测试夹具
tests/             CPU 元数据、接口和 evaluator 回归
run/               唯一运行输出根；发行版初始只有 README
```

## 文档

- [架构与解耦边界](docs/ARCHITECTURE.md)
- [Task v1](docs/TASKS.md)
- [Benchmark 协议](docs/BENCHMARK_PROTOCOL.md)
- [Evaluation v1](docs/EVALUATION.md)
- [安装与首个运行](docs/GETTING_STARTED.md)
- [接入自定义 I2V/V2V 算法](docs/CUSTOM_BASELINE_QUICKSTART.md)
- [导入已有预测视频](docs/SUBMISSION_QUICKSTART.md)
- [Run 目录契约](docs/RUN_LAYOUT.md)
- [复现实验](docs/REPRODUCIBILITY.md)

`examples/dummy_i2v_command.py` 只验证外部命令和视频协议，不是参考算法。
