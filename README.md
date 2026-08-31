# VPhysBench

VPhysBench 是面向物理视频生成模型的训练与评测框架。日期分支
`2026-08-31` 是 V14 干净发行版：它保留冻结的 Dataset、Task v1、Evaluation v1、
通用 Baseline 接口和运行时，但不集成任何具体生成算法、已注册 Baseline、模型配置、
权重或运行结果。

Dataset 14.0.0 包含 916 个 case、7 个场景。Task v1 定义两个可复现工作负载：

- `six_scene_direct_eval_v1`：直接评估六个计分场景的全部 775 个 case；
- `six_scene_train_six_scene_eval_v1`：对六场景 679 个训练 case 做场景均衡重采样，随后在六个
  计分场景的 96 个留出 case 上评估。

Dataset 中仍保留 `push_bottle` 的全部 127 个训练 case 和 14 个 test case，但两个
官方 Task v1 都不会选择它们。

两个 Task 都绑定唯一公开协议 `scene_default_v1`。

## 快速开始

需要 Git 和 ffmpeg/ffprobe。元数据环境需要 Python 3.11 或更高版本；完整评测入口
接受 Python 3.12 或更高版本，但发行版 CI 只认证 Python 3.12，验证的组合是
PyTorch 2.10.0 与 CUDA 12.8 wheel。
Dataset 在 Hugging Face 公开发布，首次下载建议预留至少 40 GB 空间。

元数据、Hub 下载和接口 smoke 的可复制粘贴安装方式如下。bootstrap 从脚本自身定位
checkout，创建或复用 `.venv`，并在安装后运行 metadata doctor：

```bash
git clone <VPhysBench repository URL>
cd VPhysBench
bash scripts/bootstrap_env.sh --profile metadata
. .venv/bin/activate
physbench doctor --level metadata
make smoke-interface
physbench dataset pull
physbench baseline list
```

可以先检查不会改动文件或访问网络的安装计划：

```bash
bash scripts/bootstrap_env.sh --profile metadata --dry-run
```

真实场景评测使用独立的 Python 3.12 环境。该 profile 的命令计划固定 Torch 2.10.0/
CUDA 12.8、直接依赖输入和 SAM2/SAM3 源码 commit，随后运行不要求 Dataset 媒体或
checkpoint 的 `runtime` doctor。其他 Python minor 可通过入口兼容性检查，但未被
发行版 CI 认证：

```bash
VPHYSBENCH_BOOTSTRAP_PYTHON=python3.12 \
  bash scripts/bootstrap_env.sh --profile evaluation
. .venv/bin/activate
physbench doctor --level runtime
```

如果 `python3.12` 已在 `PATH`，可以省略 `VPHYSBENCH_BOOTSTRAP_PYTHON`。bootstrap
会拒绝复用不兼容的虚拟环境；重复运行同一 profile 是安全的。它从不下载 Dataset
媒体或 evaluator checkpoint。未在 constraints 中列出的传递依赖在每次安装时解析，
应按 `docs/REPRODUCIBILITY.md` 为每次 Run 保存 `pip freeze`；这不是 bit-for-bit 的完整
依赖锁。

`dataset pull` 只从 binding 锁定的 40 位 Hub commit 直接下载
`assets/**`，并恢复到本地 `datasets/assets/`。下载中断或遇到 Hub
限流后直接重跑即可，Hugging Face 会复用已完成的文件。命令结束前
还会执行一次 Dataset 资产就绪检查。

干净 checkout 中 `baseline list` 应输出空数组。创建自己的 I2V 接入：

```bash
physbench baseline init my_model --backend managed-i2v
physbench baseline validate my_model
```

干净发行版只提供接口 smoke，不携带生成模型或权重。真实实验前需要将
`baselines/my_model/baseline.json` 中的 command 改为模型推理入口，并在
Git 忽略的 `baseline.local.json` 中配置 checkpoint 与工作目录。

Baseline 可以在自己的目录内实现训练和推理脚本，但不能重写官方 Task 的数据选择、
种子或评分协议。基准先冻结 canonical plan，随后 Baseline 只负责适配、编译和执行。

先运行一个 case 前，下载完整 Dataset、配置 protocol-pinned SAM3.1 checkpoint，并确认
full doctor 成功（退出 0）：

```bash
export VPHYSBENCH_SAM31_CHECKPOINT=SAM31_CHECKPOINT_ABSOLUTE_PATH
physbench doctor  # 必须退出 0
```

`physbench doctor` 是唯一的 readiness authority。`metadata` 只检查 Python 3.11+
和发行版绑定；缺少 Dataset 媒体在这一层是 warning。`runtime` 需要 Python 3.12+
并检查系统命令、evaluator imports、PyTorch/CUDA 和轻量 evaluator smoke，但不要求
Dataset 或 checkpoint。`evaluation` 保留 Dataset 和 SAM2 evaluator 检查；默认 `full`
再检查 SAM3.1 checkpoint 及其协议摘要。doctor 是只读的：不会安装依赖、下载
Dataset/权重或加载大模型。`--json` 始终输出一个 versioned JSON document；所有 required
check 通过时退出 0，否则退出 1，因此外部资产尚未配置时 `ready: false` 是预期结果。
将 `SAM31_CHECKPOINT_ABSOLUTE_PATH` 替换为本机 checkpoint 的绝对路径。

然后运行：

```bash
physbench atomic-run \
  --dataset datasets/releases/14.0.0/dataset.json \
  --task tasks/official/six_scene_direct_eval_v1.json \
  --baseline baselines/my_model \
  --case-id circular_r1_silver02cm_img_0370 \
  --run-id my_model_smoke \
  --output-root run \
  --execute
```

所有预测、日志和评测结果都属于 `run/<run_id>/`；不要把运行结果写入 Dataset。

## 仓库边界

可移植单元是**干净的 tracked checkout**，而不是它所在机器的状态。可以把只含 tracked
文件的 checkout 解压或移动到任意路径（包括路径中含空格的目录），然后从 checkout 外运行
bootstrap 或 CLI。下列内容不属于发行物、也不能通过复制 checkout 获得：`.venv`、pip/
Hugging Face cache、Dataset 媒体、`baseline.local.json` 本地覆盖、checkpoint、凭据以及
`run/` 中的输出。Dataset 媒体必须由冻结的 Hub binding 显式拉取，SAM3.1 由
`VPHYSBENCH_SAM31_CHECKPOINT` 显式指定；自定义 Baseline 的机器路径只应写在被 Git
忽略的 `baseline.local.json`。

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
- [接入自定义 I2V 算法](docs/CUSTOM_BASELINE_QUICKSTART.md)
- [Baseline Integration 完整契约](docs/BASELINE_INTEGRATION.md)
- [DataAdapter 与输入策略](docs/DATA_ADAPTER.md)
- [导入已有预测视频](docs/SUBMISSION_QUICKSTART.md)
- [Run 结果解读](docs/RUN_LAYOUT.md)
- [复现实验](docs/REPRODUCIBILITY.md)

`examples/dummy_i2v_command.py` 只验证外部命令和视频协议，不是参考算法。
