# 运行、验证与故障排查

## 1. 环境

Benchmark：

```text
/root/miniconda3/envs/phybench
```

WAN 执行：

```text
/root/miniconda3/envs/dlp
```

Cosmos3 执行：

```text
/root/Nico/cosmos/packages/cosmos3/.venv
```

不要混用：Benchmark 环境负责 Dataset、Registry、managed runtime、command host、
OpenCV、SAM2 和 evaluator；WAN/Cosmos 模型环境只由各自 Bundle driver/executor 调用。

## 2. 安装

```bash
cd /root/Steven/physics_video_benchmark
/root/miniconda3/envs/phybench/bin/pip install -e ".[scene-evaluation]"
/root/miniconda3/envs/phybench/bin/pip install -e /root/Jensen/Eval/sam2-main
```

验证：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python - <<'PY'
import cv2
import numpy
import torch
import sam2
print("opencv", cv2.__version__)
print("numpy", numpy.__version__)
print("cuda", torch.cuda.is_available())
print("sam2", sam2.__file__)
PY
```

## 3. 测试

```bash
make test
```

系统 Python 缺少 evaluation extras 时，scene 测试明确 skip。正式完整测试：

```bash
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v
```

源码编译检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  -m compileall -q src tests baselines
git diff --check
```

## 4. Dataset 验收

快速 metadata 验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json
```

发布前完整验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-asset-hashes
```

完整验收读取 468 个锁定文件，耗时取决于磁盘。

## 5. Baseline 发现与部署验收

首次接入模型前，先按[自定义 Baseline 集成指南](BASELINE_INTEGRATION.md)选择
submission、managed 或 command 接口并完成 Bundle。

只读取 manifests：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list
```

解析 `baseline.local.json` 并查看 portable/deployment identity：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect wan22_ti2v_5b_lora_r32_v3
```

按各 Bundle 的 kind 验证 command endpoint 或 managed/submission runtime，并检查组件
指纹：

```bash
for baseline_id in \
  wan22_ti2v_5b_lora_r32_v3 \
  cosmos3_nano_i2v \
  wan22_g15_sparse_motion_r32_e20
do
  PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
    baseline validate "$baseline_id"
done
```

新机器必须先从目标 Bundle 的 `baseline.local.example.json` 创建
`baseline.local.json`。不要把机器绝对路径写回 portable manifest。Cosmos build 会
验证轻量 snapshot identity；G15 task build 会验证完整 154 MiB LoRA SHA-256。

创建新接入目录：

```bash
# 标准 I2V managed Bundle
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_i2v --backend managed-i2v

# output-only submission Bundle
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline init my_outputs --backend submission
```

脚手架生成后会立即经过正式 Registry 校验。managed 的通用 CLI driver 接受
`--prompt/--image/--output/--seed`；不满足该契约时只需替换 Bundle 内的
`driver.py`，无需修改 Registry。

## 6. 编译任务

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_generic.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output /tmp/task_instance.json
```

重复执行后比较 fingerprint，结果必须稳定。

`--baseline` 也可传 Bundle 目录或 manifest 路径。重复执行后比较 instance、bundle 和
deployment fingerprint，结果必须稳定。

## 7. AtomicRun

冻结但不执行：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline wan22_ti2v_5b_lora_r32_v3 \
  --output-root runs_v2
```

确认实例、路径、checkpoint 和 GPU 后加 `--execute`。

Cosmos 一个 job 默认同时占用四张 GPU；WAN 为每张可用 GPU 创建一个常驻 worker。
运行前用 `nvidia-smi` 检查实际空闲设备，并通过 local override 改 GPU 列表。修改后
deployment digest 会变化，必须重新 build。

G15 是污染审计明确的 diagnostic baseline。全量 AtomicRun 可用于复现和诊断，但
`evaluation_score` 不得进入无泄漏排名。clean subset 必须明确选择
`provenance/benchmark_overlap_v3.json` 所述 source-unseen case，且覆盖率不是完整
Task coverage。

## 8. 重新评估

Prediction 修复或拷贝完成后：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/<run_id>
```

评估会覆盖 run 内当前 evaluation 输出，但不会修改 prediction 或 Dataset。

### 导入已有预测

模型目录中已有的视频必须先复制到 Bench 内，不能直接把外部路径写入
`predictions.jsonl`：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  prediction-import \
  --source /path/to/existing_prediction.mp4 \
  --run-dir runs_v2/<run_id> \
  --baseline-id <baseline_id> \
  --case-id <case_id> \
  --job-id <job_id> \
  --seed 42
```

命令执行原子复制，验证源与目标 SHA-256，并写入
`provenance/prediction_imports.json`。重复导入相同内容是幂等的；目标存在但内容不同
时会失败。外部源文件不会被删除。评测记录应改为导入后返回的
`destination_path`。

## 9. 结果检查

```text
evaluation/task_result.json
```

首先检查：

- `status`；
- `coverage`；
- `status_counts`；
- `score`；
- `observed_mean_score`；
- `by_scene`；
- `breakdown`。

Case 调试：

```text
evaluation/cases/<job_id>/result.json
evaluation/cases/<job_id>/per_frame.csv
evaluation/cases/<job_id>/physical_subject_iou_curve.png
```

## 10. 常见错误

### `unknown baseline ID`

运行 `physbench baseline list`。确认目录直接位于 `baselines/` 下、文件名为
`baseline.json` 且 `baseline_id` 全局唯一。路径引用不存在时不会回退为 ID。

### `Baseline command produced no response`

只适用于 `implementation.kind=command`。
先运行 `baseline validate`。检查 entrypoint 位于 Bundle 内、使用 Benchmark
`phybench` Python 可 import `physbench`，并确认 endpoint 无语法错误。模型 runtime
Python 不是协议 endpoint Python。

### `managed baseline driver not found` / `managed driver must export Driver`

`implementation.driver` 必须是 Bundle 内相对路径，且模块必须导出
`ManagedDriver` 子类 `Driver`。普通 direct-eval I2V 可直接 re-export
`StandardI2VCLIDriver`。

### `submission coverage mismatch`

submission JSONL 与 canonical jobs 不完全相同。检查是否缺 job、混入另一 Task 的 job，
以及 case/conditioning/seed 是否一致。不要把 coverage 检查改成“有多少评多少”。

### `task instance targets a different Baseline deployment`

构建实例后 `baseline.local.json`、portable manifest、driver/endpoint 代码或 profile 已改变。
重新构建 TaskInstance；不要复用旧实例绕过 digest 检查。

### `baseline does not support task family finetune_eval`

Cosmos3 base 与冻结 G15 都是 direct-only，这是能力约束，不是部署错误。使用
`five_scene_direct_eval_{generic,physics}.json`。只有
`wan22_ti2v_5b_lora_r32_v3` 当前支持 View A fine-tuning。

### `Cosmos3 checkpoint identity mismatch`

local checkpoint 不是 manifest 声明的 base snapshot，或 identity 文件已变化。不要
覆盖 digest；为另一个 snapshot/SFT 建立新的 baseline ID。

### `frozen LoRA checkpoint digest mismatch`

G15 local path 没有指向 step-2840，或文件损坏。预期 SHA-256 在 manifest 与
provenance audit 中各保存一份。

### `prediction_video_missing`

Prediction record 的 `video_path` 为空、相对到错误目录或文件不存在。修复 record/path，
不要把该 case 记零。

### `insufficient_duration`

生成视频没有覆盖协议物理区间。检查 predictor 时长、容器 FPS 和时间戳。
不要通过补 GT 首帧、复制末帧或放宽 evaluator tolerance 处理。WAN 当前分别记录
`target_frames`（reference derivative）与 `generation_target_frames`（prediction
coverage）；若两者被旧产物错误混用，应重新生成。

### `motion_prompt_failed`

主体运动 proposal 不可靠。查看采样帧和 scene-specific observation 配置，不要直接使用
reference mask 作为 prediction mask。

### `insufficient_valid_masks` / `insufficient_instance_tracks`

分割覆盖率低于协议阈值。检查主体 prompt、遮挡、模糊和 mask 面积；错误状态优于伪造分数。

### SAM2 `_C` warning

当前 fallback 可运行，但缺少 hole-filling 后处理。正式部署应按 SAM2 官方安装流程编译
扩展，并用真实自比测试验证环境变化没有改变 evaluator fingerprint/结果。

### Task score 为 `null`

这是严格 coverage 的预期行为。查看 `status_counts` 和缺失 case；不要使用
`observed_mean_score` 冒充正式分数。

## 11. 发布检查单

1. 工作树只包含本次有意修改。
2. Dataset hash 验收通过。
3. Baseline discovery、三档实现加载和两类 digest 验收通过。
4. 三个 Bundle 的能力拒绝、TaskBuilder 配对计划与 canonical-plan 防篡改测试通过。
5. 五 scene scorer 恒等性与扰动测试通过。
6. 五 scene 真实自比为 1。
7. 空 mask 的 observed ratio 正确。
8. 系统测试与 phybench 环境完整测试通过。
9. 文档链接无断链。
10. `git diff --check` 通过。
11. 提交后工作树干净。
12. 预训练数据重叠 audit 已复核，诊断结果没有混入正式可比结果。
13. `artifacts/prediction_artifacts.json` 中所有预测都位于当前 run，digest 可复核。
