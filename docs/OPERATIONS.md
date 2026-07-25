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

不要混用：Benchmark 环境负责 Dataset、通用 command host、TaskBuilder 协议、
OpenCV、SAM2 和 evaluator；WAN 模型环境只由 Bundle 内执行器调用。

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
  -m compileall -q src tests baselines/wan22_lora/plugin
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

调用 command endpoint 并验证组件指纹：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_v3
```

新机器必须先从 `baselines/wan22_lora/baseline.local.example.json` 创建
`baseline.local.json`。不要把机器绝对路径写回 portable manifest。

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

## 8. 重新评估

Prediction 修复或拷贝完成后：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/<run_id>
```

评估会覆盖 run 内当前 evaluation 输出，但不会修改 prediction 或 Dataset。

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

先运行 `baseline validate`。检查 entrypoint 位于 Bundle 内、使用 Benchmark
`phybench` Python 可 import `physbench`，并确认 endpoint 无语法错误。模型 runtime
Python 不是协议 endpoint Python。

### `task instance targets a different Baseline deployment`

构建实例后 `baseline.local.json`、portable manifest、插件代码或 profile 已改变。
重新构建 TaskInstance；不要复用旧实例绕过 digest 检查。

### `prediction_video_missing`

Prediction record 的 `video_path` 为空、相对到错误目录或文件不存在。修复 record/path，
不要把该 case 记零。

### `insufficient_duration`

生成视频没有覆盖协议物理区间。检查 predictor 时长、容器 FPS 和时间戳。

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
3. Baseline discovery、command endpoint 和两类 digest 验收通过。
4. TaskBuilder 配对计划与 canonical-plan 防篡改测试通过。
5. 五 scene scorer 恒等性与扰动测试通过。
6. 五 scene 真实自比为 1。
7. 空 mask 的 observed ratio 正确。
8. 系统测试与 phybench 环境完整测试通过。
9. 文档链接无断链。
10. `git diff --check` 通过。
11. 提交后工作树干净。
