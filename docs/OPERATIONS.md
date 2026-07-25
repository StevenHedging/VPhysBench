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

不要混用：Benchmark 环境负责 Dataset、TaskBuilder、OpenCV、SAM2 和 evaluator；
WAN 环境由 Baseline bundle 调用。

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
  -m compileall -q src tests
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

## 5. 编译任务

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_generic.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/task_instance.json
```

重复执行后比较 fingerprint，结果必须稳定。

## 6. AtomicRun

冻结但不执行：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

确认实例、路径、checkpoint 和 GPU 后加 `--execute`。

## 7. 重新评估

Prediction 修复或拷贝完成后：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/<run_id>
```

评估会覆盖 run 内当前 evaluation 输出，但不会修改 prediction 或 Dataset。

## 8. 结果检查

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

## 9. 常见错误

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

## 10. 发布检查单

1. 工作树只包含本次有意修改。
2. Dataset hash 验收通过。
3. TaskBuilder 配对计划测试通过。
4. 五 scene scorer 恒等性与扰动测试通过。
5. 五 scene 真实自比为 1。
6. 空 mask 的 observed ratio 正确。
7. 系统测试与 phybench 环境完整测试通过。
8. 文档链接无断链。
9. `git diff --check` 通过。
10. 提交后工作树干净。
