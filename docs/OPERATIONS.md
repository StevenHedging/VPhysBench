# 运行、验证与故障排查

## 1. 环境

Benchmark：

```text
/root/miniconda3/envs/phybench
```

WAN 模型进程通常使用：

```text
/root/miniconda3/envs/dlp
```

Cosmos 模型进程通常使用其工程虚拟环境：

```text
/root/Nico/cosmos/packages/cosmos3/.venv
```

`phybench` 环境负责 Dataset、Registry、TaskBuilder、run orchestration 和 evaluator；
模型环境只由 Bundle driver 调用。实际路径以各目录 Git-ignored 的
`baseline.local.json` 为准。

## 2. 安装与测试

```bash
cd /root/Steven/physics_video_benchmark

/root/miniconda3/envs/phybench/bin/pip install -e ".[scene-evaluation]"
/root/miniconda3/envs/phybench/bin/pip install -e /root/Jensen/Eval/sam2-main
```

完整测试：

```bash
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v
```

静态检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  -m compileall -q src tests baselines

git diff --check
```

## 3. Dataset 验收

快速检查 metadata 与资产存在性：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --check-assets
```

发布或迁移机器前执行完整哈希验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --check-asset-hashes
```

预期 Dataset identity：

```text
physics_video_five_scene_v4
be5ea8880be3cf8e0d0d316ee025cf02905ef5aeb544f3df5f4b966e5e14782d
```

## 4. Baseline 本机配置

新机器先复制目标目录的模板：

```bash
cp baselines/wan22_lora/baseline.local.example.json \
  baselines/wan22_lora/baseline.local.json
```

填写 checkpoint、模型 Python、工程根目录与 GPU。Local override 只能覆盖 `model` 和
`runtime`；不要把绝对路径写回 portable manifest。

同目录的 `baseline.json` 与 `physics.baseline.json` 共享该 local override。它们表示
同一模型部署的不同固定输入策略，而不是 Task 的两个运行模式。

## 5. Baseline 发现与部署验收

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline list
```

当前应发现七个 Baseline ID。检查解析结果：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect wan22_ti2v_5b_lora_r32_v3_generic

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline inspect wan22_ti2v_5b_lora_r32_v3_physics
```

验证部署与指纹：

```bash
for baseline_id in \
  wan22_ti2v_5b_lora_r32_v3_generic \
  wan22_ti2v_5b_lora_r32_v3_physics \
  wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  cosmos3_nano_i2v_generic \
  cosmos3_nano_i2v_physics \
  wan22_g15_sparse_motion_r32_e20_generic \
  wan22_g15_sparse_motion_r32_e20_physics
do
  PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
    baseline validate "$baseline_id"
done
```

`baseline validate` 不应加载完整 GPU 模型，但会检查 driver、关键外部依赖、
checkpoint identity 与 adapter/TaskBuilder fingerprint。

## 6. 编译 TaskInstance

Direct-eval：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --output /tmp/cosmos3_generic_task_instance.json
```

Fine-tune + eval：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --output /tmp/wan22_physics_finetune_task_instance.json
```

重复构建相同 Dataset + Task + Baseline deployment 时，TaskInstance digest 必须稳定。
重点核对 `canonical_plan`、`adaptations`、`input_policy`、used parameters、
generation shape 与 identity。

## 7. AtomicRun

不执行模型的 dry-run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --scene-id pendulum \
  --case-id CASE_ID \
  --run-id cosmos3_generic_pendulum_dryrun \
  --output-root runs_v2
```

Dry-run 会冻结 plan/TaskInstance、展开 adapter、写 job 与 planned prediction，但不启动
推理。确认 checkpoint、prompt、首帧、shape、seed 与输出路径后，使用新的 run ID 并
加 `--execute`：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --scene-id pendulum \
  --case-id CASE_ID \
  --run-id cosmos3_generic_pendulum_execute \
  --output-root runs_v2 \
  --execute
```

AtomicRun 不覆盖已有目录。单 case 或 subset run 是工程诊断，coverage 不完整时没有正式
Task score。

## 8. 同 Task 的 Baseline 矩阵

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --baseline cosmos3_nano_i2v_physics \
  --matrix-id cosmos3_generic_vs_physics \
  --output-root runs_v2
```

矩阵在执行前验证两个 Baseline 的 Dataset、split、seed 与 jobs 完全一致。不加
`--execute` 只创建两个 dry-run；加 `--execute` 才启动模型。

输出：

```text
runs_v2/cosmos3_generic_vs_physics.matrix.json
runs_v2/cosmos3_generic_vs_physics__cosmos3_nano_i2v_generic/
runs_v2/cosmos3_generic_vs_physics__cosmos3_nano_i2v_physics/
```

每个元素是独立 AtomicRun，不能把两个 Baseline 的预测写进同一个 run。

矩阵索引的 `orchestration_status` 表示编排过程是否完整，`status` 汇总 AtomicRun
本身的状态。正常 dry-run 会得到 `orchestration_status=complete` 与
`status=planned`；只有所有预测真正完成时，聚合 `status` 才是 `complete`。

## 9. Run 输出

```text
runs_v2/<run_id>/
├── run.json
├── state.json
├── plan.json
├── report.md
├── component_fingerprints.json
├── artifact_policy.json
├── task_builder.json
├── data_adapter.json
├── frozen/
│   ├── dataset.json
│   ├── cases.jsonl
│   ├── task.json
│   ├── baseline.json
│   ├── views.json
│   └── assets.lock.json
├── task_instance/
│   ├── manifest.json
│   ├── canonical_plan.json
│   ├── adaptations.jsonl
│   ├── training.json
│   ├── inference_jobs.jsonl
│   ├── execution_graph.json
│   ├── cache_bindings.json
│   └── baseline_payload.json
├── adaptations/
├── training/
├── jobs/
├── predictions/
├── predictions.jsonl
├── logs/
├── artifacts/
│   └── prediction_artifacts.json
└── evaluation/
    ├── manifest.json
    ├── case_results.jsonl
    ├── task_result.json
    ├── case_metrics.jsonl        # legacy consumer projection
    ├── summary.json              # legacy consumer projection
    └── cases/<job_id>/
```

预测路径不再按 Task 输入分支分层；Baseline ID 已唯一表达输入策略。具体模型可在
`predictions/` 下增加自己的 job 目录，例如 Cosmos 使用
`predictions/<job_id>/vision.mp4`。

除模型代码、权重与可重建 cache 外，prediction、job payload、日志、训练派生物与
evaluation 都必须位于当前 run。Artifact validator 会检查路径、存在性、大小和
SHA-256。

## 10. 导入已有预测

外部视频必须复制进 run：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  prediction-import \
  --source /path/to/existing_prediction.mp4 \
  --run-dir runs_v2/RUN_ID \
  --baseline-id BASELINE_ID \
  --case-id CASE_ID \
  --job-id JOB_ID \
  --seed 42
```

该命令原子复制并记录源/目标 digest。外部源不会被删除；正式 prediction record 应使用
返回的 run-local `destination_path`。

## 11. 重新评估

当前 AtomicRun：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  evaluate --run-dir runs_v2/RUN_ID
```

命令读取冻结 TaskInstance、Case 与 `predictions.jsonl`，重建 evaluation；不重新规划
Task，也不调用 Baseline。

对于没有 `task_instance/manifest.json` 的历史 v1 run，CLI 保留 legacy evaluator
路径。它对冻结模型输入与任务语义只读，只允许从已有 prediction 重建 evaluation
产物；不会重新规划或调用 Baseline。旧的主动 prompt/task registry 与 planner 已移除，
不能用于创建新实验或重新产生旧 prompt arms。

## 12. 评估结果检查

先看：

```text
evaluation/task_result.json
```

重点字段：

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

正式 Task score 只在严格 coverage 为 1 时存在。不要把部分运行的
`observed_mean_score` 当作正式结果。

## 13. 常见错误

### `unknown baseline ID`

运行 `baseline list`。确认 manifest 位于 `baselines/<dir>/baseline.json` 或
`baselines/<dir>/*.baseline.json`，且 `baseline_id` 全局唯一。

### `baseline bundle must use schema_version=5.0`

旧 v3/v4 Bundle 不能编译新 Task。迁移 manifest、input policy、adapter 与 managed
driver，不要绕过 loader。

### `baseline does not support task family finetune_eval`

Cosmos3 与 G15 是 direct-only。使用 `five_scene_direct_eval.json`；当前只有
`wan22_ti2v_5b_lora_r32_v3_*` 和
`wan22_ti2v_5b_lora_r32_quantity_embedding_v1` 声明 finetune-eval trainer。

### `managed finetune_eval baseline requires trainer recipe`

Baseline capability 声明了 fine-tune family，但 manifest 没有 `trainer`。补齐真实训练
recipe，或删除虚假 capability。

### `physics-ignored adaptation ...`

该 Baseline 声明 `usage=ignored`，adapter 却登记了物理参数/channel。修正 adapter；
若模型确实使用物理信息，应注册 physics Baseline identity。

### `physics-required adaptation ...`

该 Baseline 要求物理输入，但某条 adaptation 没有消费 annotated 参数。检查模板白名单、
单位与 Case 标注；不要静默退化成 generic 行为。

### `managed I2V case has no first-frame asset`

回到 Dataset provenance 流程补齐 `assets.first_frame`。不要从 reference 临时抽帧。

### V2V input aliases reference/source

V2V 输入资产与 evaluator reference 或 source video 相同，或内容 digest 相同。提供真正
独立的输入视频；其中 `conditioning_video` 只是媒体角色。

### `task instance targets a different managed Baseline deployment`

TaskInstance 构建后 manifest、local override、driver、adapter、checkpoint 或外部依赖
变化。重新 build，不要修改 digest 绕过验证。

### `managed output must be inside run predictions`

Driver 把视频写到了模型工程或 `/tmp`。改为
`run_dir / "predictions" / ...`。

### `prediction_video_missing`

Prediction 标记 complete 但文件缺失或路径错误。修复模型输出；不要把该 case 静默记零。

### `insufficient_duration`

生成视频没有覆盖评估物理区间。修正生成帧数、FPS 或容器时间戳；不要补 GT 首帧、
复制末帧或放宽 evaluator tolerance。

### `insufficient_valid_masks`

查看 Case 目录的采样帧、mask 与 IoU 曲线。错误状态优于伪造分数。

### Task score 为 `null`

严格 coverage 未满足。查看 `status_counts` 与缺失 job，不要用 observed mean 替代。

## 14. 运行保留与清理

- `runs_v2/`：当前、不可混合的 AtomicRun；
- `runs/`：历史实验，只为 provenance 或 legacy reevaluate 保留；
- `results/`：可选的跨 run 汇总，不是生成视频或 TaskInstance 的权威来源；
- 可删除失败 smoke、无引用 dry-run、`__pycache__` 与可重建 cache；
- 应保留发布报告引用的 run、昂贵训练制品、checkpoint identity 与 Dataset provenance。

清理前先检查 `baseline.local.json` 是否仍引用历史 run 内的 checkpoint。不要整体删除
`runs/` 或 `runs_v2/`。

## 15. 发布检查单

1. Dataset 完整哈希验收通过。
2. 七个当前 manifest 均可 discovery/validate。
3. 同一 Task 的 generic/physics canonical plan 完全相同。
4. ignored/required input policy audit 通过。
5. TaskInstance deterministic，bundle/deployment/adapter 指纹齐全。
6. 预测、日志与评估均 run-local。
7. 五个 scene evaluator 的 identity/扰动测试通过。
8. 正式 Task coverage 为 1。
9. G15 等数据重叠 Baseline 标为 diagnostic。
10. unittest、compileall 与 `git diff --check` 通过。
