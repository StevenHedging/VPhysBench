# AtomicRun Outputs

每个子目录是一个不可混合的：

```text
DatasetSnapshot × AtomicTask × BaselineSnapshot × Seeds
```

标准内容：

```text
<run_id>/
├── run.json
├── plan.json
├── predictions.jsonl
├── frozen/
├── task_instance/
├── jobs/
├── predictions/
├── artifacts/
│   └── prediction_artifacts.json
├── logs/
└── evaluation/
    ├── manifest.json
    ├── case_results.jsonl
    ├── task_result.json
    └── cases/<job_id>/
```

`predictions.jsonl` 是 Baseline 输出边界。`evaluation/` 可用
`physbench evaluate --run-dir <run>` 从相同 predictions 重新生成。

模型代码、权重和可重建 cache 可以位于 run 外；所有 prediction、日志、job payload、
训练派生物和 evaluation 必须位于当前 run。核心会拒绝外部 `video_path`，并记录每个
预测文件的 SHA-256。已有外部视频使用 `physbench prediction-import` 复制归档。
schema v4 submission Bundle 会自动执行同一复制与 provenance 流程，并要求
submission JSONL 完整覆盖 canonical jobs。

正式 Task score 只在 coverage 为 1 时存在。部分运行的 `observed_mean_score` 不能作为
正式结果发布。

Run 不得存放或覆盖权威 Dataset 文件；媒体和模型派生物只属于该 run 或共享 immutable
cache。
