# AtomicRun Outputs

每个子目录是一个不可混合的：

```text
DatasetSnapshot × AtomicTask × BaselineSnapshot × Seeds
```

标准内容：

```text
<run_id>/
├── run.json
├── task_instance.json
├── plan.json
├── predictions.jsonl
├── artifacts/
├── logs/
└── evaluation/
    ├── manifest.json
    ├── case_results.jsonl
    ├── task_result.json
    └── cases/<job_id>/
```

`predictions.jsonl` 是 Baseline 输出边界。`evaluation/` 可用
`physbench evaluate --run-dir <run>` 从相同 predictions 重新生成。

正式 Task score 只在 coverage 为 1 时存在。部分运行的 `observed_mean_score` 不能作为
正式结果发布。

Run 不得存放或覆盖权威 Dataset 文件；媒体和模型派生物只属于该 run 或共享 immutable
cache。
