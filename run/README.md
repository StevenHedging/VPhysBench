# AtomicRun outputs

`run/` 是唯一的运行输出根。干净 checkout 中这里只保留本说明；CLI 创建的
`run/<run_id>/` 不提交到 Git，也不得覆盖权威 Dataset 文件。

每个 run 都封印一个不可混合的
`DatasetSnapshot × TaskSpec × Baseline identity × Seeds` 身份。完整目录契约、重评边界、
状态含义以及正式 score/CSTI/coverage 的判读方式见
[Run 结果解读](../docs/RUN_LAYOUT.md)。
