# v2 AtomicRun outputs

每个子目录对应一个不可混合的：

```text
DatasetSnapshot × AtomicTask × BaselineSnapshot × Seeds
```

`import_validation/` 仅包含不调用模型训练/推理的 dry-run，用于验证真实 Dataset 上的
计划、条件隔离、媒体缓存绑定和 checkpoint 路由。

`*.matrix.json` 只是多个 AtomicRun 的编排索引，不是新的 Benchmark 领域要素。
