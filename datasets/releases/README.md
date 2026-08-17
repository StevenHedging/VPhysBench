# Dataset release 选择规则

当前唯一正式运行入口是：

```text
14.0.0/dataset.json
```

代码必须使用`physbench.data_layout.LATEST_DATASET`，命令行必须显式传入当前descriptor。
不得通过扫描目录、取最大版本、加载失败后回退或“找一个能加载的版本”来选择Release。

14.0.0包含916条Case和7个Scene，其中新增117条经过逐条视觉复核的竖直弹簧振子Case。每条Case在资产目录中拥有唯一的
`caption.json`与`physics.json`；`cases.jsonl`只保存身份、运行时资产、appearance和temporal，Loader负责
物化兼容的`case.text`与`case.physics`。它使用精简运行时快照：

```text
dataset.json
cases.jsonl
scenes/
views/
```

构建、迁移和验证证据不放入运行时Release，而位于
`datasets/provenance/releases/14.0.0/`，其中`cases.jsonl`保存逐Case来源、采集时序说明与alignment。
逐Case mask manifest和对象mask资产仍由Case引用。`reference_video`是唯一GT角色。

`1.0.0`至`13.0.0`已经从活动目录移除。若审计历史实验，应从Git历史读取其Dataset
ID和旧View；运行时不得扫描、自动回退或加载旧Release。
