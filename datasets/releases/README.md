# Dataset release 选择规则

当前唯一正式运行入口是：

```text
12.0.0/dataset.json
```

代码必须使用`physbench.data_layout.LATEST_DATASET`，命令行必须显式传入当前descriptor。
不得通过扫描目录、取最大版本、加载失败后回退或“找一个能加载的版本”来选择Release。

12.0.0保持799条Case的媒体、View和来源事实不变，将已校正的V11内容固化为每条Case
唯一的`physics.json`。quantity使用四字段
`value/unit/annotated/symbol`；标量为非负大小，方向由英文符号prompt承担。它使用精简
运行时快照：

```text
README.md
dataset.json
cases.jsonl
scenes/
views/
```

构建、迁移和验证证据不放入运行时Release，而位于
`datasets/provenance/releases/12.0.0/`。逐Case mask manifest和对象mask资产仍由Case引用。

`1.0.0`至`11.0.0`已经从活动目录移除。若审计历史实验，应从Git历史读取其Dataset
ID和旧View；运行时不得扫描、自动回退或加载旧Release。
