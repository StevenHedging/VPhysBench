# Dataset release 选择规则

当前唯一正式运行入口是：

```text
10.0.0/dataset.json
```

代码必须使用`physbench.data_layout.LATEST_DATASET`，命令行必须显式传入当前descriptor。
不得通过扫描目录、取最大版本、加载失败后回退或“找一个能加载的版本”来选择Release。

10.0.0保持9.0.0的799条Case事实不变，为每条Case新增一个受锁定的Case-local
`physics.json`，共6,038个锁定资产。它使用精简运行时快照：

```text
README.md
dataset.json
release.json
cases.jsonl
assets.lock.json
scenes/
views/
```

构建、迁移和验证证据不放入运行时Release，而位于
`datasets/provenance/releases/10.0.0/`。9.0.0的全局`masks.jsonl`未进入10.0.0；逐Case
mask manifest和对象mask资产仍由Case引用并受资产锁保护。

`1.0.0`至`9.0.0`仅保留用于解释既有实验记录中的Dataset ID、digest、Case和旧View。
若复现历史实验，调用者必须同时提供历史Task或run中记录的Dataset ID/digest；否则
使用旧Release属于配置错误。历史Release不可修改。
