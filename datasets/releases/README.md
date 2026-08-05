# Dataset release 选择规则

当前唯一正式运行入口是：

```text
8.0.0/dataset.json
```

`1.0.0`至`7.0.0`只保留用于解释既有实验记录中的Dataset ID、digest、Case和旧View。
它们不是候选当前版本，不得通过扫描目录、取最大版本失败后回退或“找一个能加载的版本”
来选择。代码必须使用`physbench.data_layout.LATEST_DATASET`，命令行必须显式传入当前
descriptor。

`9.0.0`在8.0.0的相同Case事实上增加逐主体首帧mask；在官方Task显式迁移到其
Dataset ID之前，不自动替换8.0.0运行默认值。

若确实复现历史实验，调用者必须同时提供历史Task或run中记录的Dataset ID/digest；
否则使用旧release属于配置错误。发布新release时可以基于全部有效Case重新划分，旧
train/test成员身份不自动继承，但必须输出partition movement audit。
