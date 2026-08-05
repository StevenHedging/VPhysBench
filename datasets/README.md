# Datasets

`datasets/` 是仓库唯一权威数据根。当前正式入口：

```text
releases/12.0.0/dataset.json
```

目录职责：

- `assets/`：按scene和Case组织source、canonical媒体、mask与唯一的Case-local
  `physics.json`；
- `provenance/`：导入来源、原始标注、原始压缩包，以及构建、迁移和验证证据；
- `releases/12.0.0/`：当前运行默认的精简Dataset快照，包含799个Case；
- `releases/`：只保留`12.0.0/`这一份活动运行快照；旧版本从Git历史追溯。

12.0.0的每条Case同时包含内联`physics`和
`assets.physics_annotation = assets/<scene>/<case>/physics.json`。内联字段是Baseline和
Evaluator的兼容运行时API；文件是与资产Case绑定的权威副本。Loader在
任何加载模式下都要求两者完全相等。

12.0.0的quantity包含稳定`symbol`，并将标量统一为非负大小；方向只写入prompt。
`annotated=true`表示独立可条件化量，其符号必须进入prompt但具体数值不得进入。

12.0.0的Release目录只包含运行时必需的五类内容：`README.md`、`dataset.json`、
`cases.jsonl`、`scenes/`和`views/`。迁移与独立
验证报告位于`provenance/releases/12.0.0/`。全局`masks.jsonl`未复制，因为当前运行时
不消费它；逐Case mask manifest和对象mask资产仍全部保留并锁定。

`assets/source_archives` 是指向 `provenance/source_archives` 的兼容链接，供冻结
Release继续解析原有相对路径；原始压缩包的权威物理位置属于provenance。

禁止向Dataset写入Baseline重采样视频、resize/pad首帧、embedding、latent、模型特征、
训练metadata或run输出。这些派生内容必须进入内容寻址cache或`runs_v2/`。完整契约见
[数据集文档](../docs/DATASET.md)，新增原始视频和XLSX时遵循
[导入规范](../docs/DATASET_INGESTION.md)。
