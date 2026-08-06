# Datasets

`datasets/` 是仓库唯一权威数据根。当前正式入口：

```text
releases/12.0.0/dataset.json
```

目录职责：

- `assets/`：按scene和Case组织source、canonical媒体、mask、`caption.json`与
  `physics.json`；
- `provenance/`：导入来源、原始标注、原始压缩包，以及构建、迁移和验证证据；
- `releases/12.0.0/`：当前运行默认的精简Dataset快照，包含799个Case；
- `releases/`：只保留`12.0.0/`这一份活动运行快照；旧版本从Git历史追溯。

12.0.0的`cases.jsonl`是轻量索引，不重复保存caption或physics。每条记录通过
`assets.caption`和`assets.physics_annotation`指向Case目录中的`caption.json`与
`physics.json`。Loader严格读取两份成员文件，并物化出下游兼容的`case.text`和
`case.physics`。两份成员文件只保留Case/Scene身份和实际内容，不逐Case重复schema、
语言或annotation source。

运行时索引每条记录只含`case_id`、`scene_id`、`assets`、`appearance`和`temporal`。
逐Case来源、原始媒体定位、采集时序说明与alignment统一位于
`provenance/releases/12.0.0/cases.jsonl`；当前GT的唯一资产角色是`reference_video`。

12.0.0只保留正式物理量。标量quantity恰好包含`value`、`unit`和`symbol`；时序quantity
恰好包含`samples`、`time_unit`、`unit`和`symbol`，每个sample显式保存`time/value`。
所有正式数值统一为非负大小，方向只写入prompt；所有保留符号都必须进入prompt但具体
数值不得进入。六个scene的`physics`均按`objects.object_N`和`environment`分组，
`object_N`与首帧mask的矩阵顺序一致。推水瓶的`applied_force`属于`object_1`，保留XLSX
中的完整采样顺序与时间戳，不插值、不平滑、不去重。

12.0.0的Release目录只包含运行时必需的四类内容：`dataset.json`、`cases.jsonl`、
`scenes/`和`views/`。迁移与独立
验证报告位于`provenance/releases/12.0.0/`。全局`masks.jsonl`未复制，因为当前运行时
不消费它；逐Case mask manifest和对象mask资产仍全部保留。

`assets/source_archives` 是指向 `provenance/source_archives` 的兼容链接，供冻结
Release继续解析原有相对路径；原始压缩包的权威物理位置属于provenance。

禁止向Dataset写入Baseline重采样视频、resize/pad首帧、embedding、latent、模型特征、
训练metadata或run输出。这些派生内容必须进入内容寻址cache或`runs_v2/`。完整契约见
[数据集文档](../docs/DATASET.md)，新增原始视频和XLSX时遵循
[导入规范](../docs/DATASET_INGESTION.md)。
