# Datasets

`datasets/` 是仓库唯一权威数据根。当前正式入口：

```text
releases/8.0.0/dataset.json
```

目录职责：

- `assets/`：逐 scene/case 组织的 source 与 canonical 媒体；
- `provenance/`：导入来源、原始标注、原始压缩包和对齐审核；
- `releases/8.0.0/`：当前运行默认的 Dataset descriptor、Case schema 4.0、
  scene、View 和 asset lock；
- `releases/9.0.0/`：在8.0.0 Case事实上增加逐主体首帧mask的release；
- `releases/<older>/`：只用于解释既有结果的历史元数据；运行时禁止自动
  发现或回退。

`assets/source_archives` 是指向 `provenance/source_archives` 的兼容链接，供已冻结
release继续解析原有的相对路径；原始压缩包的权威物理位置属于provenance。

Case 的原始文本位于 `text.prompt`，结构化物理标注位于 `physics`。二者都是数据事实；
Task 不决定是否使用物理信息，Baseline 的 `input_policy` 与 adapter 才决定忽略、转换
或注入哪些已标注物理量。

禁止写入：

- Baseline 重采样视频；
- resize/pad 首帧；
- embedding、latent 或模型特征；
- 训练 metadata；
- run 输出。

这些派生内容必须写入内容寻址 cache 或 `runs_v2/`。完整契约见
[数据集文档](../docs/DATASET.md)。
新增原始视频和XLSX时必须同时遵循
[导入规范](../docs/DATASET_INGESTION.md)。
