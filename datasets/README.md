# Datasets

`datasets/` 是仓库唯一权威数据根。当前正式入口：

```text
physics_video/releases/7.0.0/dataset.json
```

目录职责：

- `physics_video/assets/`：逐 scene/case 组织的 source 与 canonical 媒体；
- `physics_video/provenance/`：导入来源、原始标注和对齐审核；
- `physics_video/releases/7.0.0/`：当前Dataset descriptor、Case schema 4.0、
  scene、View 和 asset lock；
- `physics_video/releases/<older>/`：只用于解释既有结果的历史元数据；运行时禁止自动
  发现或回退。

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
