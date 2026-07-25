# Datasets

`datasets/` 是仓库唯一权威数据根。当前正式入口：

```text
physics_video/releases/3.0.0/dataset.json
```

目录职责：

- `physics_video/assets/`：逐 scene/case 组织的 source 与 canonical 媒体；
- `physics_video/provenance/`：导入来源、原始标注和对齐审核；
- `physics_video/releases/3.0.0/`：Dataset descriptor、cases、scene、View 和 asset lock。

禁止写入：

- Baseline 重采样视频；
- resize/pad 首帧；
- embedding、latent 或模型特征；
- 训练 metadata；
- run 输出。

这些派生内容必须写入内容寻址 cache 或 `runs_v2/`。完整契约见
[数据集文档](../docs/DATASET.md)。
