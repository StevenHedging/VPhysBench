# Datasets

`datasets/` 是仓库唯一的权威数据根。

- `_incoming/`：尚未验收、不得被 Benchmark 直接引用的导入材料；
- `physics_video/assets/`：按 scene/case 组织的不可变源资产与 canonical 资产；
- `physics_video/provenance/`：来源文档、导入记录与人工审核证据；
- `physics_video/releases/`：不可变的 Dataset metadata、View 和资产锁。

模型侧重采样、缩放、抽帧和特征文件必须写入 `cache/` 或 run 目录，不能回写这里。
