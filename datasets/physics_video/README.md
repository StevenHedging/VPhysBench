# Physics Video Dataset

当前 Dataset `physics_video_five_scene_v3` 包含 214 个 case：

```text
assets/          权威 source/canonical 媒体
provenance/      来源、标注、导入和对齐证据
releases/3.0.0/  当前 descriptor、case、View、scene 和 asset lock
```

运行时必须从 `releases/3.0.0/dataset.json` 加载，不能绕过 descriptor 直接拼接
`cases.jsonl` 与资产目录。

发布流程：

1. 把新来源放入 `_incoming` 或明确的 source archive；
2. 校验原始标注与视频成员对应关系；
3. materialize canonical asset；
4. 生成 case、scene 和 View metadata；
5. 生成 `assets.lock.json`；
6. 运行 `validate-dataset --check-asset-hashes`；
7. 冻结 release digest。

斜面和圆周导入脚本：

```text
scripts/align_inclined_plane.py
scripts/import_20260723_scenes.py
```

详细字段、数量和划分见 [数据集文档](../../docs/DATASET.md)。
