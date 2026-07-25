# Physics Video dataset

本目录将物理视频的实际资产、来源证据和版本化 DatasetSnapshot 收拢在同一数据集合中。

```text
assets/          immutable source and canonical media
provenance/      import records, source documents, alignment reviews
releases/1.0.0/  v1 compatibility manifest and views
releases/2.0.0/  prompt-free v2 Dataset and assets.lock.json
releases/3.0.0/  five-scene Dataset, subset-capable View A, complete View B
```

发布流程：

1. 导入或审核资产；
2. 用场景导入器生成 native v2-contract release metadata；
3. 运行 `scripts/build_dataset_asset_lock.py --dataset <release>/dataset.json`；
4. 使用 `validate-dataset --check-asset-hashes` 完整验收。

3.0.0 由 2.0.0 扩展而来；新增数据的重建入口是
`scripts/align_inclined_plane.py` 和 `scripts/import_20260723_scenes.py`。
