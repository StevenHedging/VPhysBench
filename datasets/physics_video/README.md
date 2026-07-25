# Physics Video dataset

本目录将物理视频的实际资产、来源证据和版本化 DatasetSnapshot 收拢在同一数据集合中。

```text
assets/          immutable source and canonical media
provenance/      import records, source documents, alignment reviews
releases/1.0.0/  v1 compatibility manifest and views
releases/2.0.0/  prompt-free v2 Dataset and assets.lock.json
```

发布流程：

1. 导入或审核资产；
2. 更新 v1 兼容清单（如仍需要）；
3. 运行 `scripts/migrate_dataset_v2.py --force`；
4. 运行 `scripts/build_dataset_asset_lock.py`；
5. 使用 `validate-dataset --check-asset-hashes` 完整验收。
