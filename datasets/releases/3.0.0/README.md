# Physics Video Dataset 3.0.0

这是当前正式五场景 DatasetSnapshot：

```text
dataset_id: physics_video_five_scene_v3
cases:      214
assets:     468
```

文件：

- `dataset.json`：唯一加载入口；
- `release.json`：Dataset 与资产集合 digest；
- `cases.jsonl`：214 个 schema 2.0 case；
- `assets.lock.json`：468 个引用资产的锁；
- `scenes/*.json`：物理量、环境因素和 metric spec；
- `views/view_a.json`：训练主导的 ID/OOD1 对照；
- `views/view_b.json`：覆盖全部 case 的直接评测分组；
- `expansion_audit.json`：release 完整性审计，不作为运行输入。

View A 覆盖 187 个纯对照 case。未进入 View A 的 15 个斜面 case 和 12 个圆周 case
仍由 View B 覆盖；它们不会被误标为 ID 或 OOD1。

验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/3.0.0/dataset.json \
  --check-asset-hashes
```

数量、物理字段和划分规则见
[数据集文档](../../../../docs/DATASET.md)。
