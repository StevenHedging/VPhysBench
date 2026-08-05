# Physics Video Dataset 4.0.0

这是正式五场景 DatasetSnapshot 的 case-owned text release：

```text
dataset_id: physics_video_five_scene_v4
cases:      214
assets:     468
```

相对 3.0.0，本 release 不改 case 集合、场景、视图或资产，只进行数据契约迁移：

- 每个 case 升级到 schema 3.0；
- `text.prompt` 固化自旧 generic profile 的场景通用描述，其完整迁移快照保存在
  `scripts/migrate_dataset_v4.py`；
- 结构化 `physics` 与原始文本并列保存，由 Baseline 自行决定是否以及如何使用；
- `assets.lock.json` 仅更新 dataset identity，`files` 保持不变。

文件：

- `dataset.json`：唯一加载入口；
- `release.json`：loader-compatible Dataset 与资产集合 digest；
- `cases.jsonl`：带原始文本、媒体与结构化物理标注的 case；
- `migration_audit.json`：3.0.0 到 4.0.0 的迁移审计；
- `expansion_audit.json`、`scenes/`、`views/`：逐字节复制自 3.0.0。

验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/migrate_dataset_v4.py --force
```
