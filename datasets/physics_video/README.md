# Physics Video Dataset

当前Dataset `physics_video_five_scene_v7`包含593个Case和1,617个锁定资产：

```text
assets/          权威 source/canonical 媒体
provenance/      来源、标注、导入和对齐证据
releases/1.0.0/…6.0.0/  仅用于解释既有结果的历史元数据快照
releases/7.0.0/  当前train/test descriptor、Case、View、scene和asset lock
```

运行时必须从`releases/7.0.0/dataset.json`加载，不能绕过descriptor直接拼接
`cases.jsonl` 与资产目录，也不能按目录版本号扫描或自动回退到历史release。历史快照
只有在复现一份明确记录了旧Dataset ID/digest的结果时才能显式使用。

Case schema 4.0将原始文本固定在`text.prompt`，将可信物理量固定在
`physics.<parameter>`（`annotated=true`）。Dataset 只声明事实；物理信息的使用方式由
各 Baseline 自己的 `input_policy` 与 adapter 决定。

当前release还为每条Case冻结：

```text
temporal.target_physical_duration_s
```

它表示从物理时刻0到canonical physics reference最后一帧的时间，计算为
`(frame_count - 1) / encoded_fps / encoded_to_physical_speed`。该标量只用于要求Baseline
尽量生成相同物理时长，不改变Dataset视频的FPS、帧数或字节。新增或更换canonical
reference后必须运行：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/freeze_target_physical_durations.py
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/freeze_target_physical_durations.py --check
```

发布流程：

1. 把新来源放入 `_incoming` 或明确的 source archive；
2. 校验原始标注与视频成员对应关系；
3. materialize canonical asset；
4. 生成 case、scene 和 View metadata；
5. 冻结并检查每条Case的目标物理时长；
6. 生成 `assets.lock.json`；
7. 运行 `validate-dataset --check-asset-hashes`；
8. 冻结 release digest。

主要数据维护脚本：

```text
scripts/align_inclined_plane.py
scripts/import_20260723_scenes.py
scripts/import_20260730_parabolic_collision.py
scripts/normalize_dataset_v51.py
```

详细字段、数量和划分见 [数据集文档](../../docs/DATASET.md)。
新数据的逐步导入流程见
[原始视频与XLSX标注导入规范](../../docs/DATASET_INGESTION.md)。
