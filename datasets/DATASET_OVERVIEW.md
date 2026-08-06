# Dataset 概览

当前且唯一活动的 Dataset 是 `physics_video_six_scene_v12`：

| 项目 | 当前值 |
| --- | --- |
| Release | `12.0.0` |
| Descriptor | `datasets/releases/12.0.0/dataset.json` |
| Case schema | `5.0` |
| Case 数 | 799 |
| Scene 数 | 6 |

六个Scene为`pendulum`、`collision_1d`、`inclined_plane_slide`、
`uniform_circular_motion`、`parabolic_motion`和`push_bottle`。View A将每个Scene划分为
train与较小的ID test；View B完整覆盖所有Case并提供确定性分组。当前两份官方五场景
Task暂不包含尚无专用Evaluator的`push_bottle`。

每条Case的资产目录包含唯一的`caption.json`和`physics.json`，分别由
`assets.caption`与`assets.physics_annotation`绑定。`cases.jsonl`不复制这两份内容；
Loader读取后提供`case.text`和`case.physics`。正式quantity固定为
`value/unit/symbol`三字段；数值表示非负大小，方向写入无数值caption。五个已分类Scene
按`physics.objects.object_N`和`physics.environment`组织，背景、颜色、视角、实验形式、
辅助测量和派生审计量均不进入结构化物理量。

`datasets/releases/`只保留`12.0.0/`这一份运行快照。V1–V11已失活，不允许扫描、自动
回退或作为当前运行输入；历史Dataset ID和迁移事实可从Git历史及
`datasets/provenance/`审计。

加载和结构验证：

```bash
PYTHONPATH=src python3 -m physbench validate-dataset \
  --dataset datasets/releases/12.0.0/dataset.json \
  --check-assets

PYTHONPATH=src:tests:. python3 scripts/validate_dataset_v12.py
```

详细契约见[数据集文档](../docs/DATASET.md)，新增视频与XLSX的处理流程见
[导入规范](../docs/DATASET_INGESTION.md)。
