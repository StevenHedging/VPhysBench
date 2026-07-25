# Physics Video Dataset 3.0.0

该 release 在 2.0.0 的单摆、一维碰撞、自由落体基础上增加：

- 95 条标注完整并完成启动帧对齐的斜面下滑；
- 36 条标注完整的匀速圆周运动。

`cases.jsonl` 和 View B 覆盖全部 214 条 case。View A 使用显式
`coverage=subset`：旧三场景划分保持不变，新场景只选取能形成纯数值 ID 与纯环境
OOD1 对照的代表性 case；可靠重复和混合因素 case 仍保留在 Dataset 与 View B。

重建与验收：

```bash
conda run -n phybench python scripts/align_inclined_plane.py --propose --workers 6
# 检查 provenance/alignment/inclined_plane_start_v1/review/
conda run -n phybench python scripts/align_inclined_plane.py --accept-proposals
conda run -n phybench python scripts/align_inclined_plane.py --materialize --workers 4
python3 scripts/import_20260723_scenes.py
python3 scripts/build_dataset_asset_lock.py \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json
PYTHONPATH=src python3 -m physbench validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-asset-hashes
```
