# 真实数据导入记录

## 原则

- 单摆在数据侧只做逐字节复制、首帧派生与标注。碰撞的原始 `reference.mov` 仍逐字节保留，但额外生成首帧对齐的无损派生视频。自由落体按明确约定在数据侧恢复 8 倍慢放，但仍不统一分辨率、不丢帧且不重编码。
- 所有原视频和目标文件均记录 SHA-256；导入器在哈希不一致时拒绝覆盖。
- 时间尺度是 case 级语义标注。官方自由落体 reference 已在数据侧恢复真实时间；空间适配和 24 fps 重采样仍由 baseline 在私有缓存中完成。
- 球规格文档中的尺寸是直径，manifest 中的 `radius` 均为直径除以 2。

## 已导入场景

### 单摆

- 35 条真实时间尺度视频：可信 R1 16 条、用于替换污染旧 R2 的新 R2 19 条。
- 文件名首项为支点到球心总摆长；球半径固定 0.010 m，绳长为总摆长减去球半径。
- R1 总摆长为 0.210、0.240、0.280、0.305 m；初始角度为 5°、10°、20°、25°。
- 新 R2 总摆长为 0.110、0.130、0.155、0.180 m；通常包含 5°、10°、20°、25°、30°，其中 `18_30.mp4` 未采集。
- `11_25.mp4` 原始帧率为 30 fps，其余新 R2 基本为 120 fps；所有文件均保持原规格，WAN 按时间戳适配。
- View A：每个已采集总摆长的 20° case 为 `test_id`，其余为 `train`。计数为 train 27、test_id 8；目前没有实拍单摆 OOD1 continuation。

### 一维对心碰撞

- 32 条真实时间尺度视频，原始 HEVC/MOV 以 `reference.mov` 保持不变；正式训练/评测资产为 `reference_aligned.mp4`。
- 对每条视频先做粗时间线定位，再以 2 帧步长检查，最后逐帧检查候选帧及其相邻帧；起始帧定义为“最左侧入射球已完整进入画面的最早人工确认帧”。
- 对齐严格按解码帧号裁切，重置起始 PTS，但保留后续每帧的原始时间戳间隔；不重采样 fps、不改分辨率、不丢弃尾帧。需要裁切的 case 用 libx265 lossless 重编码，源第 0 帧已合格的 case 仅重封装。
- 每条 case 的 `first_frame.png` 来自对齐视频第 0 帧；manifest 同时记录 `assets.source_video`、正式 reference 与 `alignment` 元数据。逐条源/输出哈希、帧数和首帧像素一致性见 `data/alignment_audits/collision_entry_v1/alignment_audit.jsonl`。
- `collision_r2_small_steel_medium_steel_large_steel_v07389` 在源第 0 帧时来球已经完整入画，且源文件没有更早帧，因此从第 0 帧开始，这是唯一的 source-frame-zero 例外。
- 文件名三字分别表示沿运动方向排列的球 1、2、3；小数为球 1 初速度（m/s）。
- `大/中/小/波` 分别映射为大钢球、中钢球、小钢球、玻璃弹珠。
- View A：同质钢球组合为 ID；每组速度排序后的中位样本为 `test_id`，其余为 `train`。异质组合为 `test_ood1`，标注 `collision_pair`，含玻璃球时同时标注 `ball_material`。
- 计数：train 11、test_id 3、test_ood1 18。

### 自由落体

- 11 条原始 8 倍慢放实拍 MP4；首帧为静止释放，文件名高度为 60/80/100 cm，`S_60` 未采集。
- 每个 case 的 `source_slowmo.mp4` 逐字节保留原始 30 fps 慢放源；`reference.mp4` 使用两遍无重编码 remux，把 PTS/DTS 精确除以 8 并修正容器尾部时长。
- 恢复后仍保留全部 HEVC 帧和 1080×1920 分辨率，帧率为 240 fps，时长严格为原来的 1/8。case 已是实际物理时间，因此标注 `encoded_to_physical_speed=1.0` 和 `time_scale=real_time`。
- View A：60/100 cm 为 train，80 cm 为 test_id；S 球只有 100 cm train 和 80 cm test_id。
- 计数：train 7、test_id 4。

60 cm reference 约 0.33–0.35 秒，对应 5 个合法的 24 fps WAN 帧；80/100 cm 通常对应 9 帧。WAN 只做常规采样，不再执行 8 倍加速，也不会循环、减速或冻结尾帧。

## 文件与划分

```text
data/
├── assets/
│   ├── collision_1d/<case_id>/{reference.mov,reference_aligned.mp4,first_frame_source.png,first_frame.png}
│   ├── free_fall/<case_id>/{source_slowmo.mp4,reference.mp4,first_frame.png}
│   └── pendulum/<case_id>/{reference.mp4,first_frame.png}
├── manifests/
│   ├── cases.jsonl
│   └── import_audit.jsonl
├── source_docs/
│   ├── collision_1d_ball_spec.txt
│   ├── free_fall_ball_spec.txt
│   ├── pendulum_annotations.json
│   └── normalized_ball_specs.json
└── splits/
    ├── view_a.json
    └── view_b_seed42_g5.json
```

碰撞对齐的可视化复核材料位于 `data/alignment_audits/collision_entry_v1/`：`coarse/`、`fine/` 和 `final/` 分别保存粗筛、边界细查和相邻帧终审；`reviewed_frames.json` 是冻结的起始帧映射。

重建命令：

```bash
python scripts/import_real_dataset.py --scenes collision_1d free_fall
python scripts/apply_collision_entry_alignment.py --workers 8 --threads-per-job 12 --overwrite
python scripts/import_real_dataset.py --scenes pendulum --confirm-pendulum-r2-real-time
PYTHONPATH=src python -m physbench validate \
  --manifest data/manifests/cases.jsonl --check-assets
```

WAN 适配 dry-run：

```bash
PYTHONPATH=src python -m physbench run \
  --task configs/tasks/view_a_three_scene_finetune.json \
  --baseline configs/baselines/wan22_ti2v_5b_lora_task1.json \
  --manifest data/manifests/cases.jsonl \
  --split data/splits/view_a.json \
  --output-root runs/import_validation
```
