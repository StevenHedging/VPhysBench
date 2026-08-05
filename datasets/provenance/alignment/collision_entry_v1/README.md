# 一维碰撞首帧对齐证据

本目录是 Dataset 3.0.0 中 `collision_1d` 的冻结 provenance。当前 32 个碰撞
case 均引用这里记录的 `collision_entry_v1` 对齐结果。

## 对齐契约

正式 `reference_video` 的第 0 帧必须完整显示最左侧入射球。对齐只删除入射球
尚未完整进入画面的前缀帧，并保持：

- 原始分辨率；
- 原始帧间时间间隔；
- 原始 `source_video` 字节不变；
- 选定源帧与 canonical 视频第 0 帧的解码像素一致。

源第 0 帧已满足契约时不裁切。其余视频从人工审定的解码帧开始，以 lossless HEVC
生成 canonical 视频。运行时资产位置由
`datasets/releases/3.0.0/dataset.json` 和 case 的 `assets` 字段解析，
不能从审计记录中的字符串路径推导。

## 冻结证据

| 文件或目录 | 内容 |
| --- | --- |
| `reviewed_frames.json` | 32 个 case 的审定起始帧 |
| `alignment_audit.jsonl` | 源/canonical probe、SHA-256、帧数与首帧像素校验 |
| `alignment_audit.csv` | 便于人工检查的核心审计字段 |
| `coarse/` | 入画区间时间线 |
| `fine/` | 起始帧附近的细粒度复核图 |
| `cases_before_alignment.jsonl` | 对齐变换的输入快照，仅用于证明变换来源 |

`alignment_audit.jsonl` 中所有 32 条记录都满足：

```text
frame_count_verified = true
frame0_identity_verified = true
review_status = visually_verified
```

审计记录中的 SHA-256 与当前 Dataset 3.0.0 的 `source_video`、`reference_video`
和 `first_frame` 一一对应。完整资产一致性由发布锁文件统一验证：

```bash
cd /root/Steven/physics_video_benchmark
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/3.0.0/dataset.json \
  --check-asset-hashes
```

## 实现边界

对齐算法保存在 `scripts/apply_collision_entry_alignment.py`。发布后的 canonical
资产和审计文件均为冻结输入；常规评估不得重新裁切、重编码或覆盖它们。若对齐契约
发生变化，应构建新的 Dataset release 和新的 provenance 版本，而不是原地修改
Dataset 3.0.0。
