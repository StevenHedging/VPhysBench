# Physics Video Dataset 5.1.0

该 release 从不可变的 5.0.0 派生，case ID 和媒体字节保持稳定。

- 使用用户确认的钢球规格统一碰撞和平抛标注；
- 由错误直径派生的光电门速度已按原遮挡时间重新计算；
- 碰撞 prompt 逐 case 描述球数、从左到右规格、方向和初速度；
- 平抛辅助装置量保留作 provenance，但不再作为可条件化物理输入；
- 平抛 View A 按球规格、发射高度和初速度成组划分，无 train/test_id
  物理 signature 重叠；
- 六个 scene 的每个 case 均使用简短的结构化物理目录名，背景不参与命名；
- 新资产路径全部是指向 5.0.0 冻结媒体的硬链接，不复制媒体 payload。

详见 `ball_spec_catalog.json`、`annotation_corrections.json`、
`asset_directory_mapping.json`、`split_audit.json` 和
`migration_audit.json`。
