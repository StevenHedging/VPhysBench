# Physics Video Dataset 5.1.0

该 release 从不可变的 5.0.0 派生；被保留的单摆 parent case 及媒体保持稳定。

- 使用用户确认的钢球规格统一碰撞和平抛标注；
- 由错误直径派生的光电门速度已按原遮挡时间重新计算；
- 文本只描述物理过程，不包含具体物理数值、背景颜色或裁剪提示；
- 移除 5 条首帧与参考视频不匹配的合成单摆 OOD case，保留其真实 parent case；
- canonical 视频保留裁剪窗口内的全部源帧和源 FPS，模型时序适配只在基线侧完成；
- 平抛辅助装置量保留作 provenance，但不再作为可条件化物理输入；
- 平抛 View A 按球规格、发射高度和初速度成组划分，无 train/test_id
  物理 signature 重叠；
- 六个 scene 的每个 case 均使用简短的结构化物理目录名，背景不参与命名；
- 当前资产使用不带版本前缀的描述性目录名；
- 395 条需裁剪的平抛/补充碰撞视频仅重编码必要的事件窗口与空间变换，
  不改变窗口内帧数和源 FPS；11 条自由落体 canonical 与慢动作源文件字节相同。

详见 `ball_spec_catalog.json`、`annotation_corrections.json`、
`asset_directory_mapping.json`、`split_audit.json` 和
`migration_audit.json`。
