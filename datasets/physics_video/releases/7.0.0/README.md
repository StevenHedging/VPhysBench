# Physics Video Dataset 7.0.0

该 release 从冻结的6.0.0派生，移除自由落体scene及其11个Case，形成五场景Dataset。
其余593个Case、View A train/test成员身份、test ID/OOD/mixed annotation与View B
分组身份均保持不变。资产锁只保留仍被五场景引用的文件；自由落体媒体不属于本release。

`split_audit.json`记录成员身份保持关系，`migration_audit.json`记录删除范围与新identity。
完整数据导入规范见`docs/DATASET_INGESTION.md`。

## 2026-08-03 原地文本/媒体修复

本release未新增版本号，直接修复逐视频审计发现的两类对齐问题：

- 185个异径球碰撞Case不再使用不严格的`central`/`head-on`措辞，改为中性的
  碰撞过程描述；
- `parabolic_img_0539`使用7.0.0专用的240 fps真实物理时标视频和对应首帧，
  79帧完整保留，`encoded_to_physical_speed`改为1.0。

原30 fps慢动作媒体仍由5.1.0和6.0.0引用，未被覆盖。修复的Case清单、前后
prompt、媒体探测信息、SHA-256和新Dataset digest见
`text_video_alignment_repair.json`。
