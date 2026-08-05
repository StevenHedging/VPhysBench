# Physics Video Dataset 6.0.0

这是从5.1.0生成的纯元数据release。View A使用完整且互斥的`train/test`主划分；
碰撞scene基于全部有效Case重新执行分层、replicate-safe划分，其它scene保留5.1.0
train成员。ID/OOD/mixed是相对于`view_a.train`的test annotation，不再是Case字段或
顶层partition。View B继续覆盖全部Case用于direct evaluation。没有媒体字节被复制或修改。

旧平抛划分审计保留在`legacy_split_audit_5.1.0.json`；新的`split_audit.json`
描述当前View A。完整数据导入规范见`docs/DATASET_INGESTION.md`。
