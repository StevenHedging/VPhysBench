# Physics Video Dataset 8.0.0

当前 release 在 7.0.0 的 593 条 Case 上新增 65 条补充单摆和 141 条推水瓶，
共 799 条、6 个 scene。View A 对每个 scene 仅使用互斥的 `train/test`，全部 test
均为 ID，单个 scene 的 test 不超过 20 条；当前 release 不再构造 OOD/mixed 测试集。

补充单摆的 canonical 第 0 帧按 Dataset 语义定义为初始释放点，时间裁剪唯一目的为
去除人手。推水瓶视频按原字节保留，不做裁剪、剪辑、重采样或重编码。完整来源与
排除记录位于 `datasets/physics_video/provenance/`。
