# Physics Video Dataset 9.0.0

本 release 以 8.0.0 的 799 条 Case 为基础，为首帧物理主体增加实例级二值
mask。视频、文本、结构化物理信息、scene 定义和 View 划分保持不变。

每个完成的 Case 在其 `canonical/masks/` 子目录中包含：

- `01.npz`、`02.npz`……：模型读取的逐主体首帧 mask；每个文件只对应一个
  物理主体，其中 `masks` 数组布局为 `[1,H,W]`、`uint8`、值域为 `{0,1}`，
  并保存单元素 `mask_ids`、单元素 `object_ids` 和固定为 0 的 `frame_index`；
- `01.png`、`02.png`……：逐主体独立可视化 mask，尺寸与首帧一致，
  单通道 `uint8`，值域为 `{0,255}`；
- `manifest.json`：mask ID、画面主体、原物理对象 ID、关联物理字段、几何信息
  及生成证据；每个 instance 通过 `npz_asset` 指向自己的 NPZ 文件。

训练代码应以各 instance 的 `npz_asset` 为权威像素输入，并使用
`mask_id → object_id` 映射关联结构化物理信息。每个对象可独立读取，不需要
依赖 Case 内对象的数组下标。PNG 仅用于人工质检、编号核对和调试，不参与
训练读取。

编号按首帧矩阵扫描顺序，即从上到下逐行扫描、同一行从左到右。碰撞球的
物理标注本身即为从左到右；圆周运动若画面顺序与原 `object_1/object_2` 身份
不同，则 `mask_id` 保持画面顺序，并在 manifest 中显式记录对应的
`object_id`，不修改原物理标注。

Case 的 `assets.first_frame_subject_mask_npz_XX`、
`assets.first_frame_subject_mask_XX` 和 `assets.first_frame_mask_manifest` 是可由
现有 Dataset loader 锁定和校验的第四类数据资产。`masks.jsonl` 提供逐 Case 的
统一索引和完成状态，其 SHA-256 写入 `dataset.json`。生成报告与跳过原因分别保存在
`mask_generation_report.json` 和 `mask_release_audit.json`。

主体口径：碰撞场景掩码每个球；圆周运动掩码每个运动块；斜面只掩码滑块；
抛体只掩码小球；单摆只掩码摆球；推瓶只掩码瓶子。手、绳、斜面和实验支架
不作为物理主体。
