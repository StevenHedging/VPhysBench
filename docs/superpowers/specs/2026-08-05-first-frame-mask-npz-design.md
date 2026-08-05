# Physics Video Dataset 9.0.0 首帧 Mask 双格式设计

## 目标

直接原地更新 Physics Video Dataset 9.0.0，为每个已完成 Mask 标注的 Case
增加模型可直接读取的 `masks.npz`，并将逐主体 PNG 调整为便于人工检查的
`{0,255}` 可视化图。NPZ 和 PNG 必须表示完全相同的首帧主体区域，且不得重新
运行 SAM2 或改变主体编号、几何形状和物理对象对应关系。

## 范围与不变量

- 处理现有 797 个 `status="complete"` 的 Case，共 1,205 个主体 Mask。
- 两个现有 `status="skipped"` 的 Case 继续保持跳过，不创建虚假的 Mask 资产。
- 不改变 799 个 Case 的身份、顺序、视频、首帧、文本、结构化物理信息、Scene
  定义或 View 划分。
- 直接更新 9.0.0，不新建 9.0.1；因此 9.0.0 的资产锁、发布 digest 和相关报告
  将随新格式重新生成。
- 现有二值 Mask 的非零像素集合是权威来源；转换过程不得重新推理或修正分割。

## 每个 Case 的目录结构

```text
canonical/masks/
├── masks.npz
├── 01.png
├── 02.png
└── manifest.json
```

主体数量为一时只存在 `01.png`；主体数量大于二时继续按 `03.png` 等顺序扩展。

## NPZ 模型资产

`masks.npz` 使用不依赖 pickle 的 NumPy 数组，包含且只包含以下字段：

| 字段 | dtype / shape | 含义 |
|---|---|---|
| `masks` | `uint8 [O,H,W]` | 逐主体首帧 Mask，值域严格为 `{0,1}` |
| `mask_ids` | Unicode `[O]` | `"01"`, `"02"` 等连续编号 |
| `object_ids` | Unicode `[O]` | 与结构化物理标注关联的对象 ID |
| `frame_index` | `int64` scalar | 固定为 `0` |

`O` 是该 Case 首帧中的物理主体数，`H,W` 与 `first_frame.png` 相同。
`masks[k]`、`mask_ids[k]` 和 `object_ids[k]` 必须一一对应。读取时使用
`numpy.load(path, allow_pickle=False)`。

## PNG 可视化资产

每张 `XX.png` 仍是单通道 `uint8 [H,W]`，但值域改为 `{0,255}`：背景为 0，
主体为 255。转换仅执行下式：

```text
png = npz.masks[k] * 255
```

PNG 不作为训练时的权威输入。它用于人工质检、编号核对和调试，因此可以被普通
图片查看器直接辨认。

## Manifest 与发布索引

每个 `manifest.json` 的 `schema_version` 升为 `"1.1"`，分别描述：

- NPZ 模型资产的相对路径、数组键、`[O,H,W]` 布局、dtype 和 `{0,1}` 值域；
- PNG 可视化资产的 dtype 和 `{0,255}` 值域；
- 每个 instance 的 `npz_index`、`mask_id`、`object_id`、PNG 路径、物理字段、
  面积、包围盒、质心和原 SAM2 生成证据。

9.0.0 的统一索引同步更新：

- `cases.jsonl`：完整 Case 的 `assets.first_frame_masks_npz` 指向
  `canonical/masks/masks.npz`；跳过 Case 为 `null`。
- `masks.jsonl`：记录的 `schema_version` 升为 `"1.1"`、增加 `npz_asset`，并
  保持 instances 与 Case manifest 一致；`dataset.json` 中对应的
  `mask_annotations.schema_version` 同步升为 `"1.1"`。
- `assets.lock.json`：锁定 797 个新增 NPZ、1,205 张重写后的 PNG 和更新后的
  manifest。
- `dataset.json`、`release.json`、验证报告和审计报告按更新后的资产重新计算。

## 转换与可重复生成

转换程序先完整读取并验证一个 Case 的旧 PNG：文件必须存在、单通道、
`uint8`、值域为 `{0,1}` 或已转换后的 `{0,255}`、非空、尺寸一致且编号连续。
程序把所有 PNG 规范化为 `{0,1}` 并按编号堆叠，再写 NPZ、可视化 PNG 和
manifest。所有单文件写入先落到同目录临时文件，再原子替换目标，以避免中途中断
留下半写文件。

转换应幂等：对已经符合新契约的 Case 再运行一次，不改变数组语义、索引或发布
元数据。生成脚本也同步改为直接产出双格式，防止以后重新生成时退回旧格式。

## 错误处理

出现以下任一情况时，不对该 Case 执行替换，并让整批任务失败：

- PNG 缺失、编号不连续、为空或包含契约外像素值；
- PNG 尺寸与首帧不一致；
- PNG 数量与 manifest/物理主体数量不一致；
- `mask_id`、`object_id` 或 instance 顺序不一致；
- 多主体 Mask 重复或相互重叠；
- 写入后的 NPZ 与 PNG 二值化结果不一致。

不把新失败静默降级成 `skipped`，避免掩盖已有 797 个标注的损坏。

## 验证标准

自动化测试和全量发布验证必须证明：

1. 797 个 NPZ 全部存在，键集合、dtype、shape、值域和 `allow_pickle=False`
   读取符合契约。
2. 1,205 张 PNG 全部是 `{0,255}`，且 `(png > 0)` 与对应 `masks[k]` 完全相等。
3. NPZ 顺序与 `mask_id → object_id → physics_keys` 映射一致。
4. 面积、包围盒、质心、多实例不重叠以及画面编号顺序保持不变。
5. 两个跳过 Case 不暴露 NPZ、manifest 或主体 PNG。
6. 8.0.0 的 2,032 个原资产哈希保持不变；9.0.0 的全部资产 SHA-256 校验通过。
7. 除新增或更新的 Mask 资产字段外，9.0.0 的 Case 内容与 8.0.0 完全相同。

## 交付结果

完成后提交转换/生成代码、测试、更新后的 9.0.0 发布元数据和验证报告。实际
PNG/NPZ/manifest 位于各 Case 的 `canonical/masks/` 目录，并由
`assets.lock.json` 记录哈希；项目现有忽略规则下的资产文件仍保留在本机资产树。
