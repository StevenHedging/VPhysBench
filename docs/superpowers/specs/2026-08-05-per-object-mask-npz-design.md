# Physics Video Dataset 9.0.0 逐对象 NPZ 设计

## 目标

直接原地修订 Physics Video Dataset 9.0.0，把每个 Case 的汇总
`canonical/masks/masks.npz` 拆成逐物理主体的 `01.npz`、`02.npz` 等文件。
每个 NPZ 只表示一个主体，同时保持现有三维模型接口 `masks=[1,H,W]`。逐主体
PNG 继续作为 `{0,255}` 可视化资产。

本设计取代此前 9.0.0 中“每个 Case 一个 `[O,H,W]` NPZ”的存储契约。

## 范围与不变量

- 处理现有 797 个完整 Case，共 1,205 个主体。
- 删除 797 个旧 `masks.npz`，创建 1,205 个逐主体 NPZ。
- 两个现有跳过 Case 继续保持跳过，不创建 Mask 目录或 NPZ。
- 不重新运行 SAM2，不改变 Mask 非零像素、编号、几何、画面顺序或物理对象映射。
- 不改变视频、首帧、文本、结构化物理信息、Scene 或 View。
- 直接更新 9.0.0，并重新生成资产锁、发布 digest、审计和验证报告。

## 每个 Case 的目录结构

```text
canonical/masks/
├── 01.npz
├── 01.png
├── 02.npz
├── 02.png
└── manifest.json
```

单主体 Case 只包含 `01.npz` 和 `01.png`。三主体 Case 继续增加 `03.npz` 和
`03.png`。目录中不得再存在 `masks.npz`。

## 单对象 NPZ 契约

每个 `XX.npz` 使用 `numpy.savez_compressed` 写入以下四个且仅以下四个数组：

| 字段 | dtype / shape | 约束 |
|---|---|---|
| `masks` | `uint8 [1,H,W]` | 值域严格为 `{0,1}`，唯一 plane 属于当前主体 |
| `mask_ids` | Unicode `[1]` | 与文件名一致，例如 `01.npz → ["01"]` |
| `object_ids` | Unicode `[1]` | 与当前主体的结构化物理对象 ID 一致 |
| `frame_index` | `int64` scalar | 固定为 `0` |

文件必须能由 `numpy.load(path, allow_pickle=False)` 读取，不得包含 object dtype
或 pickle 数据。

## PNG 契约

`XX.png` 保持单通道 `uint8 [H,W]`，背景为 0、主体为 255。其二值化结果必须
与同编号 `XX.npz` 中的 `masks[0]` 逐像素相等。PNG 只用于人工检查，训练代码
读取对应 NPZ。

## Manifest 与索引

`manifest.json` 的 `schema_version` 升为 `"1.2"`，模型存储声明为：

```json
{
  "model": {
    "asset_pattern": "assets/<scene>/<case>/canonical/masks/{mask_id}.npz",
    "array_key": "masks",
    "layout": "1HW",
    "dtype": "uint8",
    "values": [0, 1]
  },
  "visualization": {
    "asset_pattern": "assets/<scene>/<case>/canonical/masks/{mask_id}.png",
    "dtype": "uint8",
    "values": [0, 255]
  }
}
```

每个 `instances[k]`：

- 保留 `mask_id`、`object_id`、`physics_keys`、PNG `asset`、几何和 SAM2 证据；
- 新增直接指向同编号文件的 `npz_asset`；
- 删除只适用于汇总数组的 `npz_index`。

统一发布索引同步调整：

- `cases.jsonl` 删除 `assets.first_frame_masks_npz`；
- 每个完整 Case 新增 `assets.first_frame_subject_mask_npz_01`、
  `assets.first_frame_subject_mask_npz_02` 等逐主体角色；跳过 Case 对应角色为
  `null`；
- `masks.jsonl` 的记录和 `dataset.json.mask_annotations.schema_version` 升为
  `"1.2"`；删除记录级汇总 `npz_asset`，以 instance 的 `npz_asset` 为唯一索引；
- `assets.lock.json` 锁定全部 1,205 个逐主体 NPZ。

完成后的资产总数为 5,239：原 4,034 个 PNG-only 9.0.0 资产加 1,205 个逐对象
NPZ。旧 797 个汇总 NPZ 不得继续出现在资产树或资产锁中。

## 迁移顺序与失败处理

迁移程序先只读预检全部完整 Case。尚未迁移的 Case 必须存在 `[O,H,W]` 的旧
`masks.npz`，每个 plane 必须与对应 PNG 二值化结果、`mask_id`、`object_id`、
面积和几何一致。已经迁移的 Case 可以没有旧汇总文件，但必须具有一套完整且通过
相同交叉校验的逐对象 NPZ。任何 Case 失败时整批停止，且不写入或删除资产。

预检全部通过后，按 Case 执行：

1. 从旧汇总 NPZ 取第 k 个 plane；
2. 原子写入同编号 `XX.npz`，内部 shape 为 `[1,H,W]`；
3. 回读并验证四个字段、shape、dtype、值域、ID 与 PNG 像素；
4. 原子更新 manifest；
5. 当前 Case 的所有逐对象文件均验证成功后，删除旧 `masks.npz`。

该删除是可逆的：按 `mask_id` 顺序拼接全部 `XX.npz.masks[0]` 可无损重建原
`[O,H,W]` 数组。迁移脚本必须幂等，已迁移 Case 可在没有旧汇总文件时再次预检
和物化。

生成脚本同步改为直接写逐对象 NPZ，不再创建汇总文件。

## 验证标准

独立验证器必须证明：

1. 完整 Case 共存在 1,205 个 `XX.npz`，且不存在任何 `masks.npz`。
2. 每个 NPZ 的键、`[1,H,W]` shape、dtype、值域、单元素 ID 和 frame index
   符合契约。
3. 每个 `XX.npz.masks[0]` 与 `XX.png > 0` 完全相等。
4. Case、`masks.jsonl`、manifest 和资产锁中的逐对象路径一一一致。
5. `mask_id → npz_asset → object_id → physics_keys` 映射完整且无重复。
6. 原面积、包围盒、质心、多实例互斥和画面编号顺序保持不变。
7. 两个跳过 Case 不暴露 manifest、PNG 或 NPZ。
8. 8.0.0 的 2,032 个原资产哈希保持不变，9.0.0 的 5,239 个资产 SHA-256
   全部通过。

## 交付

提交存储模块、迁移脚本、SAM2 生成器、发布构建器、验证器、测试、README、
9.0.0 索引、资产锁和报告。实际 NPZ/PNG/manifest 保存在本机被 Git 忽略的
Case 资产目录中，并由 9.0.0 的资产锁封存哈希。
