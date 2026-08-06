# Dataset、Case 与划分

## 1. 权威入口

当前 DatasetSnapshot：

```text
dataset_id:     physics_video_six_scene_v12
release:        12.0.0
descriptor:     datasets/releases/12.0.0/dataset.json
cases:          799
```

`datasets/` 是唯一权威数据根。Baseline、cache 和 run 不能回写或覆盖这里的资产。
新增原始视频和XLSX的完整导入流程见
[DATASET_INGESTION.md](DATASET_INGESTION.md)。

## 2. 目录结构

```text
datasets/
├── assets/
│   ├── pendulum/
│   ├── collision_1d/
│   ├── inclined_plane_slide/
│   ├── uniform_circular_motion/
│   ├── parabolic_motion/
│   ├── push_bottle/
│   └── source_archives -> ../provenance/source_archives
├── provenance/
│   ├── source_archives/
│   ├── imports/
│   ├── source_docs/
│   └── releases/12.0.0/         # 逐Case来源及迁移、验证记录
└── releases/
    └── 12.0.0/                  # 唯一活动运行快照
        ├── dataset.json          # 当前运行默认入口
        ├── cases.jsonl           # 轻量Case索引
        ├── scenes/
        └── views/
```

原始压缩包按字节保存在 `provenance/source_archives/`；冻结release仍可通过兼容路径
`assets/source_archives/` 访问。Case 的
`provenance/releases/12.0.0/cases.jsonl`记录逐Case的archive/member、原始媒体和时间
对齐信息。供运行和评估使用的canonical
视频、首帧等按 scene/描述性物理目录存放，路径稳定。

## 3. Case schema 5.0

`cases.jsonl`每行只保存Case身份、运行时资产路径、非物理元数据和时序，不重复保存caption、
physics或来源审核信息。Loader通过资产引用读取Case-local成员，然后返回完整的schema 5.0 Case。
物化后的核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | 全 Dataset 唯一稳定 ID |
| `scene_id` | 六个正式 scene 之一 |
| `text` | Loader从`caption.json`物化的唯一`prompt` |
| `assets` | `caption`、首帧、`physics_annotation`、`reference_video`及可选mask manifest |
| `physics` | 按物理主体与环境分组的正式结构化物理量 |
| `appearance` | 非结构化物理量的情景、外观、环境、实验形式与采集信息 |
| `temporal` | 仅保存运行时所需的`encoded_to_physical_speed` |

来源、原始成员定位、采集时序说明和时间对齐审核不进入运行时Case，统一保存在
`datasets/provenance/releases/12.0.0/cases.jsonl`。当前每条Case都有同试次GT，
`assets.reference_video`是唯一reference角色。

Case 不包含 Task partition、模型分辨率、runner 参数或模型原生输入。当前View A只包含
train和ID test；历史release中的OOD字段仅用于复现旧结果，不能重新带入当前划分。

`appearance` 沿用历史字段名，但它的语义比视觉外观更宽：背景、颜色、材质、机位、
采集批次，以及球数、初始运动球数、碰撞结构等非数值实验形式都放在这里。它们可以
作为分层和覆盖检查字段，但不能混入 `physics`。

## 4. 原始 prompt

原始文本的唯一存储位置是：

```text
assets/<scene_id>/<case_directory>/caption.json
case.assets.caption
```

文件结构：

```json
{
  "case_id": "<case_id>",
  "scene_id": "pendulum",
  "caption": "A pendulum bob of mass m and radius r is released ..."
}
```

Loader校验身份后，将`caption`映射为兼容运行时字段`case.text.prompt`。Baseline和
Evaluator只消费物化后的Case，不直接解析该文件。

该prompt只描述可见物理过程和独立物理量的数学符号，不包含具体数值、单位、背景、颜色、
视角或采集提示。碰撞scene
逐case明确主体数量、初始运动/静止角色和运动方向，避免用一个模板模糊不同情景；
12.0.0中的100条单摆统一采用“首帧静止释放并绕固定点往复摆动”的语义。所有Baseline
都以它为文本源；是否原样使用、
追加结构化物理文本或转换成其它模型表示，由 Baseline 的 `input_policy` 与 adapter
决定。Task 不生成或选择 prompt。

## 5. 结构化物理标注

结构化物理量的唯一存储位置是：

```text
assets/<scene_id>/<case_directory>/physics.json
case.assets.physics_annotation
```

文件必须恰好包含`case_id`、`scene_id`和`physics`。Loader即使在
`check_assets=False`时也会读取它，校验Case/Scene身份，并将`physics`物化为下游兼容的
`case.physics`运行时API。

六个Scene的`physics`都恰好包含`objects`和`environment`。对象编号使用
`object_1`、`object_2`……，与首帧mask从左到右、从上到下的矩阵顺序一致：

```json
{
  "objects": {
    "object_1": {
      "mass": {"value": 0.03313, "unit": "kg", "symbol": "m_1"}
    }
  },
  "environment": {
    "gravity_acceleration": {"value": 9.8, "unit": "m/s^2", "symbol": "g"}
  }
}
```

标量quantity恰好包含：

```json
{
  "value": 54.55,
  "unit": "deg/s",
  "symbol": "ω"
}
```

时序quantity恰好包含：

```json
{
  "samples": [
    {"time": 0.1, "value": 0.16},
    {"time": 0.2, "value": 0.27}
  ],
  "time_unit": "s",
  "unit": "N",
  "symbol": "F(t)"
}
```

时序sample必须保留来源中的显式时间戳与顺序；允许时间戳重复或相邻间隔不等，不得为
填平间隙而插值，也不得平滑、重采样或去重。正式时序值同样必须有限且非负。

活动Dataset中的quantity全部是独立、可信、可作为模型输入的正式物理量；`symbol`必须
原样出现在`case.text.prompt`中，但Dataset prompt不得包含数值。派生量、校准量、辅助
装置量、重复别名和背景信息只进入provenance，不进入`physics`。所有`value`必须有限且
非负；velocity字段存速度大小，向左/向右、静止等方向语义由prompt承担。

主要物理量：

- 单摆：对象含摆球半径、可用时的摆球质量与初始角度；环境含绳长；
- 一维碰撞：球质量、半径、初速度；
- 斜面下滑：对象含物块质量与长度；环境仅含斜面角度与重力加速度；
- 平抛：对象含质量、半径、初始水平速度与发射高度；环境为空；
- 匀速圆周运动：对象含各自轨道半径；环境含角速度；
- 推水瓶：对象含水瓶质量、高度和完整外力时间序列`F(t)`；环境为空。均值和峰值是可从
  序列派生的摘要，不再作为正式物理量。

## 6. 资产角色

典型 Case：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
├── caption.json            # 唯一的当前Case-local文本描述
├── physics.json            # 唯一的当前Case-local符号化标注
├── source/                 # 可选原始字节
└── canonical/
    ├── reference.mp4       # evaluator 使用的物理参考
    └── first_frame.png     # I2V 输入
```

规则：

- Case 中的路径相对 `dataset.asset_root`；
- 当前Release不维护资产锁或文件哈希；
- `cases.jsonl`只引用`caption.json`和`physics.json`，不复制二者内容；
- 12.0.0 的目录名只编码 scene 的主要结构化物理量与唯一身份后缀，不使用背景、
  颜色或采集环境；
- I2V 使用显式 `assets.first_frame`，不在运行时从 GT 临时补首帧；
- `reference_video`只供训练监督或evaluator使用，不交给评测阶段的生成driver；
- source、alignment与来源定位只保存在provenance，不进入运行时Case；
- 尺寸、FPS、帧数、抽帧和特征派生物只能进入 immutable cache 或 run。

斜面下滑的 canonical 视频已经过启动时刻清洗。新增平抛和碰撞视频均从球刚穿过
光电门且所有球完整可见的首帧开始，画面已经裁去光电门与线缆；平抛另裁为合适的竖屏
画幅。首尾帧视觉复核结论、裁剪框和源帧范围均随 Dataset provenance 冻结。生成阶段
不再重复裁剪。

补充单摆裁掉了带人手的源视频前缀，但canonical第0帧在Dataset语义中就是初始释放点；
不能在prompt或Case alignment中称为“释放后的对侧转折点”。推水瓶无需清洗，canonical
reference与源MOV逐字节一致，首帧由该reference第0帧解码得到。

V2V 必须另外登记独立输入视频资产，例如 `assets.input_video`。其中
`conditioning_video` 只是 adapter 对该输入媒体 channel 的角色名，不表示 Task
层的物理信息分组；reference和source video均禁止充当V2V输入。
当前12.0.0 Release没有正式`assets.input_video`，因此V2V Bundle只是接口能力，
不能直接运行官方数据。

## 7. View A：finetune_eval

View A schema 3.0 只使用两个互斥主划分：`train` 和 `test`。两者完整覆盖799条Case。
当前release不再构造OOD或mixed测试集；所有test都是训练支持域内的独立ID试次，每个
scene的test不超过20条。兼容schema中的`generalization_regime`固定为`id`，两个factor
列表固定为空。

当前数量：

| scene | 全部 | train | ID test |
| --- | ---: | ---: | ---: |
| pendulum | 100 | 80 | 20 |
| collision_1d | 330 | 310 | 20 |
| inclined_plane_slide | 95 | 80 | 15 |
| uniform_circular_motion | 36 | 30 | 6 |
| parabolic_motion | 97 | 82 | 15 |
| push_bottle | 141 | 127 | 14 |
| **合计** | **799** | **709** | **90** |

碰撞test覆盖全部13个“碰撞结构×球材质×球规格序列”stratum，并把人工审核近重复簇、
重复组件和完全相同结构化物理signature作为不可拆分组件；速度范围端点强制留在train。
平抛按完整物理signature成组抽取。补充单摆按摆长—角度分层抽取，推水瓶的7个质量组
各抽2条，其余均进入train。

## 8. View B：direct_eval

View B：

- 完整覆盖 799 个 case；
- scene 内按冻结 seed 确定性分组；
- group 尽量均衡；
- group 是报告/抽样维度，不表达模型的物理使用方式，也不等同于 ID/OOD 层级。

## 9. Release 12.0.0 的符号化物理标注

12.0.0将V11的完整已校正内容固化为唯一当前标注，是官方默认：

- 保持799条Case、全部View成员、媒体角色、媒体字节和来源事实不变；
- 每条Case只保留一个无版本冗余字段的`physics.json`，Case由Dataset descriptor声明schema 5.0；
- 每个quantity新增稳定`symbol`，独立量符号进入英文prompt，具体数值和单位不进入；
- 将494个碰撞有符号速度值转换为非负速度大小，方向明确写在prompt中；
- 删除全部派生、校准、辅助或重复别名量，并删除quantity中的`annotated`字段；
- 六个Scene按对象与环境分类，共保留3818个正式quantity：3677个标量和141个时序量；
- 141条推水瓶外力序列共含4698个XLSX样本。原始记录中的62处重复时间间隔和193处
  大于0.1秒的间隙按原样保留；746个负传感器漂移值在正式非负大小标注中截为0，原始
  有符号值仍保存在归一化来源记录中；
- 799个Case各自只保留一个物理文件；
- Release只保留四类运行时内容；迁移、逐Case provenance与独立验证记录位于
  `datasets/provenance/releases/12.0.0/`。

独立验证：

```bash
PYTHONPATH=src:tests:. python scripts/validate_dataset_v12.py
```

## 10. 历史版本失活

V1–V11不再位于活动`datasets/releases/`目录。旧Dataset ID与迁移过程保留在Git历史和`datasets/provenance/`证据中，仅用于审计，不能作为当前运行入口。

## 11. 验收

快速 metadata 与资产存在性检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/12.0.0/dataset.json \
  --check-assets
```

Loader 同时验证 descriptor、Case schema、scene、View coverage、路径越界，以及
Case-local caption/physics文件的身份与内容。当前发布门禁还应运行
`scripts/validate_dataset_v12.py`；
旧Release没有活动兼容门禁。
