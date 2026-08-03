# Dataset、Case 与划分

## 1. 权威入口

当前 DatasetSnapshot：

```text
dataset_id:     physics_video_six_scene_v6
release:        6.0.0
descriptor:     datasets/physics_video/releases/6.0.0/dataset.json
cases:          604
locked assets:  1650
dataset digest: bb7b4a0d5292cf669e39a030b1ece3ef24515cb7b43b2d6ff9320597c1003fcd
```

`datasets/` 是唯一权威数据根。Baseline、cache 和 run 不能回写或覆盖这里的资产。
新增原始视频和XLSX的完整导入流程见
[DATASET_INGESTION.md](DATASET_INGESTION.md)。

## 2. 目录结构

```text
datasets/
└── physics_video/
    ├── assets/
    │   ├── pendulum/
    │   ├── free_fall/
    │   ├── collision_1d/
    │   ├── inclined_plane_slide/
    │   ├── uniform_circular_motion/
    │   ├── parabolic_motion/
    │   └── source_archives/
    ├── provenance/
    │   ├── imports/
    │   └── source_docs/
    └── releases/
        ├── 1.0.0/…5.1.0/            # 历史结果引用的只读元数据
        └── 6.0.0/
            ├── dataset.json          # 唯一加载入口
            ├── release.json          # Dataset 与资产集合 digest
            ├── cases.jsonl           # Case schema 4.0
            ├── assets.lock.json      # 引用资产的大小和 SHA-256
            ├── ball_spec_catalog.json
            ├── annotation_corrections.json
            ├── asset_directory_mapping.json
            ├── split_audit.json
            ├── migration_audit.json
            ├── scenes/
            └── views/
```

原始压缩包按字节保存在 `assets/source_archives/`，Case 的
`provenance.source_locator` 记录 archive/member。供运行和评估使用的 canonical
视频、首帧等按 scene/描述性物理目录存放，路径稳定且受 asset lock 保护。

## 3. Case schema 4.0

每行 Case 的核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | 全 Dataset 唯一稳定 ID |
| `scene_id` | 六个正式 scene 之一 |
| `text` | 原始 prompt、语言与标注来源 |
| `assets` | 首帧、reference、source archive、可选 mask 等 |
| `physics` | 结构化物理量及其可信状态 |
| `appearance` | 非结构化物理量的情景、外观、环境、实验形式与采集信息 |
| `temporal` | encoded time 与物理时间关系 |
| `provenance` | 来源、parent case 与原始成员定位 |
| `alignment` | 可选的时间对齐审核 |
| `has_real_reference_video` | 是否有真实 reference |

Case 不包含 Task partition、ID/OOD、模型分辨率、runner 参数或模型原生输入。
ID/OOD 必须相对于一份具体训练集定义，因此属于 View A 的 test annotation，不能作为
Case 的固有属性。5.1.0 的 `case.ood` 只保留用于历史结果复现。

`appearance` 沿用历史字段名，但它的语义比视觉外观更宽：背景、颜色、材质、机位、
采集批次，以及球数、初始运动球数、碰撞结构等非数值实验形式都放在这里。它们可以
作为 View A 的泛化因素，但不能混入 `physics`。

## 4. 原始 prompt

原始文本固定保存在：

```text
case.text.prompt
```

`text` 的完整结构：

```json
{
  "schema_version": "1.0",
  "prompt": "A fixed-camera real-world laboratory video ...",
  "language": "en",
  "annotation_source": "five_scene_prompt_v1"
}
```

该 prompt 描述 scene 与可见运动。碰撞 scene 的 5.1.0 prompt 逐 case 明确主体数量、
初始运动/静止角色和运动方向，避免用一个模板模糊不同情景；其它 scene
继续使用各自冻结的 scene 文本。所有 Baseline 都以它为文本源；是否原样使用、
追加结构化物理文本或转换成其它模型表示，由 Baseline 的 `input_policy` 与 adapter
决定。Task 不生成或选择 prompt。

## 5. 结构化物理标注

物理量与 prompt 并列保存在：

```text
case.physics.<parameter>
```

每个 quantity 至少包含：

```json
{
  "value": 54.55,
  "unit": "deg/s",
  "annotated": true
}
```

`annotated=true` 表示该字段可作为模型输入并可进入使用审计；
`annotated=false` 通常是派生值、换算值或不够可信的量，不会出现在传给 adapter 的
conditionable Case 中。数据导入时无法建立可靠标注对应关系的样本或字段不应进入正式
可条件化数据。

主要物理量：

- 单摆：摆长、绳长、摆球半径、初始角度；
- 自由落体：初始高度、球半径、质量、初速度；
- 一维碰撞：球质量、半径、初速度；
- 斜面下滑：斜面角度、质量、摩擦系数、理论加速度；
- 匀速圆周运动：角速度、一个或两个物体的轨道半径；
- 平抛运动：出门初速度、竖直落差、球质量与半径。

## 6. 资产角色

典型 Case：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
├── source/                 # 可选原始字节
└── canonical/
    ├── reference.mp4       # evaluator 使用的物理参考
    └── first_frame.png     # I2V 输入
```

规则：

- Case 中的路径相对 `dataset.asset_root`；
- `assets.lock.json` 封印所有引用文件的大小和 SHA-256；
- 6.0.0 沿用的目录名只编码 scene 的主要结构化物理量与唯一身份后缀，不使用背景、
  颜色或采集环境；
- 6.0.0 是元数据迁移，与5.1.0引用完全相同的冻结媒体，没有复制或修改视频 payload；
- I2V 使用显式 `assets.first_frame`，不在运行时从 GT 临时补首帧；
- reference/source/provenance 属于 evaluator 或数据审计，不交给生成 driver；
- 尺寸、FPS、帧数、抽帧和特征派生物只能进入 immutable cache 或 run。

斜面下滑的 canonical 视频已经过启动时刻清洗。新增平抛和碰撞视频均从球刚穿过
光电门且所有球完整可见的首帧开始，画面已经裁去光电门与线缆；平抛另裁为合适的竖屏
画幅。首尾帧视觉复核结论、裁剪框和源帧范围均随 Dataset provenance 冻结。生成阶段
不再重复裁剪。

V2V 必须另外登记独立输入视频资产，例如 `assets.input_video`。其中
`conditioning_video` 只是 adapter 对该输入媒体 channel 的角色名，不表示 Task
层的物理信息分组；reference、physics reference 和 source video 均禁止充当 V2V 输入。
当前 6.0.0 release 没有正式 `assets.input_video`，因此 V2V Bundle 只是接口能力，
不能直接运行官方数据。

## 7. View A：finetune_eval

View A schema 3.0 只使用两个互斥主划分：`train` 和 `test`。两者完整覆盖604条Case，
不再为了构造纯ID/OOD而排除有效数据。ID/OOD/mixed是每条test Case相对于
`view_a.train`的分析标签：

- `id`：测试情景处于训练集已覆盖的因素域，允许独立试次和分布内连续数值变化；
- `ood`：至少一个明确因素在训练中未出现，且控制因素仍处于训练支持域；
- `mixed`：OOD因素与另一个 held-out 因素同时变化，无法把误差归因于单一因素。

当前数量：

| scene | 全部 | train | test | test-ID | test-OOD | test-mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pendulum | 35 | 27 | 8 | 8 | 0 | 0 |
| free_fall | 11 | 7 | 4 | 4 | 0 | 0 |
| collision_1d | 330 | 230 | 100 | 58 | 42 | 0 |
| inclined_plane_slide | 95 | 58 | 37 | 15 | 16 | 6 |
| uniform_circular_motion | 36 | 18 | 18 | 6 | 12 | 0 |
| parabolic_motion | 97 | 70 | 27 | 27 | 0 | 0 |
| **合计** | **604** | **410** | **194** | **118** | **70** | **6** |

OOD因素同时记录类别：`physical_parameter`、`object_composition`、
`interaction_structure`、`appearance`、`environment`、`acquisition`。例如背景属于
`environment`，不属于物理OOD；榜单可把它单独报告为视觉/环境鲁棒性。当前6条mixed
均为斜面角度holdout与新背景共同变化的样本。

当前实际出现的OOD因素：

| factor | category | test数量 | 含义 |
| --- | --- | ---: | --- |
| `ball_spec_composition` | `object_composition` | 42 | 训练未见的碰撞球规格组合 |
| `collision_structure` | `interaction_structure` | 34 | 双球相向入射；与训练中的单入射结构不同 |
| `ball_material` | `object_composition` | 8 | 上述碰撞OOD中含玻璃球的新材质组成子集 |
| `moving_object_composition` | `object_composition` | 12 | 训练未见的圆周运动物体组成 |
| `background` | `environment` | 22 | 新背景；其中6条还与角度holdout共同变化 |

碰撞scene按replicate-safe组件划分：相同近重复簇、审核重复对或完全相同结构化物理
signature不会跨train/test。钢球单入射数据在每个“球规格序列×碰撞结构”stratum内按
80/20划为train和test-ID，并强制速度范围端点留在train；34条双球相向入射和8条含
玻璃球数据进入test-OOD。由于这42条的精确球规格组成也未在train出现，它们同时带有
`ball_spec_composition`因素；factor子组不是互斥计数。

平抛仍使用按球规格、发射高度和初速度组成的物理signature成组划分，train/test没有
完全相同signature泄漏。对环境OOD，test与train具有相同物理signature可以是刻意的
控制变量设计，不应误判为泄漏；真正禁止的是同一原始试次、重复媒体或近重复片段跨界。

## 8. View B：direct_eval

View B：

- 完整覆盖 604 个 case；
- scene 内按冻结 seed 确定性分组；
- group 尽量均衡；
- group 是报告/抽样维度，不表达模型的物理使用方式，也不等同于 ID/OOD 层级。

## 9. Release 6.0.0 的划分迁移

6.0.0 从5.1.0元数据生成，媒体字节和资产digest不变。5.1.0只提供Case事实和历史
审计，不再约束碰撞scene的新划分：

- Case升级到schema 4.0，移除视图相关的`ood`字段；
- View A升级到schema 3.0，完整覆盖train/test；
- 碰撞scene基于全部330条有效Case重新分层，得到230条train、58条test-ID和42条
  test-OOD；其它scene保留5.1.0的train成员，并把其余有效Case纳入test；
- 碰撞划分把人工审核的近重复cluster、重复对和完全相同物理signature作为不可拆分组，
  train/test之间没有replicate component重叠；
- test的ID/OOD/mixed、OOD因素及因素类别写入`view_a.test_annotations`；
- Task schema 4.0只选择`test`，总体Test分是主分，regime/factor是诊断分；
- 子组少于官方Task规定的5个job时报告`N/A`，不输出不稳定严格分；
- View B和604条Case资产引用保持不变。

可复现脚本：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/migrate_dataset_v6.py
```

逐scene数量、digest与零媒体改动声明见`6.0.0/migration_audit.json`。

## 10. Release 5.1.0 的规范化语义（历史）

5.1.0 从不可变的 5.0.0 派生：

- 统一钢球规格 ID：小球 15 mm / 14.00 g，中球 20 mm / 33.13 g，
  大球 25 mm / 64.77 g；
- 对使用旧直径除以光电门遮挡时间得到的速度，保持遮挡时间不变并同步重算；
- 平抛的质量、半径和初速度按权威规格修正；装置侧辅助测量保留但设为
  `annotated=false`；
- 文本仅描述可由首帧消歧的物理过程，不包含具体物理数值、背景颜色或裁剪提示；
- 移除 5 条首帧与参考视频不匹配的合成单摆 OOD case，保留其 3 条真实
  parent case 及媒体；
- canonical 视频仅做必要的事件窗口和空间裁剪，保留窗口内全部源帧与源 FPS；
  `4n+1`、目标 FPS、最大帧数和真实时间适配全部由基线 adapter 完成；
- 平抛 View A 改为 grouped physical-signature holdout，消除 train/test_id
  相同物理条件泄漏；
- 六个 scene 的 case 资产统一使用含主要物理量的描述性目录；背景只属于
  `appearance`/OOD 元数据，不进入 `physics`、物理 signature 或目录名；
- 除明确退役的 5 条合成 OOD case 外，case ID 和媒体内容不变。

可复现脚本：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/normalize_dataset_v51.py
```

逐 case 修正、规格别名、目录映射和新划分分别见 release 内的
`annotation_corrections.json`、`ball_spec_catalog.json`、
`asset_directory_mapping.json` 和 `split_audit.json`。

### 5.0.0 导入历史

5.0.0 从不可变的 4.0.0 派生：

- 保留 4.0.0 的 214 个 case，不修改其媒体字节；
- 新增 97 个有独立物理标注的平抛 case；
- 新增 298 个有唯一标注、且与旧 32 个碰撞源视频不重复的碰撞 case；
- 平抛 ZIP 中 31 个无独立标注的视频未导入；
- 碰撞工作簿中 107 行无对应视频的标注未导入；
- 新增来源 ZIP、逐 case 原始 MOV、canonical MP4、首帧和完整 provenance；
- 以独立 release 发布，4.0.0 继续冻结。

可复现脚本：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/prepare_20260730_parabolic_collision.py

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/import_20260730_parabolic_collision.py
```

准备脚本生成的 staging 必须完成视觉复核并封存为 `visually_verified` 后，导入脚本才会
接受。正式 release 一旦发布应视为不可变。4.0.0 从 3.0.0 固化 Case schema 3.0 与
canonical prompt 的历史迁移语义保持不变。

## 11. 验收

快速 metadata 与资产存在性检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --check-assets
```

发布前逐文件哈希检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/6.0.0/dataset.json \
  --check-asset-hashes
```

Loader 同时验证 descriptor、Case schema、scene、View coverage、路径越界、
`assets.lock.json` 和 `release.json` 中的 Dataset digest。
