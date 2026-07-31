# Dataset、Case 与划分

## 1. 权威入口

当前 DatasetSnapshot：

```text
dataset_id:     physics_video_six_scene_v5p1
release:        5.1.0
descriptor:     datasets/physics_video/releases/5.1.0/dataset.json
cases:          609
locked assets:  1660
dataset digest: ed395e528a7457d1fccb2a6d88f271009f2995120b5a0db61c52a3f1e348a781
```

`datasets/` 是唯一权威数据根。Baseline、cache 和 run 不能回写或覆盖这里的资产。

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
        ├── 3.0.0/                    # 冻结旧 release
        ├── 4.0.0/                    # 冻结五场景 release
        ├── 5.0.0/                    # 冻结的六场景首次导入
        └── 5.1.0/
            ├── dataset.json          # 唯一加载入口
            ├── release.json          # Dataset 与资产集合 digest
            ├── cases.jsonl           # Case schema 3.0
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

## 3. Case schema 3.0

每行 Case 的核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | 全 Dataset 唯一稳定 ID |
| `scene_id` | 六个正式 scene 之一 |
| `text` | 原始 prompt、语言与标注来源 |
| `assets` | 首帧、reference、source archive、可选 mask 等 |
| `physics` | 结构化物理量及其可信状态 |
| `appearance` | 背景、主体、颜色、材质、机位、采集批次 |
| `temporal` | encoded time 与物理时间关系 |
| `ood` | `id/ood1` 与变化因素 |
| `provenance` | 来源、parent case 与原始成员定位 |
| `alignment` | 可选的时间对齐审核 |
| `has_real_reference_video` | 是否有真实 reference |

Case 不包含 Task partition、模型分辨率、runner 参数或模型原生输入。

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

该 prompt 描述 scene 与可见运动。碰撞 scene 的 5.1.0 prompt 逐 case 明确球数、
从左到右的球规格、运动方向和初速度，避免用一个模板模糊不同情景；其它 scene
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
- 5.1.0 的目录名只编码 scene 的主要结构化物理量与唯一身份后缀，不使用背景、
  颜色或采集环境；
- 5.1.0 的新路径是指向冻结媒体的硬链接，不复制视频 payload；旧路径继续供
  5.0.0 使用；
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
当前 5.1.0 release 没有正式 `assets.input_video`，因此 V2V Bundle 只是接口能力，
不能直接运行官方数据。

## 7. View A：finetune_eval

View A 把物理数值与环境变化分开：

- 训练集尽可能占主导，并覆盖多个环境；
- `test_id` 使用训练未见的物理数值，环境来自训练；
- `test_ood1` 使用训练未见的环境，物理数值来自训练；
- 同时改变数值与环境且无法可靠归类的 case 不进入 View A。

冻结数量：

| scene | 全部 | train | test_id | test_ood1 | View A coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| pendulum | 40 | 27 | 8 | 5 | 40 |
| free_fall | 11 | 7 | 4 | 0 | 11 |
| collision_1d | 330 | 95 | 32 | 63 | 190 |
| inclined_plane_slide | 95 | 58 | 6 | 16 | 80 |
| uniform_circular_motion | 36 | 18 | 2 | 4 | 24 |
| parabolic_motion | 97 | 70 | 27 | 0 | 97 |

View A 是有约束的 subset；未进入 View A 的有效 case 仍属于 Dataset，并由 View B 覆盖。
碰撞补充数据中，同质球且来源 curation 为 train/test 的样本分别进入 train/test_id；
异质球的来源 test 样本进入 test_ood1。异质球来源 train 与 9 个来源侧 near-replicate
样本不进入 View A，以保持 collision-pair OOD 纯度，但仍保留在 Dataset 和 View B。
平抛使用 `grouped_physical_signature_holdout_v1`：球规格、发射高度和重算后的初速度
完全相同的 case 必须整体进入 train 或 test_id。背景不参与 signature；当前仍保持
70/27 数量，且 train/test_id 的物理 signature 交集为 0。

## 8. View B：direct_eval

View B：

- 完整覆盖 609 个 case；
- scene 内按冻结 seed 确定性分组；
- group 尽量均衡；
- group 是报告/抽样维度，不表达模型的物理使用方式，也不等同于 ID/OOD 层级。

## 9. Release 5.1.0 的规范化语义

5.1.0 从不可变的 5.0.0 派生：

- 统一钢球规格 ID：小球 15 mm / 14.00 g，中球 20 mm / 33.13 g，
  大球 25 mm / 64.77 g；
- 对使用旧直径除以光电门遮挡时间得到的速度，保持遮挡时间不变并同步重算；
- 平抛的质量、半径和初速度按权威规格修正；装置侧辅助测量保留但设为
  `annotated=false`；
- 碰撞 prompt 逐 case 描述球数、从左到右的规格、运动方向和初速度，不推测碰撞结果；
- 平抛 View A 改为 grouped physical-signature holdout，消除 train/test_id
  相同物理条件泄漏；
- 六个 scene 的 case 资产统一使用含主要物理量的描述性目录；背景只属于
  `appearance`/OOD 元数据，不进入 `physics`、物理 signature 或目录名；
- case ID 和媒体内容不变，新目录全部用硬链接实现。

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

## 10. 验收

快速 metadata 与资产存在性检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/5.1.0/dataset.json \
  --check-assets
```

发布前逐文件哈希检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/5.1.0/dataset.json \
  --check-asset-hashes
```

Loader 同时验证 descriptor、Case schema、scene、View coverage、路径越界、
`assets.lock.json` 和 `release.json` 中的 Dataset digest。
