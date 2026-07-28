# Dataset、Case 与划分

## 1. 权威入口

当前 DatasetSnapshot：

```text
dataset_id:     physics_video_five_scene_v4
release:        4.0.0
descriptor:     datasets/physics_video/releases/4.0.0/dataset.json
cases:          214
locked assets:  468
dataset digest: be5ea8880be3cf8e0d0d316ee025cf02905ef5aeb544f3df5f4b966e5e14782d
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
    │   └── source_archives/
    ├── provenance/
    │   ├── imports/
    │   └── source_docs/
    └── releases/
        ├── 3.0.0/                    # 冻结旧 release
        └── 4.0.0/
            ├── dataset.json          # 唯一加载入口
            ├── release.json          # Dataset 与资产集合 digest
            ├── cases.jsonl           # Case schema 3.0
            ├── assets.lock.json      # 引用资产的大小和 SHA-256
            ├── migration_audit.json
            ├── scenes/
            └── views/
```

原始压缩包按字节保存在 `assets/source_archives/`，Case 的
`provenance.source_locator` 记录 archive/member。供运行和评估使用的 canonical
视频、首帧等按 scene/case 存放，路径稳定且受 asset lock 保护。

## 3. Case schema 3.0

每行 Case 的核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | 全 Dataset 唯一稳定 ID |
| `scene_id` | 五个正式 scene 之一 |
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

该 prompt 描述 scene 与可见运动，不含按 case 拼接的结构化数值。所有 Baseline 都以它
为文本源；是否原样使用、追加物理文本或转换成其它模型表示，由 Baseline 的
`input_policy` 与 adapter 决定。Task 不生成或选择 prompt。

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
- 匀速圆周运动：角速度、一个或两个物体的轨道半径。

## 6. 资产角色

典型 Case：

```text
assets/<scene_id>/<case_id>/
├── source/                 # 可选原始字节
└── canonical/
    ├── reference.mp4       # evaluator 使用的物理参考
    └── first_frame.png     # I2V 输入
```

规则：

- Case 中的路径相对 `dataset.asset_root`；
- `assets.lock.json` 封印所有引用文件的大小和 SHA-256；
- I2V 使用显式 `assets.first_frame`，不在运行时从 GT 临时补首帧；
- reference/source/provenance 属于 evaluator 或数据审计，不交给生成 driver；
- 尺寸、FPS、帧数、抽帧和特征派生物只能进入 immutable cache 或 run。

斜面下滑的 canonical 视频已经过启动时刻清洗：首帧位于物体“几乎开始下滑”的时刻，
对应审核信息随 Dataset provenance 冻结。生成阶段不再重复裁剪。

V2V 必须另外登记独立输入视频资产，例如 `assets.input_video`。其中
`conditioning_video` 只是 adapter 对该输入媒体 channel 的角色名，不表示 Task
层的物理信息分组；reference、physics reference 和 source video 均禁止充当 V2V 输入。
当前 4.0.0 release 没有正式 `assets.input_video`，因此 V2V Bundle 只是接口能力，
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
| collision_1d | 32 | 11 | 3 | 18 | 32 |
| inclined_plane_slide | 95 | 58 | 6 | 16 | 80 |
| uniform_circular_motion | 36 | 18 | 2 | 4 | 24 |

View A 是有约束的 subset；未进入 View A 的有效 case 仍属于 Dataset，并由 View B 覆盖。

## 8. View B：direct_eval

View B：

- 完整覆盖 214 个 case；
- scene 内按冻结 seed 确定性分组；
- group 尽量均衡；
- group 是报告/抽样维度，不表达模型的物理使用方式，也不等同于 ID/OOD 层级。

## 9. Release 4.0.0 的迁移语义

4.0.0 相对 3.0.0：

- Case 从 schema 2.0 升级到 3.0；
- 将 canonical base prompt 固化为 `case.text.prompt`；
- 保留原结构化 physics、appearance、assets、provenance 与 OOD；
- 不改变 case 集合、View、scene catalog 或资产字节；
- 创建独立 release，未修改 3.0.0。

可复现脚本：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/migrate_dataset_v4.py --force
```

执行前应确认目标确实是重建 4.0.0；正式 release 一旦发布应视为不可变。

## 10. 验收

快速 metadata 与资产存在性检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --check-assets
```

发布前逐文件哈希检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --check-asset-hashes
```

Loader 同时验证 descriptor、Case schema、scene、View coverage、路径越界、
`assets.lock.json` 和 `release.json` 中的 Dataset digest。
