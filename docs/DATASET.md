# 数据集、资产与划分

## 1. 权威入口

当前正式 Dataset：

```text
dataset_id: physics_video_five_scene_v3
release:    3.0.0
descriptor: datasets/physics_video/releases/3.0.0/dataset.json
cases:      214
assets:     468 files
```

`datasets/` 是唯一数据根。代码、Baseline 和 run 只能读取这里的资产，不能回写重采样
或模型派生内容。

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
    │   ├── source_docs/
    │   └── alignment/
    └── releases/3.0.0/
        ├── dataset.json
        ├── release.json
        ├── cases.jsonl
        ├── assets.lock.json
        ├── scenes/
        └── views/
```

`source_archives/` 保存批量来源的原始字节；case 使用
`provenance.source_locator.member` 指向压缩包成员。Canonical 视频仍按 scene/case
独立存放，保证运行时路径稳定。

## 3. Case 契约

每行 case 使用 schema `2.0`，核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | 全 Dataset 唯一稳定 ID |
| `scene_id` | 五个正式 scene 之一 |
| `physics` | 带值、单位和 `annotated` 状态的物理量 |
| `appearance` | 背景、主体、颜色、材质、机位、采集批次 |
| `ood` | `id/ood1` 与变化因素 |
| `temporal` | 时间尺度及 encoded-to-physical 语义 |
| `assets` | 首帧、reference、source 或 source archive |
| `provenance` | 来源类型、父 case、原始成员定位 |
| `alignment` | 需要首帧对齐的 scene 的审核信息 |

Case 不保存 prompt 或模型原生输入。

## 4. 资产契约

典型单 case：

```text
assets/<scene_id>/<case_id>/
├── source/                 # 可选：原始逐字节资产
└── canonical/
    ├── reference.mp4       # Benchmark 使用的物理时间参考
    └── first_frame.png     # reference 第 0 帧或审核后的等价首帧
```

规则：

- 正式 case 引用必须位于 `asset_root` 内。
- `assets.lock.json` 封印所有引用文件的大小和 SHA-256。
- `first_frame` 必须与 canonical reference 的帧 0 一致。
- Baseline 的尺寸、FPS、帧数适配只能写入 cache/run。
- 自由落体 canonical reference 已表达真实物理时间；source 保留原始慢放字节。
- 斜面 canonical frame 0 是人工审核的启动附近帧。

## 5. 场景物理量

### 单摆

- 数值：摆长、绳长、摆球半径、初始角度。
- 环境：背景、底座和装置外观。
- Evaluator 使用摆角、周期、振幅、支点和长度稳定性。

### 自由落体

- 数值：初始高度、球半径、质量、零初速度。
- 当前只有 ID 评测，没有可验证的 OOD1 continuation。
- Evaluator 使用 case 自身的短时物理窗口。

### 一维碰撞

- 数值：三球质量、半径、初速度。
- 环境：球组合、材质和背景。
- 数据没有可信恢复系数标签；评估时从 reference 轨迹估计。

### 斜面下滑

- 数值：角度、质量、摩擦系数、理论加速度。
- 环境：背景、滑块、轨道材质、视角。
- `initial_velocity` 未被可信标注，evaluator 将其作为拟合量。

### 匀速圆周运动

- 数值：角速度、一到两个物体的轨道半径。
- 环境：圆盘、运动物体组合和视角。
- 初始角位置没有标注，evaluator 使用相对首帧角度。

## 6. View A

View A 服务 `finetune_eval`。划分原则：

- 训练集在可用对照内占主导；
- 训练集必须包含多个环境；
- `test_id` 使用训练未见数值，但环境来自训练；
- `test_ood1` 使用训练未见环境，但物理数值来自训练；
- 同时改变数值和环境的 case 不进入 View A。

当前数量：

| scene | 全部 case | train | test_id | test_ood1 | View A coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| pendulum | 40 | 27 | 8 | 5 | 40 |
| free_fall | 11 | 7 | 4 | 0 | 11 |
| collision_1d | 32 | 11 | 3 | 18 | 32 |
| inclined_plane_slide | 95 | 58 | 6 | 16 | 80 |
| uniform_circular_motion | 36 | 18 | 2 | 4 | 24 |

斜面和圆周未进入 View A 的有效 case 仍属于 Dataset，并由 View B 覆盖。

## 7. View B

View B 服务 `direct_eval`：

- 覆盖全部 214 个 case；
- scene 内使用确定性 seed 分组；
- 尽量保持各 group 数量均衡；
- group 是报告维度，不表达 ID/OOD 层级。

## 8. 数据验收

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-asset-hashes
```

验收包含 descriptor、case schema、scene、View coverage、路径越界、资产存在性和
`assets.lock.json` 逐文件哈希。

斜面启动帧重建：

```bash
/root/miniconda3/envs/phybench/bin/python \
  scripts/align_inclined_plane.py --propose --workers 6

/root/miniconda3/envs/phybench/bin/python \
  scripts/align_inclined_plane.py --accept-proposals

/root/miniconda3/envs/phybench/bin/python \
  scripts/align_inclined_plane.py --materialize --workers 4
```

新场景 metadata 重建入口是 `scripts/import_20260723_scenes.py`。任何重建都必须重新
生成 asset lock，并通过完整哈希验收后才能作为 DatasetSnapshot 使用。
