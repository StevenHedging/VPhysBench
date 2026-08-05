# Physics Video Benchmark 数据集信息与结构汇总

> 审阅范围：`/root/Steven/physics_video_benchmark/datasets/` 及其直接引用的数据规范  
> 审阅日期：2026-08-04（UTC）  
> 本文主体是 7.0.0 历史审阅快照；当前目录入口见 `README.md`，路径已迁移到
> `releases/11.0.0/dataset.json`。11.0.0的799条Case均有Case-local
> `physics.v11.json`和符号化非负物理量；10.0.0的`physics.json`保持为历史证据。
> 7.0.0时代遗留的32个无引用碰撞目录已在V10迁移中删除；本文后文
> 对旧目录和旧数量的描述只用于解释历史快照。

## 1. 一页概览

这是一个面向物理视频生成与评测的数据集。每个 Case 由真实实验视频、首帧、英文过程描述、结构化物理量、外观/环境信息和来源审计信息组成。典型 I2V 流程以 `first_frame` 为视觉输入，以 `reference_video` 为评测参考；结构化物理量是否进入模型，由 Baseline 的 `input_policy` 和 adapter 决定，而不是由 Dataset 或 Task 强制决定。

| 项目 | 当前值 |
| --- | --- |
| Dataset ID | `physics_video_five_scene_v7` |
| Release | `7.0.0` |
| Case schema | `4.0` |
| Case 数 | 593 |
| Scene 数 | 5 |
| 数据来源 | 593/593 均为 `real_capture` |
| 真实参考视频 | 593/593 均为 `true` |
| View A | Train 403，Test 190 |
| Test 泛化标签 | ID 114，OOD 70，Mixed 6 |
| View B | seed 42，5 个全量分组 |
| 锁定资产 | 1,617 个文件，约 38.09 GiB |
| 参考视频总时长 | 约 36.12 分钟 |
| Dataset digest | `9c89d2a2848253ffcc7dafeba9f7fb1b230db78c0fa09c7efc3045306aa2014f` |
| Asset-files digest | `f5698cd55fbff491906ec4f2010730b960acb9d45054972b5318055bf2a1e277` |

数据集可以抽象为三层：

1. **冻结资产层**：真实 source、canonical reference 和首帧；
2. **可追溯证据层**：原始标注、导入、排除、去重、裁剪和时间对齐审核；
3. **版本化发布层**：Case、Scene、View、资产锁和 Dataset digest。

运行时必须从 `7.0.0/dataset.json` 加载。不能扫描 `assets/` 推断 Case，也不能扫描版本目录后自动选择或回退到旧 release。

## 2. 目录结构

```text
datasets/
├── README.md
├── _incoming/                         # 新来源暂存；当前只有 .gitkeep
├── assets/                            # 按scene/case组织的冻结媒体资产
│   ├── README.md
│   ├── <scene_id>/<physical_case_dir>/
│   │   ├── canonical/
│   │   │   ├── reference.mp4
│   │   │   └── first_frame.png
│   │   └── source/                    # 可选原始成员，常见 reference.mov
│   └── source_archives -> ../provenance/source_archives
├── provenance/                        # 来源与质量审计
│   ├── source_archives/               # 批次原始ZIP，仅保存一次
│   ├── source_docs/                   # XLSX、TXT、规范化标注
│   ├── imports/                       # 导入记录与排除清单
│   ├── alignment/                     # 起始帧、入场、裁剪和时标审核
│   └── path_relocation.json
└── releases/
    ├── README.md
    ├── 1.0.0/ ... 6.0.0/             # 仅用于解释/复现历史结果
    └── 7.0.0/                         # 本文统计对应的历史快照
        ├── dataset.json
        ├── release.json
        ├── cases.jsonl
        ├── assets.lock.json
        └── scenes/*.json
            ├── views/view_a.json
            ├── views/view_b.json
            ├── ball_spec_catalog.json
            ├── annotation_corrections.json
            ├── asset_directory_mapping.json
            ├── split_audit.json
            ├── migration_audit.json
            └── text_video_alignment_repair.json
```

目录关系如下：

```text
dataset.json
├── cases.jsonl ──资产相对路径──> assets/
├── scenes/       ──参数/指标契约
├── views/        ──训练、测试和报告分组
├── assets.lock.json ──大小/SHA-256/角色/Case反向引用
└── release.json  ──Dataset与资产集合digest

provenance/ ──为导入、标注、裁剪和对齐提供审计证据
```

在本文对应的7.0.0历史快照中，主要空间来自碰撞媒体和原始压缩包；当前原始压缩包的
权威物理位置已迁移到`provenance/source_archives`。

## 3. 当前 Release 中各文件的职责

| 文件或目录 | 作用 |
| --- | --- |
| `dataset.json` | 唯一加载入口；声明 Dataset ID、release、Case schema、asset root 及其它元数据路径 |
| `cases.jsonl` | 593 行，一行一个 Case；记录数据事实，不记录 train/test 身份 |
| `scenes/*.json` | 定义结构化物理参数、不可条件化参数、泛化因素、约束与评测关注量 |
| `views/view_a.json` | Finetune/eval 的互斥 Train/Test 划分及 Test 的 ID/OOD/Mixed 注释 |
| `views/view_b.json` | Direct-eval 使用的 seed 42、5 组全量确定性分组 |
| `assets.lock.json` | 锁定 1,617 个资产的相对路径、角色、大小、SHA-256 与关联 Case |
| `release.json` | 冻结 Dataset digest、资产集合 digest 和资产数量 |
| `ball_spec_catalog.json` | 球规格权威表及别名规范 |
| `annotation_corrections.json` | 标注和物理量修正记录 |
| `asset_directory_mapping.json` | 稳定 `case_id` 与描述性物理目录之间的映射 |
| `split_audit.json` | View A 划分数量、成员保持和泄漏检查摘要 |
| `migration_audit.json` | 从 6.0.0 到 7.0.0 的删除和身份迁移记录 |
| `text_video_alignment_repair.json` | 2026-08-03 文本/媒体原地修复的逐项证据 |

## 4. Case 数据模型

593 条 Case 均包含以下核心字段：

| 字段 | 语义 |
| --- | --- |
| `case_id` | Dataset 内唯一、稳定的 Case 身份 |
| `scene_id` | 所属物理场景 |
| `text` | 英文过程描述、语言、文本来源和 text schema |
| `physics` | 结构化物理量；每项包含 `value`、`unit`、`annotated` |
| `appearance` | 背景、材质、对象组成、相机、采集批次等非数值情景信息 |
| `assets` | 首帧、参考视频、可选 source/source archive/mask 路径 |
| `temporal` | 编码时间与物理时间的关系及说明 |
| `provenance` | 来源类型、原始成员定位、parent/generator 信息 |
| `alignment` | 可选的裁剪、帧范围、时标和视觉复核记录 |
| `has_real_reference_video` | 是否具有同一试次的真实参考视频 |

物理量的基本结构为：

```json
{
  "value": 0.03313,
  "unit": "kg",
  "annotated": true
}
```

- `annotated=true` 表示可信且可作为模型条件量；
- `annotated=false` 通常表示换算值、辅助装置量或不应直接条件化的量；
- 全集共出现 35 种物理参数名、4,397 条参数记录，其中 3,880 条为 `annotated=true`，517 条为 `false`；
- ID/OOD、Train/Test、模型分辨率和运行参数都不属于 Case 的固有事实。

文本方面，593 条 prompt 均非空且均为英语，但只有 11 个不同 prompt：碰撞 6 种，斜面、平抛和单摆各 1 种，圆周运动 2 种。因此文本主要是场景/过程模板，不是 593 条彼此独立的自然语言描述。

字段覆盖情况：

- 522/593 条带 `alignment`；碰撞、斜面和平抛全部具有显式对齐记录，单摆和圆周运动没有；
- 427/593 条带逐 Case `source_video`；
- 526/593 条引用批次 `source_archive`；
- 593 条的 `subject_mask` 当前均为 `null`；
- 593 条均为真实采集，`generator=null`、`parent_case_id=null`；
- `reference_video` 与 `physics_reference_video` 在当前全集中指向同一路径。

## 5. 场景、规模与划分

| Scene | 中文名 | Case | 占比 | Train | Test | Test-ID | Test-OOD | Test-Mixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `collision_1d` | 一维碰撞 | 330 | 55.65% | 230 | 100 | 58 | 42 | 0 |
| `inclined_plane_slide` | 斜面下滑 | 95 | 16.02% | 58 | 37 | 15 | 16 | 6 |
| `parabolic_motion` | 平抛运动 | 97 | 16.36% | 70 | 27 | 27 | 0 | 0 |
| `pendulum` | 单摆 | 35 | 5.90% | 27 | 8 | 8 | 0 | 0 |
| `uniform_circular_motion` | 匀速圆周运动 | 36 | 6.07% | 18 | 18 | 6 | 12 | 0 |
| **合计** |  | **593** | **100%** | **403** | **190** | **114** | **70** | **6** |

### 5.1 一维碰撞

- 330 Case，占全集 55.65%；
- 两球单侧入射 230 条、两球对向入射 34 条、三球单侧入射 66 条；
- 球质量为 0.00346、0.014、0.03313、0.06477 kg 四档；
- 球半径为 0.00675、0.0075、0.01、0.0125 m 四档；
- 球速度整体约覆盖 -1.38760 至 1.26476 m/s；
- 第三球的质量、半径和速度字段只在 66 个三球 Case 中出现；
- 评测关注碰撞前后速度、动量/恢复系数残差、接触顺序、穿模和近似一维运动。

### 5.2 斜面下滑

- 95 Case；斜面角度为 32°、35°、38°、41°、44°；
- 物块质量固定 0.08768 kg，长度固定 0.11 m，动摩擦系数固定 0.463；
- 理论加速度约为 1.34619–3.54612 m/s²；
- 四类背景为默认白色、黑色泡沫板、绿色卡纸和油画；
- `initial_velocity=0` 被保存但为 `annotated=false`，Scene catalog 将它列为不可条件化参数；
- canonical 第 0 帧被定义为人工/视觉复核后的即将开始或刚开始下滑时刻。

### 5.3 平抛运动

- 97 Case；大钢球 77 条，中钢球 20 条；
- 质量为 0.06477 或 0.03313 kg，半径为 0.0125 或 0.01 m；
- 初始水平速度约为 0.695604–1.83655 m/s，发射高度固定 0.77 m；
- 可信条件量为质量、半径、水平初速度和发射高度；
- 光电门遮挡时间、光电门到出口距离、斜坡角和释放距离用于审计，均为 `annotated=false`；
- 有 1 条 Case（`parabolic_img_0613`）不含 `ramp_angle` 和 `release_distance`；这是辅助审计字段缺失，不影响四个可信条件量。

### 5.4 单摆

- 35 Case；初始角为 5°、10°、20°、25°、30°；
- 摆长约 0.11–0.305 m，绳长约 0.10–0.295 m，摆球半径固定 0.01 m；
- 四个物理参数均为可信标注；
- 评测关注摆球轨迹、周期、振幅衰减、固定支点和连续周期运动。

### 5.5 匀速圆周运动

- 36 Case；单对象 24 条、双对象 12 条；
- 银色金属块、木块、银色金属块加木块三种对象组成各 12 条；
- 角速度固定 54.55 deg/s，轨道半径为 0.02、0.04、0.06、0.08 m；
- 双对象 Case 才有 `object_2_orbit_radius`；
- `angular_velocity_rad_s` 是换算量，保存为 `annotated=false`；
- 初始角位置被明确设为不标注。

以上范围是 **7.0.0 已收录样本的经验最小值/最大值**，不是物理场景允许值的理论边界。

## 6. View 与泛化语义

### 6.1 View A：finetune/eval

View A 只有两个互斥主分区：`train` 和 `test`，没有 validation 分区。二者完整覆盖 593 条 Case。`id`、`ood`、`mixed` 是每条 Test Case **相对于当前 `view_a.train`** 的分析标签，不是 Case 自身属性。

| Test 因素 | 类别 | 命中数 | 说明 |
| --- | --- | ---: | --- |
| `ball_spec_composition` | `object_composition` | 42 | 训练未见的碰撞球规格组合 |
| `collision_structure` | `interaction_structure` | 34 | 对向入射结构相对训练为新结构 |
| `ball_material` | `object_composition` | 8 | 含玻璃球的新材质组成子集 |
| `moving_object_composition` | `object_composition` | 12 | 圆周运动中的新对象组成 |
| `background` | `environment` | 22 | 斜面新背景；其中 16 条 OOD、6 条 Mixed |

因素计数不是互斥的，同一 Case 可以同时命中多个因素。6 条 Mixed 均为背景 OOD 与斜面角度 holdout 共同变化；`incline_angle` 被记录为 co-varying physical factor。

碰撞划分按重复/近重复组件和物理 signature 做防泄漏约束，审计记录的 Train/Test replicate component overlap 为 0。

### 6.2 View B：direct eval

View B 以 seed 42 将全部 593 条 Case 确定性分成 5 组。它是抽样和报告维度，不表达 ID/OOD，也不决定是否向模型提供物理量。

| Scene | 5 个组的 Case 数 |
| --- | --- |
| 碰撞 | 66 / 66 / 66 / 66 / 66 |
| 斜面 | 19 / 19 / 19 / 19 / 19 |
| 平抛 | 20 / 20 / 19 / 19 / 19 |
| 单摆 | 7 / 7 / 7 / 7 / 7 |
| 圆周 | 8 / 7 / 7 / 7 / 7 |

## 7. 媒体资产

典型 canonical 资产布局如下：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
└── canonical/
    ├── reference.mp4
    └── first_frame.png
```

Case 指向的 `assets.first_frame` 应与其 `assets.reference_video` 解码帧 0 一致。碰撞和平抛还普遍保留逐 Case 的 `source/reference.mov`；斜面、圆周、平抛和补充碰撞批次通过 source archive 保留原始包。

文件名不是运行契约。例如 `parabolic_img_0539` 的同一目录同时保留旧 release 使用的 `reference.mp4` 和当前 v7 使用的 `reference_realtime.mp4`；对应首帧也使用 `first_frame_realtime.png`。必须读取 Case 中的精确资产路径。

### 7.1 锁定资产数量与体积

| 类型 | 文件数 | 锁表声明体积 |
| --- | ---: | ---: |
| PNG 首帧 | 593 | 0.307 GiB |
| MP4 canonical reference | 593 | 15.029 GiB |
| MOV 逐 Case source | 427 | 9.425 GiB |
| ZIP source archive | 4 | 13.326 GiB |
| **合计** | **1,617** | **38.087 GiB** |

`reference_video` 和 `physics_reference_video` 是同一个 MP4 文件的两个角色，不是两份文件。593 条 Case 的所有非空 asset 引用去重后恰好为 1,617 条，与 `assets.lock.json` 精确一致。

### 7.2 Canonical reference 视频画像

下表由本次使用 `ffprobe` 扫描 593 个 MP4 得到，不是 release descriptor 中的静态字段。

| Scene | 分辨率（数量） | 名义 `r_frame_rate`（数量） | 帧数范围 / 均值 | 时长范围 / 均值 |
| --- | --- | --- | --- | --- |
| 碰撞 | 832×480（298）；1920×1080（32） | 240（330） | 103–1728 / 595.6 | 0.429–8.092 s / 2.536 s |
| 斜面 | 1920×1080（95） | 240（95） | 452–1841 / 771.5 | 1.887–7.679 s / 3.219 s |
| 平抛 | 480×960（97） | 240（97） | 65–105 / 91.4 | 0.271–0.438 s / 0.381 s |
| 单摆 | 1080×1920（35） | 30（17）；120（18） | 158–834 / 597.7 | 4.867–27.800 s / 11.864 s |
| 圆周 | 1080×1920（36） | 60（36） | 818–1076 / 953.4 | 13.633–17.933 s / 15.891 s |

全部 reference MP4 总时长约 2,166.887 秒，即 36.115 分钟。当前 593 条 Case 的 `encoded_to_physical_speed` 均为 1.0；`time_scale` 仍保留 `real_time` 与 `source_timing` 两种来源语义标签。

编码并不完全同构：碰撞的 298 个低分辨率视频为 H.264、32 个高分辨率视频为 HEVC；斜面为 HEVC；平抛为 H.264；单摆包含 16 个 H.264 和 19 个 HEVC；圆周运动为 10-bit HEVC。碰撞、斜面和平抛没有音频，35 个单摆和 36 个圆周视频带 AAC 音频。表中的 240 FPS 是名义 `r_frame_rate`；受容器时间戳影响，碰撞实际 `avg_frame_rate` 约为 212.586–240，斜面约为 239.470–239.788。下游统一抽帧、时序建模或音频处理时应显式处理这些差异。

### 7.3 目录命名

媒体目录采用 `assets/<scene_id>/<descriptive_physical_case_directory>/`。目录名通常编码主要结构化物理量，不编码背景、颜色、机位或采集环境：

- 小数点常写作 `p`，负号写作 `neg`；
- 单位常直接进入目录名，如 `mm`、`g`、`mps`、`deg`、`dps`；
- 末尾使用 `img####` 或短 hash 避免冲突；
- `case_id` 才是稳定身份，目录名只是人类可读的资产位置；二者绑定见 `asset_directory_mapping.json`。

例如：

```text
collision_n2-opposed_b1-steelM-d20mm-m33p13g-v0p6004mps_...
incline_a32deg_m87p68g_mu0p463_img0341
projectile_steelM-d20mm-m33p13g_h0p77m_v1p199mps_img0522
pendulum_l210mm_ls200mm_r10mm_a5deg_id1a03a76e
circular_w54p55dps_r1-20mm_img0370
```

不要用实际目录数推断 Case 数。例如碰撞目录中还保留 32 个旧路径，正式 Case 集合只能由 descriptor、`cases.jsonl` 和 asset lock 确定。

## 8. Provenance、质量控制与完整性

### 8.1 Provenance 内容

- `source_docs/` 保存原始 XLSX/TXT、规范化标注和球规格资料；
- `imports/` 保存逐条导入、排除和去重审计；
- `alignment/` 保存斜面启动帧、碰撞入场、光电门清理、裁剪和 native timing 审核；
- Case 的 `provenance.source_locator` 在可用时记录 archive/member、工作簿和来源成员；
- `assets.lock.json` 为每个当前资产保存大小和 SHA-256；
- `release.json` 进一步冻结整个 Dataset 与资产集合 digest。

导入审计记录显示：

- 5.0.0 新增 298 个不重复的碰撞 Case 和 97 个平抛 Case；
- 平抛来源中 31 个没有独立物理标注的视频未导入；
- 碰撞工作簿中 107 行没有对应视频，因此未导入；
- 395 个新增碰撞/平抛 Case 均有首末帧视觉审核记录。

### 8.2 本次只读一致性检查

- 593 个 `case_id` 全部唯一；
- View A 与 View B 均完整、互斥覆盖全部 Case，且没有未知 Case ID；
- 593 条 prompt 全部非空；
- Case 的非空资产引用集合与 1,617 条 asset lock 路径完全一致；
- 官方 loader 的 `validate-dataset --check-assets` 检查通过：593 Case、5 Scene，digest 与 `release.json` 一致；
- 本次没有重新读取约 38 GiB 媒体计算全部 SHA-256，因此文件哈希在本文中是锁表声明值，不是本次复算结果。

## 9. Release 演进

| Release | Dataset ID | Case schema | Case 数 | Case 中的 Scene | 锁定资产 | 主要变化 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 1.0.0 | 早期快照，无 descriptor | 1.0 | 83 | 碰撞 32、自由落体 11、单摆 40 | 无 lock | 仅 Case 与早期 View |
| 2.0.0 | `physics_video_three_scene_v2` | 2.0 | 83 | 同上 | 204 | 引入 descriptor、Scene catalog、asset lock 和 release manifest |
| 3.0.0 | `physics_video_five_scene_v3` | 2.0 | 214 | 新增斜面 95、圆周 36 | 468 | 扩展到五个有 Case 的场景 |
| 4.0.0 | `physics_video_five_scene_v4` | 3.0 | 214 | 与 3.0.0 相同 | 468 | 固化 Case-owned `text.prompt` |
| 5.0.0 | `physics_video_six_scene_v5` | 3.0 | 609 | 新增平抛 97、碰撞补充 298 | 1,655 | 扩展至六场景 |
| 5.1.0 | `physics_video_six_scene_v5p1` | 3.0 | 604 | 六场景 | 1,650 | 移除 5 条首帧不匹配的合成单摆 OOD；统一球规格、物理目录和划分 |
| 6.0.0 | `physics_video_six_scene_v6` | 4.0 | 604 | 六场景 | 1,650 | 元数据迁移；View A 改为完整 Train/Test，ID/OOD/Mixed 下沉到 Test annotation |
| 7.0.0 | `physics_video_five_scene_v7` | 4.0 | 593 | 当前五场景 | 1,617 | 移除自由落体 11 条，其余五场景的 View 成员身份保持 |
| 8.0.0 | `physics_video_six_scene_v8` | 4.0 | 799 | 六场景 | 2,032 | 新增补充单摆和推水瓶，View A 全局改为 Train/ID-Test |
| 9.0.0 | `physics_video_six_scene_v9` | 4.0 | 799 | 六场景 | 5,239 | 新增逐Case主体mask资产与清单 |
| 10.0.0 | `physics_video_six_scene_v10` | 4.0 | 799 | 六场景 | 6,038 | 新增Case-local `physics.json`并精简运行时Release边界 |
| 11.0.0 | `physics_video_six_scene_v11` | 5.0 | 799 | 六场景 | 6,038 | 当前默认；新增symbol、非负大小语义、符号prompt与`physics.v11.json` |

7.0.0 在 2026-08-03 又做过一次同版本号内修复：

- 185 个异径球碰撞 prompt 去除不够严谨的 `central`/`head-on` 措辞；
- `parabolic_img_0539` 改用 7.0.0 专用的 240 fps 真实物理时标视频和对应首帧；
- 当前 digest 已包含上述修复。

因此复现实验时应同时记录 `dataset_id` 和完整 `dataset_digest`，不能只记录字符串 `7.0.0`。历史 release 只有在已有结果明确记录了旧 Dataset ID/digest 时才应显式加载。

## 10. 使用注意事项与当前缺口

1. **本文是历史快照**：正文统计对应7.0.0，不代表当前release。涉及当前数量和入口时，应以`datasets/README.md`及显式选择的descriptor、release、Case和View文件为准。
2. **当前没有 validation split**：View A 只有 Train/Test；如训练流程需要验证集，应在运行配置中另行设计，不能就地修改正式 Dataset。
3. **当前没有正式 V2V 输入资产**：没有 `assets.input_video`。`reference_video`、source 或 physics reference 不得冒充 V2V conditioning input。
4. **模板文本多样性有限**：593 条 Case 只有 11 个不同英文 prompt；这更适合过程语义控制，不适合评估自然语言表达多样性。
5. **部分元数据有历史缺失**：旧的 32 个碰撞 Case 没有 `appearance.background`；35 个单摆没有 `provenance.source_locator`；平抛有 1 条缺少两个辅助参数；所有 mask 为空。
6. **碰撞措辞需谨慎**：185 条异径球 prompt 已主动避免 `central/head-on`，但 `scenes/collision_1d.json` 仍保留 “central” 约束；讨论几何精确性时应结合修复审计，不宜过度解读。
7. **禁止把派生物写回 Dataset**：重采样/resize 视频、模型输入首帧、embedding、latent、训练 metadata 和 run 输出应进入内容寻址 cache 或 `runs_v2/`。
8. **机构、作者与许可信息缺失**：在数据集目录及仓库内未发现明确的 Dataset 机构归属、维护人/联系信息、Citation 或 LICENSE。现有 provenance 能说明采集批次和来源文件，但不能替代法律授权。对外分发前建议补齐 Dataset Card、机构/作者、引用格式、使用许可和隐私/伦理说明。
9. **平抛保留了显式质量标记**：77 条带 `source_mass_material_consistency_unverified`，表示源表中的球规格/材质一致性仍需追溯确认；`parabolic_img_0525` 另带 `track_coverage_below_0.9`。这些 Case 仍在正式集内，下游筛选和报告不应忽略该 provenance。
10. **画幅和编码差异较大**：既有横屏也有竖屏、小目标、黑边、8-bit/10-bit 和有/无音频。统一 resize/crop 时需避免丢失小球、让黑边主导指标或误把音频存在性当作场景标签。

## 11. 推荐加载与验证方式

代码应使用项目定义的当前 Dataset 常量，命令行应显式传入 descriptor。快速检查命令：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/11.0.0/dataset.json \
  --check-assets
```

发布或归档前再执行逐文件哈希检查：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/releases/11.0.0/dataset.json \
  --check-asset-hashes
```

后者会完整读取当前6,038个锁定资产，耗时明显更长。

## 12. 主要依据

- [Dataset 根说明](README.md)
- [Dataset 根说明](README.md)
- [Canonical Assets 说明](assets/README.md)
- [Release 选择规则](releases/README.md)
- [7.0.0 说明](releases/7.0.0/README.md)
- [7.0.0 descriptor](releases/7.0.0/dataset.json)
- [7.0.0 release manifest](releases/7.0.0/release.json)
- [7.0.0 Cases](releases/7.0.0/cases.jsonl)
- [View A](releases/7.0.0/views/view_a.json) / [View B](releases/7.0.0/views/view_b.json)
- [划分审计](releases/7.0.0/split_audit.json)
- [迁移审计](releases/7.0.0/migration_audit.json)
- [文本/媒体修复审计](releases/7.0.0/text_video_alignment_repair.json)
- [当前数据集规范](../docs/DATASET.md)
- [数据导入规范](../docs/DATASET_INGESTION.md)
