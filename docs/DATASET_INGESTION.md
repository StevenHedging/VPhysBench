# 原始视频与 XLSX 标注导入规范

## 1. 目的与适用范围

本文是向 Physics Video Benchmark 导入数据的规范入口。适用于：

- 已有 scene 的新增试次；
- 全新实验情景；
- 输入为一个或多个原始视频、ZIP/目录，以及字段可能不规范的 XLSX 标注表；
- 执行者为 AI 助手或真人。

导入工作的目标不是“尽量把文件放进去”，而是建立可审计的一一对应关系：

```text
一个有效 Case
= 一个明确的原始试次
+ 一份能够理解且能够归一化的物理标注
+ 一个清洗并人工/视觉复核的视频
+ 一条不泄露数值的过程描述
+ 可追溯的来源记录
+ 一份相对于当前 View A train 的泛化标签
```

任何一项不能可靠建立时，不得猜测。无法找到对应物理标注的视频必须舍弃；完全无法
理解的物理标注必须暂停该条导入并询问数据提供者。

## 2. 必须先理解的三个数据层

### 2.1 Case：事实层

Case只记录与该试次本身有关的事实：

- `physics`：有用、具体、可解释的物理量；
- `appearance`：背景、环境、颜色、材质、实验形式、对象组成、机位和采集批次等；
- `text`：简洁且无歧义的物理过程描述；
- `assets`、`temporal`、`alignment`和`provenance`：媒体与来源；
- `has_real_reference_video`：是否存在同一试次的真实参考视频。

Case不得记录`train/test`或ID/OOD。ID/OOD永远相对于某个训练集定义，不是样本的固有
属性。

### 2.2 View：划分与泛化解释层

- View A用于`finetune_eval`，主划分只有`train/test`；
- View A的每条test Case另有`id/ood/mixed`和因素列表；
- View B用于`direct_eval`，完整覆盖Case并按冻结seed分组，不表达ID/OOD。

### 2.3 Task：运行与计分层

Task选择Dataset、View、scene、test范围、seed和评估协议。Task不决定如何使用物理量，
也不重新解释XLSX字段。官方finetune Task以全部test为主评测集；ID/OOD/mixed和具体
factor只作为必须报告的诊断维度。

## 3. 不可违反的导入原则

1. 原始ZIP/XLSX按字节保留；正式release只新增，不覆盖历史release。
2. 视频和标注必须逐条建立确定映射；不靠文件排序、模糊相邻行或主观猜测配对。
3. 物理单位必须归一化，同时保留原字段、原值、换算依据和修正记录。
4. 背景、颜色、环境、机位和实验形式不得伪装成结构化物理量。
5. canonical视频只做必要的时间窗口和空间裁剪；不改FPS，不抽帧，不改变窗口内帧数。
6. 相同原始试次、相同媒体或近重复片段不得跨train/test。
7. 对已有scene先查重；确定重复的数据不得再次导入。
8. 发布前必须同时通过机器校验和视觉验收，不能用旧日志替代本次检查。
9. 导入脚本、标准化中间表、排除列表和复核结果都要进入provenance。

## 4. 标准工作流与停机点

### 阶段0：建立只读清单

在解压或转码前记录：

- 来源路径、文件大小、SHA-256；
- ZIP成员完整列表；
- XLSX的sheet、表头、合并单元格、公式和非空行；
- 每个视频的容器、编码、宽高、帧数、nominal FPS、duration和旋转元数据；
- 当前最新Dataset digest与目标scene。

解压必须防止绝对路径和`../`路径穿越。原始包放入
`datasets/physics_video/assets/source_archives/`，XLSX原件放入
`datasets/physics_video/provenance/source_docs/<import_id>/`。

阶段0不能修改正式资产。

### 阶段1：理解XLSX并生成标准化中间表

不要直接从XLSX写`cases.jsonl`。先生成一份标准化表，并为每个原始字段形成字段决策表：

| 原始字段 | 示例 | 归类 | 目标位置 | 处理 |
| --- | --- | --- | --- | --- |
| 小球质量/g | 33.13 | 直接物理量 | `physics.ball_mass` | 转为kg |
| 小球直径/cm | 2.0 | 直接物理量 | `physics.ball_radius` | 除以2并转为m |
| 遮光时间/ms | 8.2 | 辅助测量 | `physics.photogate_block_time` | 转为s，通常`annotated=false` |
| 初速度 | 由直径/时间计算 | 派生物理量 | `physics.initial_velocity` | 记录公式和来源可信度 |
| 背景颜色 | 黑色 | 环境 | `appearance.background` | 可作View factor |
| 球数 | 3 | 实验形式 | `appearance.object_count` | 可作结构OOD factor |
| 有初速度球数 | 2 | 实验形式 | `appearance.collision_structure` | 规范枚举 |
| 视频文件名 | IMG_0123.MOV | 来源键 | `provenance.source_locator` | 用于确定映射 |
| 备注 | “本次碰歪” | 质量信息 | ingest audit | 决定排除或复核 |

字段分成四类，而不是只分成两类：

1. **直接物理量**：仪器或人工明确测得的质量、半径、长度、角度、时间、速度等；
2. **派生/辅助物理量**：由公式换算或只用于校准的量；仍可放入`physics`，但是否
   `annotated=true`必须有依据；
3. **非物理情景元数据**：背景、颜色、材质标签、实验形式、对象组成、视角和批次；
   放入`appearance`，可用于View划分；
4. **来源与质量控制信息**：文件名、行号、备注、操作者和异常；放入provenance/audit。

“材质”需要逐scene判断：若已有质量、半径等具体物理量，`steel/glass`标签本身通常是
对象组成或外观因素；若材质直接决定评测所需且没有其它等价物理量，则应先扩充scene
定义，不能由导入者临时猜测。

标准化记录至少包含：

```json
{
  "source_workbook": "原始表.xlsx",
  "sheet": "Sheet1",
  "source_row": 17,
  "trial_id": "T017",
  "video_member": "实验/IMG_0123.MOV",
  "normalized_physics": {},
  "normalized_appearance": {},
  "conversion_notes": [],
  "annotation_status": "accepted|needs_clarification|rejected"
}
```

以下情况必须询问数据提供者，不能自行修饰：

- 同一字段在不同sheet中含义冲突；
- 单位缺失且从数值范围无法唯一判断；
- 球的左右/编号顺序无法与视频主体对应；
- “初速度”“高度”“长度”等存在多个合理物理定义；
- 标注与视频明显矛盾，但无法判断哪一方正确；
- 公式、缩写或实验装置含义完全无法理解。

### 阶段2：建立视频—标注一一对应

优先级从高到低：

1. XLSX中显式视频成员路径或稳定trial ID；
2. 同时可由文件名、采集时间和实验条件交叉验证的唯一映射；
3. 经逐视频视觉核验后仍唯一的映射。

禁止仅按目录排序和XLSX行序配对。每条映射写入alignment/ingest audit，并记录使用的
证据。映射为零或多于一个候选时，该条进入排除列表；不能为了提高导入数量而猜一个。

### 阶段3：对已有scene查重

至少执行四层检查：

1. 原始文件SHA-256完全相同；
2. 来源archive/member、trial ID或采集时间相同；
3. 视频感知近重复：抽取多个时间点的图像指纹并检查时间偏移后的重复片段；
4. 相同标注与高度相似画面人工复核。

完全重复或同一长视频的重复切片只保留一条。仅物理参数相同但属于独立试次的视频不是
重复，可以保留；它们仍须作为replicate group参与train/test泄漏检查。

### 阶段4：视频清洗与事件对齐

先为scene写清楚“canonical首帧事件”。若是新scene且事件未定义，必须先补scene规范，
不能由剪辑者临场决定。

通用要求：

- 首帧已经进入目标物理过程，不能包含无关准备动作；
- 所有物理主体必须完整出现在首帧；
- 会泄露装置或干扰生成的光电门、手、线缆、标尺等按scene规则裁掉；
- 裁剪后主体整个有效运动区间尽可能保留，不能过度裁到过程未完成；
- 画幅要适合主体轨迹，但不得以改变物理过程为代价；
- 空间裁剪框、源起止帧、源视频probe和输出probe全部写入alignment audit；
- canonical `first_frame.png`必须由最终reference视频的第0帧解码得到并逐像素/哈希核验。

当前特别规则：

- **碰撞**：第0帧对应入射球刚穿过光电门；所有球都出现；光电门不出现；视频需覆盖
  碰撞及足够的碰后运动；
- **平抛**：第0帧对应球刚穿过光电门且球可见；裁去光电门，必要时同步裁去上方空画面
  以维持合理长宽比；保留完整可见抛物轨迹，不能只剩几帧；
- **斜面**：第0帧为经过视觉复核的即将开始或刚开始下滑时刻；
- 其它scene以其scene config和已有清洗约定为准。

时间处理的硬规则：

```text
输出FPS = 源视频nominal FPS
输出帧数 = 选定源窗口内的全部解码帧
不抽帧，不补帧，不插帧，不为4n+1/固定FPS/最大帧数做适配
```

慢动作素材的encoded time与physical time关系写入`temporal`，不能通过改FPS偷偷修正。
模型侧的帧率、帧数和分辨率适配只能进入Baseline adapter/cache/run。

最终canonical reference确定后，还必须冻结：

```text
temporal.target_physical_duration_s
= (frame_count - 1) / encoded_fps / encoded_to_physical_speed
```

它是Baseline的输出时长目标，不是新的物理参数标注，也不能通过更改canonical视频FPS
或抽帧来凑整。reference、裁剪窗口或慢动作比例发生变化时必须重新计算并复核该值。

每条视频至少视觉检查首帧、中间运动、末帧；事件对齐要求高的scene应逐条检查。复核状态
只有在本次真正看过最终媒体后才能写`visually_verified`。

### 阶段5：编写文本描述

`case.text.prompt`与首帧共同说明“接下来发生什么”。文本必须：

- 无歧义地描述可见物理过程；
- 在需要时说明主体数量、哪个主体初始运动/静止、运动方向和交互顺序；
- 足够简洁，不写实验报告或物理结论；
- 不包含任何具体数值、单位、公式或结构化物理参数；
- 不提背景、颜色、材质外观、机位、视角、裁剪、光电门、分辨率和采集设备；
- 不描述视频中不可见、也无法由标注可靠确定的结果。

碰撞不能用一个模板掩盖所有情景。例如“两球中右球向左运动、左球初始静止”和“两球
相向运动”是不同过程，必须分别准确描述。描述实验形式不等于泄露物理数值；它用于消除
过程歧义。

发布前自动扫描数字、单位和禁用视觉词，再由人工/AI逐条结合首帧与视频复核。自动扫描
不能代替语义检查。

### 阶段6：生成Case与资产目录

推荐目录：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
├── source/                 # 可选的原始成员硬链接/副本
└── canonical/
    ├── reference.mp4
    └── first_frame.png
```

目录名简要编码主要结构化物理量和唯一身份后缀，用于人工区分；不得编码背景、颜色、
视角、实验室或其它环境信息。`case_id`一经发布保持稳定，不把可变路径当身份。

每个物理量使用：

```json
{"value": 0.03313, "unit": "kg", "annotated": true}
```

- 统一使用scene约定单位，优先SI；
- `annotated=true`仅用于可信、可作为模型条件的物理量；
- 派生量、辅助装置量或可信度不足但仍值得审计的量设为`annotated=false`；
- 不确定的值不能用`0`代替缺失；若scene要求该量而无法得到，整条Case不进入正式集；
- 多主体字段的编号必须与首帧从左到右/scene定义一致，并在provenance中记录映射依据；
- 规格球等复用对象必须先查权威catalog，不能为同一规格创建多个别名。

### 阶段7：设计View A的train/test与泛化标签

先确定不可拆分的数据组：完全重复、同一原始试次的多片段、同一事件的多机位，以及会
造成目标泄漏的近重复必须整体进入train或test。物理signature是否必须成组取决于评测
目标：

- 数值ID测试中，相同完整物理signature通常整体划分，避免把同条件复刻当泛化；
- 环境/外观OOD控制实验可以有意让train/test共享物理signature，以隔离OOD因素；
- 无论哪种设计，同一媒体或同一试次都不能跨界。

Test标签定义：

| regime | 判定 |
| --- | --- |
| `id` | 情景因素位于train已覆盖域；可以是独立试次和分布内连续数值holdout |
| `ood` | 一个或多个明确因素未在train出现，且控制因素由train覆盖 |
| `mixed` | OOD因素和其它held-out因素共同变化，无法隔离归因 |

OOD factor记录`name/category`，category只能是：

- `physical_parameter`：超出训练支持域的物理参数或组合；
- `object_composition`：新对象类型、材质标签或组成；
- `interaction_structure`：球数、入射角色数、碰撞/交互结构；
- `appearance`：颜色、纹理等视觉变化；
- `environment`：背景、光照、轨道/支撑环境；
- `acquisition`：机位、视角或采集方式。

`appearance/environment/acquisition`的OOD分数是视觉鲁棒性诊断，不应包装成物理泛化。

#### 已有scene的新数据

1. 旧release只作为历史结果的不可变证据，不得直接覆盖；但它的train/test成员身份不
   自动约束新release；
2. 合并全部有效旧Case和新增Case后，重新统计每个replicate group、stratum与factor的
   数量，先判断旧比例是否仍然合理；
3. 先处理重复/replicate group，同组样本不得跨train/test；
4. 若新增批次使旧Train过窄、Test严重失衡或OOD因素失去可归因性，应在新Dataset ID/
   digest下重新划分全部Case，不能把所有新增数据机械地塞入Test；
5. 为test优先构造可归因的OOD，同时保留数量足够且覆盖训练支持域的Test-ID；无法隔离
   的样本标为mixed，不再排除；
6. 记录从旧release到新release的每条划分变化、比例、因素支持域和理由；历史结果不得
   与新划分结果直接混报。

#### 全新scene

先新增scene config，至少定义：

- 物理主体、过程边界和canonical首帧事件；
- 必需/可选物理量、单位、编号规则和约束；
- 非物理情景字段与候选generalization factors；
- prompt应描述的过程信息；
- evaluator所需reference和验收条件。

若数据量或factor覆盖不足，允许该scene暂时只有test-ID；不得合成或误标OOD来填表。

### 阶段8：构建View B、asset lock与新release

- View B完整覆盖所有Case，使用冻结seed确定性分组；
- `assets.lock.json`包含所有Case引用资产的相对路径、大小和SHA-256；
- `release.json`冻结Dataset digest和资产集合digest；
- 新release目录在全部校验通过前由staging原子生成；
- 旧release、旧Case和旧媒体不得就地覆盖。

## 5. 必须保留的审计产物

每次导入至少交付：

| 产物 | 内容 |
| --- | --- |
| source inventory | ZIP/XLSX/video清单、hash和probe |
| normalized annotations | 逐行标准化结果与单位换算 |
| field decision table | 每个XLSX字段的归类和目标字段 |
| video-annotation mapping | 一一对应证据 |
| exclusion audit | 所有未导入视频/行及原因 |
| duplicate audit | 与当前Dataset和本批次的查重结果 |
| alignment audit | 源起止帧、crop、probe和视觉复核状态 |
| prompt audit | 自动禁词检查与语义复核 |
| split audit | group、train/test、regime、factor和理由 |
| migration audit | 基础release、数量变化、digest与媒体改动量 |

推荐排除原因码：

```text
missing_annotation
ambiguous_video_annotation_match
unintelligible_physics_annotation
annotation_video_contradiction
duplicate_existing_case
duplicate_within_import
invalid_or_incomplete_physical_event
unrecoverable_video_quality
failed_visual_alignment_review
```

舍弃不等于静默删除：排除项必须保留来源定位和原因，但不能进入Case主表。

## 6. 发布验收清单

### 6.1 机器校验

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/<version>/dataset.json \
  --check-assets

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/<version>/dataset.json \
  --check-asset-hashes
```

还必须运行：

- Case/scene/View/Task schema测试；
- case ID、资产路径和source locator唯一性检查；
- train/test媒体hash与近重复泄漏检查；
- first frame与reference第0帧一致性检查；
- reference与source窗口的FPS、帧数、duration关系检查；
- `temporal.target_physical_duration_s`与reference最后一帧物理时间一致；
- prompt数值/单位/背景/颜色/视角禁词检查；
- `physics`中背景/颜色/环境字段检查；
- View A完整覆盖、互斥、annotation完整性和factor类别检查；
- release和asset digest复算。

### 6.2 视觉验收

逐条或按风险分层高覆盖复核：

- 首帧事件正确；
- 所有主体可见且编号可对应；
- 不应出现的装置已经裁掉；
- 运动过程没有被过早截断；
- 画面与文本描述一致；
- 画面与物理标注一致；
- 最终媒体保持源FPS和窗口全部帧。

碰撞、平抛以及任何依赖精确事件首帧的新增数据应逐条复核，不使用低比例抽样。

## 7. AI助手或真人的最终交付格式

完成导入后应明确报告：

1. 收到多少视频和XLSX行；
2. 成功建立多少一一对应；
3. 导入、排除、重复各多少，逐原因统计；
4. 哪些标注经过单位换算、catalog修正或人工澄清；
5. 视频清洗和视觉复核覆盖率；
6. 各scene的train/test及ID/OOD/mixed数量；
7. 新增OOD factor及类别；
8. Dataset、asset和Task digest；
9. 是否修改或复制过媒体字节；
10. 尚未解决的风险和需要数据提供者回答的问题。

如果仍有不理解的物理标注，最终状态只能是“部分完成/等待澄清”，不能把猜测包装成
已完成导入。
