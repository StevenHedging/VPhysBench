# 全场景开放世界评估 v6

日期：2026-07-30

> 历史快照：文中的全局外置可视化路径记录当时实现。当前实现统一写入所属
> AtomicRun/reevaluation 的 `evaluation/visualizations/`，不要照搬旧路径。

## 1. 结论与边界

`scene_default_v6` 已将单摆、自由落体、斜面下滑和匀速圆周运动迁移到公共
`open_world_v2/2.0`；一维碰撞逐配置复用已经审计的 evaluator 2.2。v6 的目标不是只
跟踪一个最佳 mask，而是持续核对 manifest 声明的物理身份、所有 prediction 侧可信
对象及合法生命周期，使新增、复制、消失、replacement、ID switch 和 overflow 都有
有限、可解释的惩罚。

本次文档审阅时的协议身份为：

```text
protocol_id          = scene_default_v6
protocol_fingerprint = 095f40ab72f8e42dadba104bb0c69dce7314153923d3c75d3a65ecef405288ff
```

Evaluator 身份为：

| scene | evaluator | version | fingerprint |
| --- | --- | ---: | --- |
| pendulum | `pendulum_open_world_structure` | 2.0 | `696605bc08595fc820d2a174a36f47768e862f90522dc12576cb463d841b8dfd` |
| free_fall | `free_fall_open_world` | 2.0 | `25872d31ff7ad21590ee5ee48395aa2260cf265853355be3a8c32bd941a7edc6` |
| inclined_plane_slide | `inclined_plane_open_world` | 2.0 | `31a3df4ad6d1983489920f8c0d3e61cf013b1a8e1d16b3fec6a3b6b1e4f7094c` |
| uniform_circular_motion | `uniform_circular_motion_open_world` | 2.0 | `0fe16b1c21a51c19f13e27327f61bdeb365146827753536b7d931bc09a019666` |
| collision_1d | `collision_1d_open_world_nbody` | 2.2 | `686b705e426f43d29acdc745a95b765fe249bd48628c86f5167f40f851caa156` |

这些 fingerprint 绑定完整 evaluator config；设备或配置覆盖会生成不同 identity。
形成 release commit 后应以该 commit 的 protocol snapshot 为准。

v6 是 shadow protocol，不覆盖当前官方 `scene_default_v3`，也不覆盖 v4/v5
reevaluation。所有 baseline 必须用同一个 v6 fingerprint 重评后才能形成 v6
leaderboard。

## 2. 冻结回归

v6 没有修改 v3/v4/v5 文件，协议回归常量为：

```text
scene_default_v3 = 14dac014a9311ce32be65052f88875e9a9432d64f044dfeb7cc5eceb9adc41c6
scene_default_v4 = 54cefd0a75dc927e783ba7bcd8c75719b6c8b619fc07dce055eb50f94ca60532
scene_default_v5 = 93703d6afdf8bbdea86b69d8d8a427653c68030bd7660f3801341e3b2b6209f6
collision 2.2  = 686b705e426f43d29acdc745a95b765fe249bd48628c86f5167f40f851caa156
```

`tests/test_evaluation_protocol_v6.py` 还断言 v6 的完整 collision 配置等于 v5，注册器
仍解析到 evaluator 2.2。因此碰撞 r7/r8 数值和既有可视化协议不受本轮影响。

## 3. 公共 `open_world_v2`

实现位置：

```text
src/physbench/evaluation/common/entities/v2.py
src/physbench/evaluation/common/artifacts/open_world_v2.py
```

每个 expected entity 都有三条相互独立的监督通道：

```text
existence     何时应存在、何时可审计基数
localization  何时有合法 reference/condition 位置
association   何时可审计持久身份
```

Expected timeline 只由 manifest、condition、reference 和版本化 scene policy 构造，
不读取 prediction 来缩短分母。`persistent`、`may_enter`、`may_exit` 和
`may_enter_and_exit` 都要求合法的连续 lifecycle；prediction 自身消失不能宣告 exit。

Prediction 侧执行：

```text
condition-frozen semantic identity
+ directed expected-object channel
+ scene-specific residual discovery
→ causal persistent tracks
→ evidence-tiered rectangular Hungarian
→ explicit null assignment
```

低于 `minimum_match_position_similarity=0.1` 的边被拒绝，结果同时是 expected
missing 和 prediction extra，不能让远处 replacement 接管原 ID。每帧 assignment
保持一对一并检查逐 entity/track exposure 守恒。超过 track/candidate 上限的对象进入
`__overflow__` false exposure。

真实秒 exposure 定义为：

```text
R = expected entity-time exposure
P = all formal prediction-track exposure, including residual/overflow
M = matched exposure

PresenceDetA = M / (R + P - M)
AssA = matched-exposure-weighted pair association IoU
raw_integrity_gate = PresenceDetA × AssA
deletion_ceiling = (M / R)^3
integrity_gate = min(raw_integrity_gate, deletion_ceiling)
```

位置使用 reference/condition 尺度的连续 Cauchy 核；SoftDetA 与 GOSPA 分解定位、
missed 和 false exposure，但不重复进入正式 gate。最终统一为：

```text
content = weighted_geometric_mean(scene-specific components)
case_score = integrity_gate × content
```

Missing 还会降低 scene physics 的完整 reference evidence coverage。这是
anti-abstention：完整性层回答“对象集合是否完整”，physics 层回答“缺失时段是否提供
正确物理证据”。三次方 recall ceiling 进一步保证删除错误 replacement 帧不能提高
分数；该规则仅存在于新增的 `open_world_v2`，冻结的 collision 2.2 继续使用原始
`PresenceDetA × AssA`。

## 4. 四个新 scene adapter

### 4.1 单摆

- `PendulumStructureSpec/1.0` 从当前 condition 冻结 pivot、string、bob、半径与摆长；
- condition prompt 驱动 SAM2，bob 与 string/support 分离；
- Hough circle、string segment 和 condition-frame change 发现第二 bob；
- support corridor 排除支架，可信 residual 保留为正式 prediction track；
- directed SAM2 只有连续两帧违反 condition 外貌/尺度契约后才切成 replacement，
  避免单帧高光误杀，同时禁止漂移复制体接管冻结 ID；
- 位于 frozen pivot–bob 主绳中段、贴近中心线且缺少独立 body-change 证据的 Hough
  圆只记 rejection，不再成为幽灵 bob；
- bob missing 同时降低 integrity 与摆动 physics evidence coverage；
- pivot–bob 中段 string occupancy 检查断绳；
- 主绳 corridor 外与主绳/pivot 连通的细长 residual 检查无 bob 分叉绳，按真实秒
  exposure 平滑扣分；支架、pivot 横杆、matched bob 与零散纹理显式排除；
- extra bob exposure 另形成 branch penalty；
- same-case 完整主体 IoU 比较 bob+string 与全部正式 prediction union；
- physics-parent OOD 只在首帧比较当前 condition 主体，未来 parent pixels 不可用。

内容权重为 physics `0.45`、shape `0.15`、appearance `0.20`、topology `0.20`。

### 4.2 自由落体与斜面下滑

两者共享 `rigid_body_open_world.py`：

- condition-anchored SAM2 追踪 manifest ID；
- condition difference、双向 temporal difference 和 compact-shape proposal 发现
  静止/运动复制体；
- residual 必须满足 condition 外貌与主体相对尺寸契约；单源弱候选保持 tentative，
  不伪造多源证据；
- 仅有 condition-difference 且严格触碰画布边界的候选作为 apparatus/background
  rejection 留在审计中，不形成 formal exposure；一旦还有 temporal 或 compact
  支持即正常保留；
- 斜面 ROI 外的复制滑块仍可由全画布强外貌、尺度和形状通道发现，不能靠远离冻结轴
  隐藏；
- 与 condition 颜色不兼容的 directed 对象标为 replacement，不能填补 missing；
- prediction union 保留全部正式 residual；
- reference 证实终端进度和持续尾段缺失后才允许 `may_exit`；短 detector gap 不算
  exit，合法退出后的重现是 extra；
- observer failure 不能退回 directed-only，而是 fail-closed。

自由落体冻结竖直重力轴；scene state 比较竖直轨迹、归一化加速度、impact time、
横漂和单调性。斜面从 same-case reference 或当前 OOD condition apparatus 冻结斜面
轴；prediction 不能自拟合；scene state 比较沿面轨迹、加速度、descent time、法向
接触、单调性和姿态。两者内容权重均为 physics `0.55`、shape `0.20`、
appearance `0.25`。

### 4.3 匀速圆周运动

- 绿色圆盘是 apparatus，先移除并腐蚀；
- 盘内所有合法 connected components 都保留，不再按 expected N 截断；
- 超过安全候选/track 上限的对象转成 overflow；
- condition/初始可靠窗口按外貌、半径和相位冻结 ID；
- 大 Lab 外观跳变拆成 ID 断点，避免颜色/材质交换被一个 track 吞掉；
- 位置使用 disk-normalized relative polar distance，圆心附近回退 Cartesian；
- scene state 比较角轨迹、角速度、轨道几何和匀速性。

内容权重为 orbit physics `0.55`、shape `0.15`、appearance `0.30`。

## 5. OOD 与 physics parent

Physics parent 只监督规范化动力学，不能提供 OOD 的背景、底座、对象外貌或未来 raw
pixel position。公共 timeline 在代码层拒绝 physics-parent 的 `t>0` localization。

OOD 的以下信息来自当前 Case condition：

- expected entity count 与持久身份；
- condition appearance、尺度和初始位置；
- pendulum pivot–string–bob topology；
- free-fall 重力轴、incline apparatus 轴与 circular disk anchor。

Condition ROI 必须由 condition 图自身的结构检测或分割产生，禁止继续用 prediction
mask 定义“GT”区域。无法从 condition 可靠冻结必要 anchor 是 reference-side
`unavailable`，不能猜测对应关系。

## 6. Fail-closed 与有限输出

Prediction 侧短视频、坏帧、SAM2、residual discovery、identity freeze、comparison
或 scene physics 失败都会产生显式 degradation。公共失败结果：

- 保留全部 expected missing exposure；
- 加入 `__observer_failure__` false exposure；
- `integrity_gate=0`；
- Case score 为有限 `[0,1]` 低分，通常为 0；
- per-frame audit 仍写 missing、extra 和 failure reason。

Reference 视频、condition asset、manifest 或 reference observer 失败返回
`unavailable`。只有未捕获的代码 invariant 才是 evaluator `error`。可视化写入失败
只生成 `status=failed` manifest，不改变已经计算的分数。

## 7. 产物

每个新 v6 evaluator 都保存：

```text
per_frame.csv
physical_subject_iou_curve.png
entity_position_curve.png
object_cardinality_timeline.png
scene-specific trajectory curve / JSON audit
open_world_v2_artifact_manifest.json
```

完整主体 IoU 的 prediction mask 是所有正式对象的 union，不只包含 matched expected
track；extra 因而能直接出现在曲线中。外置 bundle 默认位于：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_visualizations/
  scene_default_v6/<scene>/<case>/<job>-<artifact-identity>/
    open_world_v2_overlay.mp4
    open_world_v2_audit.json
```

Overlay 并列展示 reference/prediction、逐 ID mask/轨迹、expected/predicted
cardinality、missing/extra/rejected、birth/death/switch、overflow 和完整主体 IoU。
Audit JSON 保留 expected timeline、全部 prediction tracks/detections、逐帧日志和
完整性分解。本地 manifest 保存外部路径、仓库链接、SHA-256、大小与 evaluator config
digest；仓库顶层 `visualizations` 链接可直接查看。

## 8. 自动化反例与阶段性审计

专项测试覆盖：

- common：physics-parent 能力隔离、missing/extra、far replacement、ID switch、
  frozen replacement、合法 exit、overflow、fail-closed、matched mask/IoU；
- pendulum：五条 physics-parent condition anchor、第二 bob、断绳、无 bob 分叉绳、
  主绳 Hough ghost、condition prompt 不依赖 prediction、缺失时长单调降低 coverage；
- rigid body：静止/运动第二主体、10%/25%/50% disappearance、短尾段、replacement、
  合法 exit/reappearance、边界 apparatus rejection、轴外 duplicate、错误 prediction
  axis、overflow、residual failure；
- circular：1/2/N、第三 orbiter、extra/missing 时长单调性、far replacement、外观
  swap、短遮挡、中心回退、overflow、empty prediction 与 physics-parent 像素隔离；
- artifacts：外置 overlay/audit、逐 ID/逐帧日志、SHA-256 manifest 与渲染失败隔离；
- regression：旧协议 fingerprint、collision 2.2 与 Task 主指标不变。

阶段性真实 GT-self 审计为：

| scene / Case | score | integrity gate | 正式 tracks | missing / extra |
| --- | ---: | ---: | ---: | --- |
| circular，单对象 | `0.9999999994` | `1.0` | 1 | 0 / 0 |
| circular，双对象 | `0.9999999993` | `1.0` | 2 | 0 / 0 |
| incline，`incline_r1_a32deg_bgblack_img_0341` | `0.944144` | `1.0` | 1 | 0 / 0 |
| pendulum，`...a020deg` | `0.791340` | `1.0` | 1 | 0 / 0 |
| free fall，`freefall_r2_l_h060cm` | `0.831280` | `1.0` | 1 | 0 / 0 |

Pendulum 五条 OOD condition 均能从当前 condition 冻结 bob，而不是借用 parent 或
prediction 像素。最新 pendulum GT-self 的 topology 为 `0.920608`，
`branch_frame_count=0`、branch exposure 为 0；content 小于 1 主要来自 scene
segmentation/topology 近似，而不是把支架误报成新增分叉。实体 gate 已保持 1，因此
observer 偏差没有成为新增或消失惩罚。

真实 free-fall reference 还制作了两个只存于 `/mnt/nvme1` 的视频级反例：

| variant | score | gate | missing | extra | evaluator error |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT-self | `0.831280` | `1.000000` | 0 帧 | 0 帧 | 0 |
| persistent extra | `0.453587` | `0.555556` | 0 帧 | 8 帧 | 0 |
| middle/late disappearance + reappearance | `0.014294` | `0.091125` | 6 帧 | 2 帧 | 0 |

重现对象被识别为新实体，没有填补 condition-frozen ID。三组均为正常
`evaluated`，overlay、audit JSON、SHA-256 和 ffprobe 校验完整。汇总位于：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_audits/
  scene_default_v6_adversarial/summary.json
```

仓库可从
`visualizations/scene_default_v6_adversarial/free_fall/freefall_r2_l_h060cm/`
直接查看三条 overlay。

统一的定向审计入口为：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_open_world_evaluator_v6.py \
  --case-id freefall_r2_l_h060cm \
  --self-check \
  --output /mnt/nvme1/physics_video_benchmark/evaluation_audits/my_v6_audit
```

也可重复传入 `--prediction CASE_ID=/absolute/prediction.mp4`。脚本只覆盖四个
`open_world_v2` scene；冻结 collision 2.2 继续使用
`scripts/audit_collision_evaluator_v5.py`。

最终仓库回归为 `408/408` tests 通过；`compileall src tests scripts` 与
`git diff --check` 通过。协议测试同时固定 v3/v4/v5、最终 v6、四个新增 evaluator
和 collision 2.2 的 fingerprint，并断言 v6 collision 配置逐字段等于 v5。

这些结果证明公共契约、合成反例和一组真实像素扰动路径已落地，但不等于完成正式
leaderboard 审计。发布 leaderboard 前仍需扩大四场景真实 baseline prediction 与
OOD 批量审计，并让所有 baseline 使用同一个最终 v6 fingerprint 重评；same-case 与
physics-parent capability 必须继续分层报告。
