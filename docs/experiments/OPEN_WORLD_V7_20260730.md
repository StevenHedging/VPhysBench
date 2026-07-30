# 开放世界对象评估 v7：迭代加固与审计说明

日期：2026-07-30

## 1. 状态

`scene_default_v7` 是建立在 `open_world_v2` 公共内核上的 shadow protocol。它针对
v6 真实 GT-self 审计暴露出的观察器问题，迭代单摆、自由落体、斜面下滑和匀速圆周
运动；一维碰撞继续逐字段复用冻结的 evaluator 2.2。

本轮按用户指示在自由落体数据质量瓶颈处终止。以下为已提交 shadow
checkpoint 的身份；它用于复现和后续续作，不表示 v7 已成为正式 leaderboard
协议：

| scene | evaluator | version | fingerprint |
| --- | --- | ---: | --- |
| pendulum | `pendulum_open_world_structure` | 2.1 | `b59f44c166a2ab4344ced25304147014ed014bdf0eccd0233ebae133e4bd3380` |
| free_fall | `free_fall_open_world` | 2.1 | `3620bacdcd670deba38212c94eaca55d0999db1c133a63284285f2b00a92f23a` |
| inclined_plane_slide | `inclined_plane_open_world` | 2.1 | `2ed49fbedf3d5911a3fbb85d7c12662b1075f33844acab9733d10b32c3471e45` |
| uniform_circular_motion | `uniform_circular_motion_open_world` | 2.1 | `1543a9ac87e85d10487df2005f51a74932534d92befae0a1a5cd44fcbce6a5b8` |
| collision_1d | `collision_1d_open_world_nbody` | 冻结的 2.2 | 沿用冻结身份 |

```text
protocol_id          = scene_default_v7
protocol_fingerprint = 93d2ba46e7b904691504e28c05c8ab3312b0f9e5acad29d55efcc6cc8d730e36
```

本文中的开发审计数值不能替代冻结后的 protocol identity，也不能与 v3/v4/v5/v6
分数直接拼接成 leaderboard。

## 2. 目标与非目标

v7 的目标是：

- 对同一像素输入使用对称、condition-causal 的 reference/prediction 观察路径；
- 保留所有可信物理参与体，使新增、复制、消失、replacement、ID switch 和 overflow
  都进入有限、可解释的惩罚；
- 将 apparatus、运动残影和弱噪声与真实 participant 区分，同时让分类依据留在审计
  日志中；
- 保持 `scene_subject_state_similarity`、Task 聚合接口和旧协议结果不变；
- 对 prediction 侧 corner case fail closed，而不是用 evaluator error 或观察器失效
  逃避计分。

v7 不以以下事项为目标：

- 不重写 `open_world_v2` 的正式 gate，也不修改 collision 2.2；
- 不保证一个通用视觉 proposal 能覆盖任意未知背景、材质和摄像机；
- 不从结构化物理量合成 OOD 的未来 GT 像素位置；
- 不把 GT-self 当作 baseline 质量结论，也不在本轮发布新 leaderboard。

## 3. 公共架构与正式计分

四个非碰撞 scene 共享以下数据流：

```text
Case manifest + current condition + reference capability
  -> existence / localization / association expected timeline
  -> condition-frozen expected identity

directed expected-object observation
  + scene-specific residual discovery
  -> causal tracks / tentative candidates / rejected apparatus / overflow
  -> evidence-tiered Hungarian assignment with explicit null

object integrity gate
  x scene-specific weighted geometric content
  -> scene_subject_state_similarity
```

Expected timeline 的分母不能由 prediction 决定。位置相似度低于配置阈值的 assignment
边被拒绝，并同时形成 expected missing 和 prediction extra；因此远处替代物或复制体
不能接管冻结 ID。Tentative、apparatus rejection 和 participant 必须保留各自的证据
来源，超过安全容量的候选进入 `__overflow__` false exposure。

以真实秒 exposure 记：

```text
R = expected entity-time exposure
P = all formal prediction exposure, including extra and overflow
M = matched exposure

PresenceDetA = M / (R + P - M)
AssA         = matched-exposure-weighted pair association IoU
raw_gate     = PresenceDetA * AssA
delete_cap   = (M / R)^3
integrity    = min(raw_gate, delete_cap)

case_score   = integrity * weighted_geometric_mean(scene_content)
```

Missing 还会减少 scene physics 的有效证据覆盖。GOSPA、SoftDetA、逐帧距离和完整主体
IoU 是诊断量，不重复乘入正式 gate。`delete_cap` 防止模型通过删除错误帧提高总分。

## 4. 各 scene 的 v7 变化

### 4.1 单摆

v6 的主要问题是 reference 与 prediction 观察路径不对称，以及错误的 condition
anchor 导致 SAM/residual 分裂成多个身份。v7 改为：

- 只从当前 condition 帧融合 Hough bob、pivot/string 几何、物理半径/摆长比、初始
  角度、raw edge 与 body contrast，冻结 `pivot-string-bob` 结构；
- same-case reference 与 prediction 使用完全相同的 condition prompt、proposal
  fusion、tracking 和 topology 路径；
- directed 与 residual 是证据源而不是身份；先融合重叠 proposal，再建立
  condition-causal 主轨；
- 独立 residual 必须满足多帧/真实时长 persistence 才晋级为 extra participant；
- 断绳和分叉采用对称、多源、持续性证据，单帧 dropout 不形成正式拓扑事件；
- 支架、pivot 横杆、主绳 corridor 和 matched bob 有显式 exclusion/rejection 日志。

Scene content 仍比较摆角轨迹、周期/幅度、外貌、形状和拓扑；完整性 gate 负责 bob 的
新增、消失和身份连续性。

### 4.2 自由落体与斜面下滑

两者继续共享 `rigid_body_open_world.py`，v7 的重点是消除“同一视频、两套观察器”：

- reference 先进行 compact multi-hypothesis 观察，候选按形状、连续性和场景运动约束
  验证，不再默认选择面积最大的运动组件；
- same-case reference 和 prediction 均由当前 condition 冻结的身份与同一 residual
  policy 观察；
- condition difference、双向 temporal motion 与独立 compact-shape proposal 在
  跟踪前融合；
- 与主实体保持固定偏移和速度、但缺少独立 compact 证据的运动残影可降为
  auditable optical artifact；有持续 compact 证据的第二球/滑块仍是 participant；
- directed 短 gap 只有在当前帧仍有几何、外貌和运动像素证据时才可恢复，不能只靠
  轨迹外推制造 mask；
- 物理约束可按 empirical reference difference 比较，prediction 不能重新拟合重力
  轴或斜面轴。

自由落体按冻结竖直轴比较轨迹、加速度、impact time、横漂和单调性；斜面按冻结斜面
轴比较沿面轨迹、加速度、descent time、法向接触、姿态和单调性。合法落地或末端退出
必须由 reference lifecycle 证据确认，退出后重现仍计 extra。

### 4.3 匀速圆周运动

v6 将圆盘 apparatus 硬编码为绿色，导致红/蓝圆盘和 metadata 颜色错误时 reference
失败。v7 改为：

- 在饱和度、亮度、面积和圆形几何约束下，自适应选择主圆盘 hue；
- 只从 condition/reference 首帧冻结 center、radius、ROI 与 apparatus palette；
  prediction 后续帧不能重新估计圆盘并改写坐标；
- 将 condition Lab/value palette、同坐标外观和 temporal residual 结合，使同 hue
  但外观不同的静止/运动复制体不会被当作圆盘擦除；
- 保留所有满足面积与宽松 compactness 条件的内部 component，不按 expected N
  截断；
- condition 冻结外貌、半径与 absolute phase 身份，明显外观跳变形成 ID 断点；
- 盘中心附近使用 Cartesian fallback，其余位置使用 disk-normalized polar 距离；
- 候选和 track 超限仍进入 overflow，而不是静默丢弃。

same-case reference/prediction 共用 condition 坐标；physics-parent OOD 的 parent 和
当前 condition 各自冻结环境坐标，仅以各自 condition-relative phase 比较规范化
动力学。正式位置误差只进入 orbit content；SoftDetA/GOSPA 与完整主体 IoU 是诊断，
不再重复进入 integrity gate。已知观察风险是 condition 首帧本身若只显示部分圆盘，
仍需要有界 ellipse/hull fallback；这不应通过放宽到全画布前景来规避 apparatus
约束。

### 4.4 一维碰撞

v7 不修改碰撞 evaluator、配置或可视化 namespace。碰撞继续使用 manifest 数量、
多帧圆体 proposal、SAM2、开放世界 residual、Hungarian null assignment 和
scene-specific N-body physics。旧的 r7/r8 数值必须由逐字段配置相等与 fingerprint
回归保护。

冻结实现的 32 条 GT-self 只读审计得到 31 evaluated、1 unavailable、0 evaluator
error；31 条可评样本均值 `0.91785`、中位数 `0.96938`，其中 8 条低于 `0.9`。
低分主要来自 2.2 的 reference/prediction 观察不对称：prediction-only residual 会把
directed SAM 的短缺口、同体重复响应和运动残影计为 extra，并进一步产生虚假 contact
event。这是 evaluator 误罚证据，不应解释成 GT 物理错误。唯一 unavailable 的
`collision_r2_small_steel_small_steel_small_steel_v07845` 同时包含数据与逻辑因素：
首帧实际只见两球而标注声明三球均可见，第三球稍后入画且较早出画；2.2 又按全视频
而非合法生命周期区间计算 coverage，最终为 `0.1268 < 0.2`。

这些问题留给未来 shadow 2.3：只允许增加对称 directed+residual 观察、轨迹级 residual
fusion、生命周期感知 coverage 和完成身份归属后的 contact graph；本轮不能通过修改
2.2 或简单放宽阈值获得更高分。

## 5. Physics-parent 隔离

`physics_parent` 只提供规范化动力学证据，不提供 OOD child 的背景、底座、物体外貌
或未来 raw pixel localization。隔离规则是：

- 实体数量、身份、外貌、尺度、初始位置和 apparatus anchor 来自当前 child
  condition；
- `t>0` 的 parent raw mask/centroid 不得进入 expected localization 或 assignment；
- parent 轨迹先变为无像素语义的规范化进度，再映射到 child condition 冻结的坐标系；
- 单摆 pivot/string/bob、斜面轴和圆盘 anchor 均从 child condition 冻结；
- 无法可靠冻结 child anchor 是 reference-side `unavailable`，不能读取 prediction
  mask 自行定义 ROI。

提交前必须用 parent 未来帧像素扰动反例确认：在规范化动力学不变时，OOD 得分和
expected timeline 不随 parent 原始位置、背景或外貌改变。

## 6. 失败语义

协议保持明确的责任边界：

| 失败位置 | 正式状态 | 计分语义 |
| --- | --- | --- |
| prediction record/media/timeline | `evaluated` | 有限低分；缺失 exposure 保留 |
| prediction observer/comparison/scene physics | `evaluated` | fail closed，通常 gate 为 0 |
| reference/condition/manifest/reference observer | `unavailable` | 不猜测 GT，不计作 prediction 错误 |
| 未捕获的实现 invariant | `error` | 视为 evaluator 缺陷，必须修复 |
| overlay/绘图写入 | score 不变 | manifest 写 `status=failed` 和原因 |

Prediction observer 失败会加入 `__observer_failure__` false exposure，不能退回
directed-only 高分。短视频、坏帧、空 mask、residual failure 和容量 overflow 都应
产生有限、可序列化的 audit 结果。

## 7. 可视化与审计产物

每个非碰撞 Case 至少保留：

```text
per_frame.csv
physical_subject_iou_curve.png
entity_position_curve.png
object_cardinality_timeline.png
scene-specific trajectory curve
open_world_audit.json / object_centric_audit.json
open_world_v2_artifact_manifest.json
```

完整主体 IoU 的 prediction union 包含全部正式 participant 和 extra，不只包含 matched
expected track。外置 bundle 默认写入：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_visualizations/
  scene_default_v7/<scene>/<case>/<job>-<artifact-identity>/
    open_world_v2_overlay.mp4
    open_world_v2_audit.json
```

`PHYSBENCH_VISUALIZATION_ROOT` 可覆盖外置根目录；仓库 `visualizations` 软链接用于
直接查看。Manifest 记录绝对路径、仓库链接、SHA-256、文件大小和 evaluator config
digest。Overlay 同时显示逐 ID mask/轨迹、expected/predicted cardinality、
missing/extra/rejected、birth/death/switch、overflow 和完整主体 IoU。

## 8. 已有验证证据

Dataset identity：

```text
dataset_id = physics_video_five_scene_v4
digest     = be5ea8880be3cf8e0d0d316ee025cf02905ef5aeb544f3df5f4b966e5e14782d
```

证据必须按版本和范围解释：

| 范围 | 已观察结果 | 可支持的结论 |
| --- | --- | --- |
| circular v7，36 个 same-case GT-self | 36 evaluated，0 unavailable/error/degraded；score min/mean/median/max 为 `0.941176469726` / `0.992761977253` / `0.999999999215` / `0.999999999869`；48/48 expected entity 均冻结成功 | 自适应圆盘与当前真实数据对齐；最低分来自持续 5 帧的保守 residual extra 惩罚 |
| circular v7，持续第三 orbiter | 3 tracks，所有帧 extra，gate `0.6666666667` | extra 未被 shape filter 或 N 截断吞掉 |
| circular v7，末三帧第三 orbiter | 三个受影响帧均 extra，gate `0.8648648649` | extra exposure 随持续时间平滑扣分 |
| pendulum v7 开发审计，v6 最差 R1 小角度 Case | 81 帧无 missing/extra/overflow，gate 1；完整主体 IoU 每帧 1；object-centric comparison score `0.9992120658` | condition-only anchor 与对称 bob 跟踪修复了该具体失败 |
| pendulum v7 定向反例 | proposal fusion、持久第二 bob、residual persistence、对称 trace/topology、单帧断绳和有限不确定性均有单元测试 | 核心策略有可重复的局部回归 |
| v6 pendulum 真实审计 | 40/40 完成且 0 error；35 个真 GT-self 均值仅 `0.5590`，missing 11/35、extra 22/35、overflow 5/35 | v7 的观察对称性修改有真实问题依据，不代表 v7 已全量通过 |
| v6 rigid 真实审计 | free fall 10/11 evaluated；incline 95/95 evaluated；合计 0 error | 旧路径可运行，但 v7 刚体仍需完整真实重跑 |
| 冻结 collision 2.2 真实审计 | 31/32 evaluated，0 error；唯一 unavailable 的第三实体 reference coverage 为 `0.1268 < 0.2` | collision 保持冻结；该 reference-side 缺口未被 v7 掩盖 |

证据目录：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_audits/
  iterative_v6_round1/circular/scene_default_v7_gtself/
  iterative_v6_round1/circular/ADAPTIVE_PATCH_REVIEW.md
  iterative_v6_round1/pendulum/ANALYSIS.md
  iterative_v6_round1/rigid/audit_report.json
  iterative_v6_round1/collision/audit_report.json
  pendulum_v7_dev_r3/
```

圆周报告中的开发期 protocol/evaluator fingerprint 只绑定当时 snapshot；本文不将其
提前声明为最终 v7 identity。

## 9. 尚未验证与发布门槛

冻结 v7 前仍需完成：

- 全量 pendulum 35 个真 GT-self 与 5 个 physics-parent capability 审计，确认 R1/R2
  都不再出现系统性假 missing、extra、overflow、断绳或分叉；
- free-fall 11 个和 incline 95 个真实 GT-self 的 v7 重跑，包括旧 unavailable/低分
  Case 的人工 overlay 复核；
- 刚体静止复制体、运动复制体、残影、合法末端退出、退出后重现、错误轴自拟合和
  current-pixel gap recovery 的视频级反例；
- physics-parent 未来像素 counterfactual，以及 parent/child 外貌和位置完全不同的
  OOD Case；
- circular 高饱和背景、top-K hue、透视/partial disk、中心附近主体和真实 ID swap；
- 各 scene 的 10%/25%/50% disappearance、远处 replacement、ID 接力、overflow、
  短视频与 residual failure，并验证异常持续越久分数越低；
- 真实 baseline prediction，而不只是 GT-self 和合成反例；
- 0 evaluator error、GT-self 高分、artifact hash/ffprobe 完整，以及旧
  v3/v4/v5/v6、collision 2.2、Task coverage 与宏平均回归不变；
- 完整 unittest、compileall、schema/registry resolution 和 `git diff --check`
  通过后，再将本文所有“提交前冻结”替换为提交所绑定的 identity。

在上述条件完成前，`scene_default_v7` 只能用于 shadow audit，不能作为正式
leaderboard 的混合来源。

本轮终止说明：自由落体真实数据中的手、阴影、球体释放遮挡和首帧主体歧义使
condition-only observer 无法达到与其余场景相同的可靠校准水平。按用户指示不再继续
为该数据分布调参；现有实现和反例测试保留为有限、fail-closed 的 shadow evaluator。

## 10. 复现

环境与数据：

```bash
cd /root/Steven/physics_video_benchmark

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  validate-dataset \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --check-assets
```

统一审计入口的文件名沿用 v6，但可显式加载 v7：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_open_world_evaluator_v6.py \
  --protocol scene_default_v7 \
  --case-id circular_r1_silver02cm_img_0370 \
  --self-check \
  --device cuda:0 \
  --output /mnt/nvme1/physics_video_benchmark/evaluation_audits/repro_v7
```

`--self-check` 只把 same-case reference 标为 `gt_self`。对于没有同 Case GT 的 OOD
Case，解析到的 physics parent 会标为 `physics_parent_as_prediction`；这是 capability
隔离探针，不得并入 GT-self 的最小值、均值或通过率。报告的 `summary.by_variant`、
`summary.by_scene_variant` 会分别给出状态与有限分数统计。

评估真实 prediction：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_open_world_evaluator_v6.py \
  --protocol scene_default_v7 \
  --prediction CASE_ID=/absolute/path/to/prediction.mp4 \
  --device cuda:0 \
  --output /mnt/nvme1/physics_video_benchmark/evaluation_audits/repro_v7_prediction
```

Collision 仍走冻结入口：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_collision_evaluator_v5.py \
  --case-id CASE_ID \
  --self-check \
  --device cuda:0 \
  --output /mnt/nvme1/physics_video_benchmark/evaluation_audits/repro_collision_2_2
```

定向与全量回归：

```bash
PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -p 'test_pendulum_open_world_v7.py' -v

PYTHONPATH=src:tests /root/miniconda3/envs/phybench/bin/python \
  -m unittest discover -s tests -v

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  -m compileall -q src tests scripts

git diff --check
```

复现时以 `audit_report.json` 中的 dataset、protocol、evaluator identity、runtime
override 与 `implementation.source_tree.digest` 为准。后者覆盖实际
`src/physbench/**/*.py`、审计入口、协议 JSON 和 schema，并同时记录 Git HEAD/dirty
状态；这样开发期同一 `2.1` 标签下的不同源码不会被误认为同一结果。未绑定同一最终
fingerprint 和实现 digest 的结果不得横向汇总。
