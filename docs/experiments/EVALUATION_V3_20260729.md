# 五场景鲁棒主体评估 v3（2026-07-29）

## 1. 结论与发布边界

本轮把五个 scene 的正式评估升级为 `scene_default_v3`。实现 commit 为 `86dc481`，
协议 fingerprint 为：

```text
14dac014a9311ce32be65052f88875e9a9432d64f044dfeb7cc5eceb9adc41c6
```

官方 Task 已升级为 `five_scene_finetune_eval_v6` 和 `five_scene_direct_eval_v6`，两者都
显式固定 v3。历史 v1/v2 协议和已有 canonical evaluation 未覆盖；七组冻结预测通过
独立 reevaluation variant 重评。

批量审计结论：

- 七个 variant 均为 `workflow_status=complete`；
- 七组 evaluator internal error 均为 0；
- View A 三组 coverage 统一为 48/66，View B 四组统一为 167/214；
- prediction 侧媒体、分割或跟踪失败均成为 `evaluated + score=0 + degraded`，不会再把
  难例移出分母；
- 剩余 18 个 View A、47 个 View B unavailable Case 都是 reference 侧不可评估，不
  惩罚 Baseline；
- 七组 strict Task score 仍为 `null`，因为 reference coverage 不完整；
- observed mean 只用于诊断，不能作为正式排行榜分数；
- G15 两组仍是 `diagnostic_pretrained`，不得进入正式排名。

v3 改变了主 metric 和失败语义，不能把 v2 与 v3 observed mean 的变化解释为模型性能
提升或下降。

## 2. 设计规划与落地

### 2.1 目标分解

评估器按三层组织：

```text
媒体与责任归一化
    ├─ 统一物理时间轴、分辨率、FPS 和画布
    ├─ 区分 reference / prediction / internal failure
    └─ 生成稳定 reason code
              │
通用物理主体层
    ├─ position
    ├─ shape
    └─ appearance
              │
scene-specific physics state
    ├─ pendulum
    ├─ free_fall
    ├─ collision_1d
    ├─ inclined_plane_slide
    └─ uniform_circular_motion
```

通用主体层集中实现位置、形状和外貌，避免五个 scene 复制出互不一致的视觉评分。
scene evaluator 只负责主体观察与该实验独有的运动状态。TaskEvaluator 负责冻结 job
全集、失败责任、coverage 和严格聚合。

### 2.2 有同 Case GT

逐帧主体分数为：

```text
position   = 0.60 × centroid similarity + 0.40 × raw-canvas mask IoU
shape      = 0.60 × canonical mask IoU + 0.40 × boundary F
appearance = 0.45 × Lab color
           + 0.35 × masked SSIM
           + 0.20 × gradient texture

subject = 0.50 × position + 0.20 × shape + 0.30 × appearance
case    = 0.60 × subject + 0.40 × scene physics state
```

raw-canvas IoU 和质心距离反映绝对位置；canonical crop 保持宽高比、居中并缩放，令
shape 和 appearance 不再重复惩罚绝对位置与尺度。空 mask、非有限数值和过小 ROI 都
得到有限保守分数，不会产生 NaN 或“空集完美匹配”。

### 2.3 无同 Case GT 的 OOD

对于只有 parent physics reference 的 OOD Case：

```text
case = 0.70 × parent-relative physics state
     + 0.30 × generated appearance vs Case conditioned first frame
```

parent 视频只提供动力学，不参与 OOD 外貌和像素位置扣分。外貌 reference 是当前
Case 自己的 `assets.first_frame`；prediction frame 0 的 mask 仅作为该不可变条件图上的
主体 ROI。不存在真实 continuation 时，不伪造后续绝对位置或形状 GT。

### 2.4 失败责任矩阵

| 故障 | v3 status | score | 是否惩罚 Baseline |
| --- | --- | ---: | --- |
| prediction 记录缺失或未完成 | `evaluated` | 0 | 是 |
| prediction 视频缺失、损坏或过短 | `evaluated` | 0 | 是 |
| prediction 主体分割/跟踪失败 | `evaluated` | 0 | 是 |
| reference 资产、时长或主体观测失败 | `unavailable` | `null` | 否 |
| 数据契约冲突、重复记录 | `error` | `null` | 不静默掩盖 |
| 未捕获 evaluator 内部异常 | `error` | `null` | 不静默掩盖 |

把所有未知异常都改成零分会隐藏 evaluator bug，因此只有已知 prediction-side failure
降级；真正的内部异常继续显式失败。

### 2.5 五个 scene

| scene | 主体观察 | scene physics state | 主体差异 |
| --- | --- | --- | --- |
| pendulum | motion proposal + SAM 2 bob mask | 相对角轨迹、周期、振幅、支点/摆长稳定性 | 位置、bob 形状、外貌 |
| free_fall | motion proposal + SAM 2 | 竖直轨迹、归一化加速度、落地时刻、横漂 | 位置、物体形状、外貌 |
| collision_1d | 首帧多实例候选 + SAM 2 + 连续性匹配 | 多实例轨迹、接触时刻、碰前后速度、碰撞状态、一维约束 | 匹配实例的联合位置、形状、外貌 |
| inclined_plane_slide | motion proposal + SAM 2 + 运动轴整流 | 沿斜面轨迹、归一化加速度、下滑时间、接触/姿态 | 原画布位置、canonical 形状、外貌 |
| uniform_circular_motion | 颜色主体候选 + 连续性实例分配 | 相对角轨迹、角速度、轨道几何、匀速性 | 多主体位置、形状、外貌 |

每个正常 Case 保留 Jensen 风格的 `physical_subject_iou_curve.png`，并新增
`subject_components.csv` 和 `subject_similarity_curve.png`。退化 Case 也尽可能写出
同名低分曲线和 CSV，便于区分“模型失败”和“评估器没有产物”。

## 3. 七组冻结预测重评

主表中的 v2 列只用于展示 coverage 语义变化。Quantity embedding 的 v2 数字来自此前
已记录的 alternate v2 重评；其余六组来自各自冻结的 canonical v2 结果。

| Baseline | View | eligibility | v2 eval/coverage | v3 eval/coverage | degraded | unavailable | error | v3 observed mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantity embedding | A | `regular` | 46/66 (0.696970) | 48/66 (0.727273) | 2 | 18 | 0 | 0.459783 |
| WAN generic | A | `regular` | 45/66 (0.681818) | 48/66 (0.727273) | 3 | 18 | 0 | 0.392730 |
| WAN physics | A | `regular` | 44/66 (0.666667) | 48/66 (0.727273) | 4 | 18 | 0 | 0.386042 |
| Cosmos generic | B | `regular` | 152/214 (0.710280) | 167/214 (0.780374) | 15 | 47 | 0 | 0.288784 |
| Cosmos physics | B | `regular` | 148/214 (0.691589) | 167/214 (0.780374) | 19 | 47 | 0 | 0.261278 |
| G15 generic | B | `diagnostic_pretrained` | 154/214 (0.719626) | 167/214 (0.780374) | 13 | 47 | 0 | 0.389498 |
| G15 physics | B | `diagnostic_pretrained` | 150/214 (0.700935) | 167/214 (0.780374) | 17 | 47 | 0 | 0.394140 |

### 3.1 逐 scene

单元格为 `observed mean (evaluated / expected)`。

| scene | Quantity embedding | WAN generic | WAN physics | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pendulum | 0.511961 (13/13) | 0.506596 (13/13) | 0.568142 (13/13) | 0.418662 (40/40) | 0.406896 (40/40) | 0.408568 (40/40) | 0.434563 (40/40) |
| free_fall | 0.397972 (3/4) | 0.438534 (3/4) | 0.319650 (3/4) | 0.222801 (10/11) | 0.076819 (10/11) | 0.388314 (10/11) | 0.312078 (10/11) |
| collision_1d | 0.285344 (7/21) | 0.093631 (7/21) | 0.104656 (7/21) | 0.326275 (12/32) | 0.315346 (12/32) | 0.255627 (12/32) | 0.305429 (12/32) |
| inclined_plane_slide | 0.541421 (21/22) | 0.376571 (21/22) | 0.369077 (21/22) | 0.345410 (91/95) | 0.374722 (91/95) | 0.356238 (91/95) | 0.334895 (91/95) |
| uniform_circular_motion | 0.562219 (4/6) | 0.548318 (4/6) | 0.568684 (4/6) | 0.130770 (14/36) | 0.132606 (14/36) | 0.538741 (14/36) | 0.583735 (14/36) |

不同 View 的样本分布和训练条件不同，不能横向解释 A/B 数值。相同 View 内也因为
reference coverage 不满，只能把这些数值视为 observed-only 诊断。

### 3.2 View A 的 ID/OOD1

单元格为 `observed mean (evaluated / expected)`。Free-fall 没有 OOD1 Case。

| scene/partition | Quantity embedding | WAN generic | WAN physics |
| --- | ---: | ---: | ---: |
| collision_1d/test_id | 0.376046 (1/3) | 0.000000 (1/3) | 0.000000 (1/3) |
| collision_1d/test_ood1 | 0.194642 (6/18) | 0.187262 (6/18) | 0.209313 (6/18) |
| free_fall/test_id | 0.397972 (3/4) | 0.438534 (3/4) | 0.319650 (3/4) |
| inclined_plane_slide/test_id | 0.596146 (6/6) | 0.284706 (6/6) | 0.274593 (6/6) |
| inclined_plane_slide/test_ood1 | 0.486695 (15/16) | 0.468436 (15/16) | 0.463561 (15/16) |
| pendulum/test_id | 0.463663 (8/8) | 0.423457 (8/8) | 0.527332 (8/8) |
| pendulum/test_ood1 | 0.560259 (5/5) | 0.589734 (5/5) | 0.608952 (5/5) |
| uniform_circular_motion/test_id | 0.467624 (2/2) | 0.441235 (2/2) | 0.460391 (2/2) |
| uniform_circular_motion/test_ood1 | 0.656815 (2/4) | 0.655402 (2/4) | 0.676977 (2/4) |

### 3.3 View B 的确定性 group

每个 group 单元格先对五个 scene 的 observed mean 做宏平均，括号为该 group 跨 scene
的 `evaluated / expected`。这些 group 是均衡切片，不是 ID/OOD 标签。

| group | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: |
| group_1 | 0.315318 (36/45) | 0.288894 (36/45) | 0.391018 (36/45) | 0.423022 (36/45) |
| group_2 | 0.244433 (34/43) | 0.229982 (34/43) | 0.348711 (34/43) | 0.402477 (34/43) |
| group_3 | 0.250442 (32/42) | 0.250079 (32/42) | 0.325062 (32/42) | 0.362558 (32/42) |
| group_4 | 0.293397 (33/42) | 0.237800 (33/42) | 0.305614 (33/42) | 0.304925 (33/42) |
| group_5 | 0.326064 (32/42) | 0.286355 (32/42) | 0.442623 (32/42) | 0.397655 (32/42) |

### 3.4 Direct-only Baseline 对齐到 View A 66 Case

这里从同一次 View B 生成结果中精确切出 `view_a.json` 的 66 个测试 Case，不伪造
fine-tune。聚合顺序与 View A observed macro mean 一致。

| Baseline | evaluated | coverage | aligned v3 observed macro mean |
| --- | ---: | ---: | ---: |
| Cosmos generic | 48/66 | 0.727273 | 0.338171 |
| Cosmos physics | 48/66 | 0.727273 | 0.305518 |
| G15 generic | 48/66 | 0.727273 | 0.421046 |
| G15 physics | 48/66 | 0.727273 | 0.436281 |

### 3.5 Prediction-side degradation reason

| reason | Quantity embedding | WAN generic | WAN physics | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `circular_disk_not_found` | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `collision_frame_zero_objects_missing` | 0 | 1 | 1 | 0 | 0 | 2 | 1 |
| `collision_striker_missing` | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
| `insufficient_incline_motion` | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `insufficient_instance_tracks` | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| `insufficient_valid_masks` | 1 | 1 | 2 | 5 | 10 | 6 | 10 |
| `insufficient_vertical_motion` | 0 | 0 | 0 | 2 | 1 | 0 | 2 |
| `prediction_pendulum_observation_failed` | 0 | 0 | 0 | 0 | 0 | 1 | 1 |

这些条目都已进入 evaluated 分母并记零。reference-side reason 不列入该表，因为它们
属于 Dataset/evaluator 可评估性诊断，不是模型错误。

## 4. 验证与可复现性

### 4.1 回归验证

完整测试集：

```text
225 tests passed
```

覆盖的关键不变量包括：

- 同一主体自比严格为 1；
- 单独改变位置或外貌时分数严格下降；
- 空 mask 不会得到完美 IoU；
- prediction 主体缺失得到有限零分而不是 evaluator failure；
- reference 主体不可观测保持 `unavailable`；
- parent reference 不会把 parent 像素当成 OOD 外貌 GT；
- missing prediction record 在 v3 中进入 coverage 并记零；
- v1/v2/v3 identity、evaluator 版本和官方 Task identity 相互隔离。

论文依据和工程取舍详见 [`docs/EVALUATION.md`](../EVALUATION.md)：

- DAVIS region \(J\) / boundary \(F\)；
- SSIM；
- LPIPS 的感知距离原则；
- SAM 2；
- CoTracker 与 TAP-Vid 的跟踪/遮挡评估思路。

LPIPS 暂不作为硬依赖，以免模型下载、backbone 或设备失败使单个 Case 变成 evaluator
error。若以后加入，必须发布新的版本化协议。

### 4.2 Variant 与 canonical 完整性

七组均使用：

```text
evaluation_id = robust-subject-v3-20260729
```

路径模式：

```text
runs_v2/<RUN_ID>/reevaluations/
  scene_default_v3/
    14dac014a9311ce32be65052f88875e9a9432d64f044dfeb7cc5eceb9adc41c6/
      robust-subject-v3-20260729/
```

本轮 `<RUN_ID>`：

| Baseline | Run ID |
| --- | --- |
| Quantity embedding | `five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728` |
| WAN generic | `quantity_controls_viewa_v5_seed42_20260729__wan22_ti2v_5b_lora_r32_v3_generic` |
| WAN physics | `quantity_controls_viewa_v5_seed42_20260729__wan22_ti2v_5b_lora_r32_v3_physics` |
| Cosmos generic | `quantity_controls_viewb_v5_seed42_20260729__cosmos3_nano_i2v_generic` |
| Cosmos physics | `quantity_controls_viewb_v5_seed42_20260729__cosmos3_nano_i2v_physics` |
| G15 generic | `quantity_controls_viewb_v5_seed42_20260729__wan22_g15_sparse_motion_r32_e20_generic` |
| G15 physics | `quantity_controls_viewb_v5_seed42_20260729__wan22_g15_sparse_motion_r32_e20_physics` |

最终核验：

- 七个 `reevaluation.json` 均为 `workflow_status=complete`；
- 七个 `artifact_manifest.json` 均存在；
- 4,900 个 variant 文件逐一复核 size 和 SHA-256，0 个不一致；
- 49 个关键 canonical 文件按重评前 `source_integrity.json` 复核，0 个不一致；
- 原 `run.json`、`state.json`、report、component fingerprints 和 v1/v2 evaluation
  未被修改。

Reevaluation 产物由仓库的运行产物策略忽略，不进入 Git；协议、实现、测试和本报告
进入版本控制。

## 5. 仍需解决的数据侧限制

v3 消除了 prediction-side failure 造成的 coverage 偏差，但没有伪装解决 reference
观测失败：

- View A 有 18 个 reference unavailable；
- View B 有 47 个 reference unavailable；
- 碰撞和圆周运动的 reference coverage 最低，是 strict score 仍不可发布的主要原因。

下一步应单独修复或补充 Dataset reference observation，不应再次改变 prediction 失败
语义，也不应把 reference failure 记到 Baseline 上。任何 reference 观察后端、阈值或
评分权重变化都必须发布新的 protocol/evaluator identity，不能静默重算 v3。
