# 碰撞评估器 v4：真实生成视频审计

日期：2026-07-30

> 历史快照：文中的全局外置可视化路径记录当时实现。当前实现统一写入所属
> AtomicRun/reevaluation 的 `evaluation/visualizations/`，不要照搬旧路径。

## 1. 结论

`scene_default_v4` 对碰撞评估的主要收益是：

- reference 观测链路从 View B 的 `12/32` 提高到 `32/32`；
- 两组冻结真实 prediction 都达到碰撞 Case `100% evaluated`；
- v3 因 frame-0 发现和硬阈值造成的 `unavailable` 或机械 0 分，被转成可解释的有限低分；
- 多帧角色候选、双向互斥 SAM2 与外置可视化适合继续作为下一版的观测基础。

但当前证据不能支持“v4 已经能准确排序所有生成视频”：

- 正式 subject 分仍比较三球 union，不是三个物理身份；
- 评估器只挑选并追踪恰好三个候选，不能发现第四球、复制体或其持续时间；
- reference/prediction 各自按 seed 几何顺序赋予角色，没有用 Case 条件首帧做跨视频 ID
  锚定；
- 接触事件总是取 `striker-target_1` 最小中心距，没有验证表面接触、closing/separation
  或 `target_1→target_2` 的传播；
- collision state 对全局平移和统一缩放严格不变；
- G15 在 v3/v4 都能给出非零正常分的共同集上，排序相关性较弱。

因此 v4 应定位为“稳定的碰撞 observation/audit 协议”，而不是最终对象中心评分。下一版
必须把逐 ID 位置/外貌、cardinality、lifecycle 和 residual discovery 纳入正式分数。

## 2. 审计范围

协议：

```text
scene_default_v4
fingerprint =
54cefd0a75dc927e783ba7bcd8c75719b6c8b619fc07dce055eb50f94ca60532
```

真实 prediction：

| 冻结运行 | 视图 | 碰撞 Case |
| --- | --- | ---: |
| `five_scene_wan22_quantity_embedding_r32_e10_seed42_918e9f7_20260728` | A | 21 |
| `quantity_controls_viewb_v5_seed42_20260729__wan22_g15_sparse_motion_r32_e20_generic` | B | 32 |

两组重评 variant 的 `evaluation_id` 均为：

```text
object-centric-v4-audit-20260730
```

Reference 审计和 identity 结果继续沿用：

```text
docs/experiments/COLLISION_EVALUATOR_V4_20260730.md
```

## 3. 真实 prediction 结果

### 3.1 Coverage 与分数

| Baseline / split | v3 evaluated | v3 mean | v3 median | v4 evaluated | v4 mean | v4 median | v4 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantity / A | 7/21 | 0.220557 | 0.273617 | 21/21 | 0.377753 | 0.329168 | 0 |
| G15 / B | 12/32 | 0.255627 | 0.295057 | 32/32 | 0.400499 | 0.397850 | 0 |

v3 的低 coverage 主要不是 prediction 缺失，而是 reference 观测失败：

- Quantity：14 个 reference `unavailable`；已 evaluated 的 7 个中还有 2 个 prediction
  观测假零；
- G15：20 个 reference `unavailable`；已 evaluated 的 12 个中还有 4 个 prediction
  观测假零。

v4 解锁了上述全部 Case，且两组分别 `21/21`、`32/32` 为有限分数，没有
`error`、`unavailable` 或 `reason_code`。这证明 v4 已有效修复 frame-0 三球发现和
0.8 valid-ratio 硬门造成的 coverage 问题。

### 3.2 不能把均分上涨解释为排序更准

v3 与 v4 的观测、画布和评分实现不同，不能只比较均分。进一步只取 v3 中已有非零正常
分、v4 中也正常分的共同 Case：

| Baseline | 共同 Case | Spearman(v3, v4) | v4-v3 mean |
| --- | ---: | ---: | ---: |
| Quantity | 5 | 0.900 | +0.0968 |
| G15 | 8 | 0.286 | +0.0387 |

Quantity 小样本中排序大体一致；G15 排序变化明显，且共同集只有 8 个。没有逐 Case
人工盲审标签时，不能判断变化方向哪一版更真实。当前最强证据是 coverage 与假零恢复，
不是 ranking accuracy。

## 4. Reference 与 proposal 质量

View B 的 32 个 reference 虽然全部可评，但角色观测并不都完整：

```text
all role valid ratios:
min    = 0.2568
median = 1.0000
mean   = 0.8917

target_2:
min    = 0.2568
median = 0.888
mean   = 0.751
```

32 个 Case 中：

- 15 个至少一个角色 valid ratio `< 0.8`；
- 10 个至少一个角色 `< 0.5`。

缺失位置目前会线性插值和端点外推后继续估计事件、速度、动量与恢复系数；随后仅以
整段最差角色 coverage ratio 乘到 state score。它没有区分短遮挡、中途永久消失、
轨迹碎裂和 ID switch，也不比较缺失发生在同一帧还是不同帧。

对仓库七组冻结预测的 191 个碰撞视频，仅运行 v4 多帧 Hough prompt builder：

```text
191/191 找到三元组
13/191 seed frame >= 20
 5/191 seed frame >= 30
```

32 个 reference 的 seed 最大为 17。Prediction 侧极晚 seed 说明 proposal recall 很高，
同时也提示选择条件可能过宽松：它能在后段找到“几何上像三球”的候选，但未必仍对应
条件首帧中的三个物理身份。这一桶应在下一版作为 observation risk 单独报告。

只看本次完整重评的 53 个真实 prediction：

```text
3/53 selected seed >= 20
latest selected seed = 34
32/53 在最终选中阈值下存在超过三个 Hough candidate
```

候选超过三个不等于视频一定生成了额外球，也可能包含反光或伪圆；但它证明“选三后丢弃
全部 residual”不是纯理论风险。

## 5. 评分盲区的实证

### 5.1 Union 掩盖逐角色错误

Quantity Case：

```text
collision_r2_glass_marble_glass_marble_glass_marble_v04374
```

v3 因 prediction 只有 `16/69` valid masks，返回机械 0。v4 得到：

```text
case score              = 0.281671
collision state         = 0.149380
union subject           = 0.369865
observation reliability = 0.721311

matched-instance IoU mean = 0.010554
per role                  = [0.0, 0.0, 0.034429]
union appearance          = 0.692691
```

v4 把不可解释的 0 变成了可审计低分，这是进步；但前两个角色全程 IoU 为 0 时，union
appearance 仍接近 0.69，并以 60% 权重进入最终 Case。逐实例 IoU 目前只是 diagnostic，
没有正式制约 subject score。

可视化：

```text
visualizations/scene_default_v4/
collision_r2_glass_marble_glass_marble_glass_marble_v04374/
five_scene_finetune_eval_v4__collision_r2_glass_marble_glass_marble_\
glass_marble_v04374__seed000042-e1d400ff4b63df3d/
```

两组真实 prediction 中：

| 集合 | instance IoU `<0.05` | 同时 final score `>0.3` |
| --- | ---: | ---: |
| Quantity A，21 Case | 6 | 4 |
| G15 B，32 Case | 8 | 3 |

这不表示这些 Case 必须为 0：位置接近但不重叠理应有连续正分。它说明正式分数必须使用
逐 ID 连续位置距离，再单独比较形状与外貌；不能继续用 union 提供角色完整性。

### 5.2 额外对象不可见

`build_multiframe_collision_prompts` 从候选中选恰好三个角色，SAM2 也只传播这三个
seed。未选中的第四球不进入：

- prediction union；
- subject score；
- instance IoU；
- collision state；
- observation reliability。

因此“生成正确三球，再复制一个第四球”可能与原视频得分几乎相同。必须增加
open-world residual discovery；仅有 expected-ID tracking 不能解决 cardinality。

### 5.3 消失和错误生命周期不够敏感

当前：

```text
observation_reliability =
    min_role clip(prediction_valid_ratio / reference_valid_ratio, 0, 1)
```

它只乘在占 Case 40% 的 state score，union subject 仍可能由其他球保持较高。总 valid
ratio 相同但缺失时间完全不同的两条轨迹会得到相同 reliability。SAM2 在物体已经消失
后继续 hallucinate mask 时，valid ratio 还可能虚高。

下一版应在每个 GT 可观测 entity-time slot 上计算 missing exposure，并把全部 residual
prediction exposure 作为 false positive；短接触遮挡、正常出画与永久消失使用不同
visibility/lifecycle 状态。

### 5.4 Role swap 和全局变换

已有 synthetic role-swap 回归中：

```text
collision state after swap = 0.58627
```

测试只要求 `<0.95`。如果三球 union 不变，则按当前：

```text
case = 0.4 × state + 0.6 × subject
```

最终仍可约为 `0.8345`。这个测试只能证明 state 不是 identity，不能证明身份交换受到
足够惩罚。

另外，将三条轨迹共同做任意全局平移或统一缩放时，`score_collision` 仍为 1：

- reference/prediction 分别拟合自己的轴；
- 每侧减去自己的初始位置；
- 每侧除以自己的 scene span。

绝对错位只剩 union subject 的 position 分支承担，最多占最终 Case 的 30%。下一版的
逐 ID 距离必须只使用 reference/condition 尺度和锚点。

完整 counterfactual 组合进一步给出：

| 扰动 | 当前 v4 可得到的分数 |
| --- | ---: |
| identity | 1.0000 |
| 全程第四球 | 1.0000 |
| 后半程第四球 | 1.0000 |
| 一个球后半程消失 | 约 0.6833 |
| union 不变的 target swap | 约 0.8345 |
| 整体平移 200 px | 约 0.7245 |
| 球半径减半 | 约 0.8332 |
| 延迟 16 帧 | 约 0.6249 |
| 冻结首帧 | 约 0.4034 |

其中第四球的完全不变性最严重；平移、缩放和 swap 仍偏高，说明 union subject 与
independent normalization 能把结构性错误线性救回。

### 5.5 Contact 并未被验证

`event_frame = argmin(|s_striker - s_target1|)` 总会返回一帧，即使：

- 两球从未表面接触；
- 两球穿透；
- target_1 与 target_2 的传播顺序错误；
- 最近距离发生在视频边界；
- 轨迹大部分来自插值。

可信接触至少要联合：

```text
surface gap
+ closing → contact → separation velocity pattern
+ minimum duration / temporal neighborhood
+ striker→target_1→target_2 contact graph
```

无法确认接触时应给低 event confidence，而不是制造一个确定碰撞帧。

当前实现还有两个内部语义不一致：

- event frame 用 `striker↔target_1`；
- effective restitution 的 closing/separation speed 却用
  `striker↔target_2`。

三球链式碰撞应分别建立 `0→1` 和 `1→2` 的接触事件与局部碰前/碰后窗口，不能混用
不同对象对。

另外 `minimum_motion_span_px` 当前取三对象所有时空点在拟合轴上的总范围。即使三个球
各自完全静止，只要初始彼此分开，总范围也能超过门槛。v5 应使用每角色相对自身初始
位置的时间位移和速度证据，并单独要求 striker 碰前运动及接触后的状态变化。

## 6. 下一版验收标准

v4 保持冻结。新的对象中心协议只有满足以下条件后才可替代它：

1. Condition/初始窗口冻结每个物理 ID，不用完整未来 GT 重新匹配；
2. 逐角色连续位置、形状、尺寸和外貌进入正式分；
3. 同时运行 expected-ID tracking 与 residual discovery；
4. 对 missing、extra、duplicate、fragmentation 和 ID switch 逐帧计 exposure；
5. 静态 track anchor、球半径和 contact graph 进入 collision state；
6. 遮挡、出画、未出现和 evaluator unknown 有不同语义；
7. 通过平移、缩放、删除、复制、第四球、role swap、无接触、穿透、错误接触顺序和
   中途消失的 metamorphic tests；
8. 对多个 Baseline 的高/中/低分 Case 完成人工盲审相关性验证。

通用对象合同、匹配与集合评分的 shadow 实现和五场景设计见：

```text
docs/OBJECT_CENTRIC_EVALUATION.md
src/physbench/evaluation/common/entities/
```
