# VPhysBench CSTI 通用轨迹评估设计

## 目标

在保留五个正式 Scene 专家 evaluator 及其现有主分数的前提下，为 VPhysBench
增加一个独立的通用轨迹评估维度 `CSTI`。CSTI 复用专家 evaluator 已经得到的
GT/预测物理主体掩码轨迹和角色匹配，在每个 GT 主体上计算累计三维 Soft Tube
IoU，再对全部 GT 主体等权平均，得到 Case CSTI；Scene 和 Task 继续采用严格覆盖
聚合。

CSTI 的完整算法含义是 Cumulative Spatio-Temporal Soft Tube IoU。公开结果使用
短名称 `csti`，避免把长算法名扩散到结果和接口字段中。

## 当前状态与兼容边界

- 实现基线位于提交 `c20e495` 之后；该提交已保存此前工作区状态。
- 当前正式协议是 `scene_default_v10`，五个正式 Scene 为 pendulum、collision_1d、
  inclined_plane_slide、uniform_circular_motion 和 parabolic_motion。
- v13 的 916 个 Case 全部具有自身的 `assets.reference_video`，物化出的
  `reference_capability` 全部是 `same_case_gt`；没有活动的 `physics_parent` 或
  `annotation_only` Case。
- 当前 `resolve_physics_reference` 只解析当前 Case 的 `assets.reference_video`，
  不再解析 parent Case。仓库内残留的 physics-parent 枚举、分支和测试仅用于旧格式
  兼容及历史协议回归。
- `scene_default_v10`、现有官方 Task 和历史结果保持不变。CSTI 通过新增
  `scene_default_v11` 和新的官方 Task 变体启用，避免原地改变历史协议语义。

## 非目标

- 不修改或替换任何 Scene 专家评分算法。
- 不把 CSTI 与专家分数相加、相乘或加权成新的主分数。
- 不重新运行一套 CSTI 专用检测、分割、追踪或身份匹配。
- 不对多余预测主体施加第二次 CSTI 惩罚；它们继续由 object-centric integrity
  处理。
- 不把生成视频拉伸到 GT 时长，也不通过 CSTI 增加时长不足惩罚。
- 不支持 physics-parent 的未来像素 Tube 评分。
- 不降低或统一缩放现有分析掩码的空间分辨率。
- 不在首版中引入近似距离变换、未来帧距离场复用或 hard-IoU 累计递推。
- 不把原始掩码 Tube 写入 Case 结果。

## 已确认的评分语义

1. CSTI 是独立评估维度；`CaseEvaluationResult.score` 继续是专家分数。
2. GT entity manifest 是分母。每个 GT 物理主体必须贡献一个物体分数。
3. 整段没有匹配到预测轨迹的 GT 实体直接得 0 分。
4. 多余预测实体不进入 CSTI 分母。
5. GT 与生成视频只比较公共有效时间区间：
   `T_eval = min(T_gt, T_pred)`，沿用 v10 的 physical-overlap 时间协议。
6. 每个物体的累计曲线 `p(t)` 使用严格算术平均，不使用真实时间梯形积分。
7. Case CSTI 是所有 GT 实体物体分数的严格算术平均。
8. 使用现有共享无填充坐标系中的原生分析掩码，不做额外 resize。
9. 正式参数为 `temporal_weight=1.0`、`tolerance_radius=0.05`。
10. 只有 `same_case_gt` 可以计算；历史 physics-parent/annotation-only 输入返回
    CSTI `not_applicable`。

## 架构选择

采用“Scene 输出已匹配 Tube，总评估器统一计算”架构：

```text
ReferenceCaseEvaluator
├── 统一视频解码、公共时间网格和共享空间坐标系
├── 调用现有 Scene evaluator 的 analyze
│   ├── 现有 GT/预测主体观察
│   ├── 现有角色和身份匹配
│   ├── 现有专家评分
│   └── 新增 csti_input（仅内存）
├── 公共 CSTI scorer
│   ├── 校验 GT manifest 完整覆盖
│   ├── 逐实体、逐前缀计算精确 Soft Tube IoU
│   └── 生成 Case CSTI 及诊断曲线
└── CaseEvaluationResult
    ├── score：现有专家分数
    └── metrics.csti：独立 CSTI 维度
```

拒绝以下替代方案：

- 各 Scene 自行调用公共数值函数：会复制实体遍历、空 Tube 和聚合语义。
- 总评估器重新观察和匹配视频：会重复昂贵推理，并产生与专家 evaluator 不一致
  的主体身份。

## 模块边界

新增一个小型公共包：

```text
src/physbench/evaluation/common/csti/
├── __init__.py
├── contracts.py
├── metric.py
└── adapters.py
```

### `contracts.py`

只定义内存数据契约和结构验证，不依赖 Scene：

```text
CSTIEntityTube
├── entity_id: str
├── role_id: str
├── reference_masks: tuple[np.ndarray, ...]
├── prediction_masks: tuple[np.ndarray, ...] | None
└── matched_prediction_track_ids: tuple[str, ...]

CSTIInput
├── reference_capability: ReferenceCapability
├── times_s: tuple[float, ...]
├── frame_shape: tuple[int, int]
└── entities: tuple[CSTIEntityTube, ...]
```

`prediction_masks=None` 只表示整个时间段没有匹配身份。已经匹配的实体在个别帧
缺失时，`prediction_masks` 仍存在，对应帧使用空掩码。

现有 observer 的二值 Mask 通常以 `{0,255}` 保存。适配层在创建上述契约时使用
`mask > 0` 无损转成 bool，前景像素集合不变；float 概率图在该边界被拒绝，不能
借此引入隐式阈值。数值核心因此始终只接收 bool 或严格 `{0,1}` 整数数组。

### `metric.py`

只实现：

- 单个二值 Tube 前缀的 soft membership 和 Soft IoU；
- 单实体累计 `p(t)` 曲线；
- 多实体 Case 聚合；
- `metrics.csti` 的纯数据结果。

该模块不了解摆球、碰撞块、轨道或抛射体，也不访问视频、manifest、文件系统或
Scene 配置。

### `adapters.py`

提供从 `ExpectedEntityTimeline`、`OpenWorldObservation` 和现有 frame matches
构造逐实体预测掩码序列的公共工具。它不执行 assignment；只把已经存在的
`entity_id ↔ track_id` 匹配物化为 Tube。

Scene 特有的变量选择仍在相应 evaluator 内完成，不建立一个包含五个 Scene 分支
的中央适配器。

### `SceneAnalysis`

为 `SceneAnalysis` 增加默认值为 `None` 的 `csti_input` 字段。它只在
`ReferenceCaseEvaluator.evaluate` 内消费，不被 `asdict` 或 Case JSON 直接
序列化，因此大型掩码不会写入结果。

旧协议未启用 CSTI 时，各 Scene 可以继续返回 `None`，行为完全不变。

## CSTI 数值定义

### 输入

对一对已匹配实体，输入为：

```text
G, P ∈ {0,1}^[T,H,W]
```

- `T` 是公共物理时间区间内的采样帧数。
- `H,W` 是现有 Scene 分析掩码分辨率。
- 接受 bool 或值域严格为 `{0,1}` 的整数掩码。
- Scene 适配层将现有 `{0,255}` 二值观察结果按 `mask > 0` 转成 bool。
- 不直接接受概率图；Scene 继续提供现有二值分割结果，不新增阈值超参数。

### 严格累计前缀

对每个 `t ∈ [0,T-1]`：

```text
G_t = G[:t+1]
P_t = P[:t+1]
p(t) = soft_tube_iou(G_t, P_t)
```

`p(t)` 只能依赖当前及过去掩码。禁止计算完整视频距离场后截取前缀，因为未来
前景会改变早期背景体素的最近距离。

### 归一化三维距离

```text
d = sqrt(
    (dx / (W-1))^2
  + (dy / (H-1))^2
  + temporal_weight^2 * (dt / (T-1))^2
)
```

NumPy Tube 的轴顺序是 `[T,H,W]`，EDT sampling 固定为：

```text
(
  temporal_weight / max(T-1, 1),
  1 / max(H-1, 1),
  1 / max(W-1, 1),
)
```

时间轴归一化始终使用完整评估长度 `T`，不能使用当前 prefix 长度，否则容差会随
时间改变。

### Soft membership

对非空二值 Tube `A`：

```text
D_A = scipy.ndimage.distance_transform_edt(~A, sampling=spacing)
S_A = max(0, 1 - D_A / tolerance_radius)
```

前景内部为 1，距离在容差半径内线性衰减，超出半径为 0。

### Soft IoU

```text
p(t) = sum(min(S_G, S_P)) / sum(max(S_G, S_P))
```

空前缀必须显式处理，不依赖 SciPy 对全空数组的边界行为：

- 双方为空：`p(t)=1`；
- 仅一方为空：`p(t)=0`；
- 双方非空：计算 Soft IoU。

### 物体与 Case 聚合

```text
object_score_i = mean(p_i(0), ..., p_i(T-1))
case_csti = mean(object_score_1, ..., object_score_N)
```

`N` 是 GT manifest 实体数。整段 unmatched 的实体直接得到 `object_score=0`、
`curve=null`，避免生命周期尚未开始时的双空前缀使 unmatched 实体获得分数。

## 输入契约与公共校验

`ReferenceCaseEvaluator` 在调用 scorer 前重新物化 GT manifest，并检查：

- `CSTIInput.reference_capability` 与 manifest 一致；
- `same_case_gt` 下 `entities` 的 ID 集合与 manifest 完全相等；
- 实体无重复、无未知 ID，输出顺序规范化为 manifest 顺序；
- 每个 reference Tube 长度等于 `len(times_s)`；
- matched prediction Tube 长度与 reference 相同；
- 所有掩码 shape 等于共享 `frame_shape`；
- 掩码为二维、二值、非空数组对象；单帧掩码允许没有前景像素；
- 所有对象共享同一 `T,H,W`；
- `T>=1`，时间戳数量匹配且单调，但时间戳不参与最终算术平均；
- scorer 输出有限且落在 `[0,1]`。

这些是 evaluator 内部契约，不做自动 resize、自动截断、概率阈值猜测或实体跳过。

## 五个正式 Scene 的适配

### Pendulum v7

- GT：`reference_bobs`。
- 预测：`comparison.matched_masks`，其中 `None` 转为空掩码。
- 匹配 provenance：`comparison.matches` 中出现的 track ID。
- manifest 中的摆球是唯一 CSTI 实体。
- string/support 保持为 topology/apparatus 证据，不作为独立 CSTI 实体。

### Collision 1D v5

- GT：`reference_masks[entity_index][frame_index]`。
- 预测：从 `prediction_objects` 按 `(track_id, frame_index)` 索引 detection mask，
  再按 `comparison.matches` 写入相应 `entity_id` 的 Tube。
- 现有 `_matched_subject_masks` 的跨实体 union 只保留给旧 subject 诊断，不能用于
  CSTI。
- 每个碰撞物体分别得分；不同 GT 实体之间绝不 union。
- 同一 GT 实体在一帧内若现有匹配语义合法地产生多个片段，则只在该实体内部
  OR；重复 reference/prediction slot 仍视为匹配契约错误。

### Inclined Plane v7

- GT：`reference.masks`。
- 预测：`result.matched_prediction_masks`。
- 实体：`reference.entity_id` 对应唯一滑动物体。
- 合法离开画面后的空掩码按 CSTI 空前缀规则自然处理，不延长生命周期。

### Uniform Circular Motion v7

- GT 顺序：`frozen_reference.entity_ids`。
- GT Tube：`frozen_reference.instance_masks[i]`。
- 预测 Tube：`prediction_tracks.instance_masks[i]`。
- 匹配 provenance：`comparison.matches`。
- `matched_prediction_union` 继续只用于旧诊断，不用于 CSTI。
- 数组全空且整个时间段无 match 的实体必须显式标记 unmatched，得 0 分。

### Parabolic Motion v1

- GT：`reference.masks`。
- 候选预测：`prediction.masks`。
- 只有 `scored["binding"]["accepted"]` 为真时才建立 matched Tube。
- binding 失败时实体直接 unmatched、CSTI 为 0。
- 不用理论抛物线、拟合中心或 CSTI 专用插值补掩码。

## 协议设计

新增：

```text
configs/evaluation/protocols/scene_default_v11.json
```

它复制 v10 的五 Scene 专家、时间、空间、robustness 和可视化配置，只在协议顶层
新增：

```json
{
  "general_metrics": {
    "csti": {
      "enabled": true,
      "algorithm": "exact_prefix_edt",
      "temporal_weight": 1.0,
      "tolerance_radius": 0.05,
      "prefix_aggregation": "arithmetic_mean",
      "case_aggregation": "mean_gt_entities",
      "timeline_policy": "physical_overlap",
      "mask_resolution": "scene_analysis_native"
    }
  }
}
```

`schemas/v3/evaluation_protocol.schema.json` 增加严格 CSTI 配置定义。参数必须完整且
匹配首版支持的固定算法选择，不提供没有测试语义的额外开关。

`SceneEvaluatorRegistry` 将协议顶层 `general_metrics` 作为只读公共配置合并进
实例化时的 Scene evaluator config，使其进入 evaluator fingerprint；不在五个 Scene
JSON 块中复制同一配置。

## Task 选择与迁移

保留现有：

```text
tasks/official/five_scene_direct_eval.json
tasks/official/five_scene_finetune_eval.json
```

它们继续指向 v10，保证历史命令和结果可复现。

新增：

```text
tasks/official/five_scene_direct_eval_csti.json
tasks/official/five_scene_finetune_eval_csti.json
```

新 Task 的 ID 分别使用：

```text
five_scene_direct_eval_v13_csti
five_scene_finetune_eval_v13_csti
```

它们与当前正式 Task 使用同一 v13 Dataset、View、Scene selection 和 seed，仅将
evaluation protocol 切换到 `scene_default_v11`。README/运行文档同时明确：旧 Task
复现专家-only v10，新 Task 产生专家+CSTI 双维结果。

## Case 结果契约

正常 evaluated Case：

```json
{
  "score": 0.81,
  "metrics": {
    "csti": {
      "status": "evaluated",
      "score": 0.73,
      "algorithm": "exact_prefix_edt",
      "parameters": {
        "temporal_weight": 1.0,
        "tolerance_radius": 0.05
      },
      "frame_count": 3,
      "frame_shape": [480, 270],
      "aggregation": {
        "prefixes": "arithmetic_mean",
        "entities": "mean_all_gt_entities"
      },
      "objects": [
        {
          "entity_id": "object_1",
          "role_id": "moving_block",
          "matched": true,
          "matched_prediction_track_ids": ["track_3"],
          "score": 0.76,
          "curve": [1.0, 0.94, 0.91]
        }
      ]
    }
  }
}
```

- 顶层 `score` 仍是专家分数。
- `metrics.csti.score` 是独立 Case CSTI。
- objects 按 manifest 顺序排列。
- matched 实体保存完整 `p(t)` 曲线；unmatched 实体保存 `curve=null`。
- 不保存原始 Tube。
- 核心 scorer 不依赖 matplotlib；现有 visualization 可在未来选择消费曲线。

历史不适用输入使用：

```json
{
  "status": "not_applicable",
  "score": null,
  "reason_code": "csti_requires_same_case_gt"
}
```

## 失败语义

### Reference 失败

GT manifest、参考视频或参考主体观察失败时，沿用现有 Case `unavailable`。不伪造
CSTI，也不把 reference failure 记为模型零分。

### Prediction 失败

现有 robustness 策略会把 prediction record 缺失、prediction incomplete、部分预测
媒体/观察失败转为 evaluated zero。CSTI 必须同步产生 evaluated zero：

- 通过 GT manifest 列出全部实体；
- 每个实体 `matched=false`、`score=0`、`curve=null`；
- Case CSTI 为 0；
- 保存原有 degradation code/reason。

这要求 Task preflight 的 `_prediction_zero_result` 接收 Case，并在 v11 启用 CSTI 时
调用公共 zero-result 构造器；`ReferenceCaseEvaluator._degraded_output` 使用同一
构造器。不能因 evaluator 提前返回而漏掉失败样本的 CSTI。

### 内部契约失败

实体遗漏、重复、未知 ID、非二值掩码、shape/长度不一致、非有限结果或超界分数
均是 evaluator `error`。不静默跳过、不改成 not_applicable，也不用零分隐藏实现
错误。

## Scene 与 Task 聚合

### 归一化维度记录

把每个 Case 先映射为与具体 metric 无关的维度记录：

```text
DimensionCaseRecord
├── job_id
├── scene_id
├── evaluation_partition
├── status
└── score
```

- expert 记录直接来自 Case 顶层 status/score；
- CSTI 记录来自 `metrics.csti`；
- Case 非 evaluated 时，CSTI 记录继承 Case status；
- CSTI `not_applicable` 明确保留，并从该维度的 expected 分母中排除。

`aggregate_task_results` 的现有 Scene、partition、generalization 和严格覆盖逻辑
抽取为一个纯维度聚合 helper。expert 和 CSTI 均调用它，避免两套宏平均规则随时间
漂移。重构必须用回归测试证明 v10 顶层聚合输出语义不变。

### Scene CSTI

Scene CSTI 是该 Scene 全部适用 Case CSTI 的算术平均。只有全部适用 Case 都成功
产生分数时发布正式 `score`；否则 `score=null`，同时提供
`observed_mean_score`。

### Task CSTI

Task CSTI 是选定 Scene CSTI 的宏平均，每个 Scene 权重相等。finetune Task 继续
复用当前 partition 聚合政策。任一必需 Scene CSTI 不完整时，正式 Task CSTI 为
null。

### Task JSON

保留现有顶层 expert 字段，并增加：

```json
{
  "score": 0.81,
  "dimensions": {
    "expert": {
      "score": 0.81,
      "source": "top_level_score"
    },
    "csti": {
      "status": "complete",
      "score": 0.73,
      "coverage": 1.0,
      "by_scene": {},
      "breakdown": {}
    }
  }
}
```

顶层 `score`、`by_scene`、`breakdown` 和 generalization diagnostics 继续表示 expert
维度，旧消费者无需修改。`dimensions.csti` 是完整独立聚合，不生成 expert+CSTI
复合分数。`dimensions` 只在协议启用至少一个 general metric 时输出，因此 v10
Task JSON 的字段集合和历史语义保持不变。

若某个 Scene 没有任何 CSTI-applicable Case，该 Scene 维度状态为
`not_applicable`，且不进入 Task CSTI 的 Scene 分母；若全部 Scene 均不适用，则
Task CSTI 同样为 `not_applicable`。当前 v13 正式 Task 不会进入这些分支。

## 性能与内存策略

首版的正式实现同时也是 correctness reference：每个实体、每个 prefix 独立调用
精确 EDT。复杂度约为 `O(T^2 H W)`，因此必须明确测量，但不能为了速度静默改变
指标。

实现时采用以下有限的无损措施：

- 一次只处理一个实体和一个 prefix；
- 每个实体只规范化并堆叠一次完整二值 Tube，随后使用 prefix view，不为每个
  prefix 重复复制二值 Tube；
- 显式短路双方空、单方空和整段 unmatched；
- 不缓存所有 prefix 的三维 membership；
- 每个 prefix 得到标量后立即释放 EDT 工作数组；
- 结果只保留 `p(t)` 标量曲线。

发布前在单实体、双实体、最高分辨率和最长时间轴 Case 上记录 wall time 与峰值
RSS。正式最大 Case 必须能够完成且不发生 OOM；耗时数据如实进入性能报告，不在
本设计中猜测一个脱离运行环境的固定秒数阈值。后续优化必须：

1. 保留本参考实现；
2. 使用独立 backend 名称；
3. 在完整测试矩阵和代表性真实 Case 上证明数值等价；
4. 不通过 resize、改变容差或未来距离场复用获得加速。

首版不设置猜测性的超时、像素上限或自动降采样 fallback。

## 测试设计

### 数值单元测试

新建 `tests/test_csti_metric.py`，覆盖：

1. 完全相同 Tube 的整条曲线和最终分数均为 1。
2. 双方全空为 1；仅一方为空为 0。
3. unmatched 实体始终为 0，不受早期双空前缀影响。
4. 近距离偏移得分高于远距离偏移。
5. 空间偏移增加时分数单调不增。
6. 时间延迟增加时分数单调不增。
7. 提高 `temporal_weight` 会加重相同时间偏移。
8. tolerance radius 增大时近邻 Tube 分数不下降。
9. 不同物体宽度、轨迹断点和间歇缺失的预期行为。
10. `T=1`、`H=1` 或 `W=1` 的安全归一化。
11. bool 与 `{0,1}` uint8 语义一致。
12. 概率值、负值、错误维度、shape/长度不一致被拒绝。
13. 所有输出有限并落在 `[0,1]`。
14. 因果性：两个同长度视频拥有相同前 k 帧但不同未来时，前 k 个 `p(t)` 完全
    相同。
15. 手工构造的小 Tube 与独立直接 EDT 参考计算一致。

### 契约与适配测试

新建 `tests/test_csti_adapters.py`，覆盖：

- manifest 顺序和完整覆盖；
- GT 实体遗漏、重复和未知实体失败；
- frame match 正确取出 `(track_id, frame)` 掩码；
- 现有 `{0,255}` observer Mask 无损规范化为 bool，float Mask 被拒绝；
- 某帧缺失为空，但整段未匹配为 `None`；
- collision 多实体保持独立，不意外 union；
- 同实体碎片合并与重复 slot 拒绝；
- circular 冻结实体顺序；
- parabolic binding 成功/失败边界；
- native mask shape 不被改变。

### Case evaluator 集成测试

对五个正式 evaluator 增加最小集成断言：

- v11 evaluated Case 同时产生专家 score 与 `metrics.csti`；
- self-comparison CSTI 接近或等于 1；
- 将一个 GT 实体置为 unmatched 后只影响对应物体和 Case CSTI；
- expert score 字段与相同输入的 v10 结果不变；
- prediction observation fail-closed 产生 CSTI 0；
- synthetic physics-parent 返回 CSTI not_applicable；
- `csti_input` 不出现在序列化结果。

### Task 聚合测试

扩展 `tests/test_task_evaluator.py` 或建立 `tests/test_csti_aggregation.py`：

- Case → Scene → Task 的两层均值；
- Case 数量不均衡时仍按 Scene 宏平均；
- finetune partition 策略与 expert 相同；
- 缺少一个必需 CSTI 时正式分数为 null、observed mean 仍存在；
- not_applicable 从 CSTI expected 分母排除；
- prediction preflight zero 进入 CSTI 覆盖和均值；
- v10 没有 CSTI 时历史顶层聚合结构和数值保持不变。

### 协议与 Schema 测试

- `scene_default_v11` 通过 v3 schema。
- 缺少 CSTI 固定参数、非法半径或未知算法时拒绝协议。
- 新 Task 只在 protocol 和 Task ID 上与现有 v13 Task 有预期差异。
- protocol/evaluator fingerprint 包含 CSTI 配置。

### 性能基准

增加非默认 CI 的 benchmark/脚本，至少报告：

- 小型合成 Tube；
- 一个正式单实体 Case；
- collision 多实体 Case；
- v11 中最大 `T×H×W` Case；
- 单 Case wall time 和 peak RSS。

性能结果是发布审计材料，不在 core scorer 中加入基于机器速度的分支。

## 文档与可审计性

README 或 evaluator 文档应说明：

- expert 与 CSTI 的定义和独立关系；
- CSTI 的 GT 实体分母、unmatched 零分及 extra prediction 策略；
- physical-overlap、不惩罚时长差和算术时间平均；
- fixed parameters、算法 ID 和协议 fingerprint；
- 如何从 `metrics.csti.objects[*].curve` 定位轨迹开始偏离的时刻；
- v10 与 v11 Task 的复现用途。

Case provenance 记录 CSTI 算法 ID、参数、公共时间策略、frame count/shape 和
manifest materializer ID。匹配 track IDs 记录在 object 结果中；不重复保存视频或
Mask 像素。

## 实施顺序

1. 先以测试驱动方式实现纯 CSTI 数值核心和数据契约。
2. 实现公共 OpenWorld match→Tube 适配器。
3. 在 `SceneAnalysis`/`ReferenceCaseEvaluator` 接入内存输入和公共后处理。
4. 依次接入 pendulum、collision、inclined plane、circular 和 parabolic。
5. 补齐 prediction degraded/preflight zero。
6. 泛化 Task 聚合器并加入 dimensions 输出，同时锁定 v10 回归。
7. 增加 v11 protocol/schema 和 CSTI Task 变体。
8. 更新文档、运行单元/集成/全量协议校验。
9. 最后运行代表性真实性能基准；任何优化另立经过等价验证的后续工作。

## 验收标准

实现只有同时满足以下条件才算完成：

1. v10 官方 Task 的 expert 数值和顶层聚合语义没有改变。
2. v11 的五个正式 Scene 都为每个 `same_case_gt` Case 产生 CSTI。
3. 每个 Case 的 CSTI objects 与 GT manifest 一一对应且顺序稳定。
4. unmatched GT 实体为 0；extra prediction 不直接进入 CSTI。
5. 所有 CSTI 数值落在 `[0,1]`，且满足因果性、空 Tube 和单调性测试。
6. CSTI 使用公共时间区间、算术 prefix 平均、原生掩码和固定正式参数。
7. Task 同时输出兼容的 expert 顶层结果与独立 `dimensions.csti`。
8. robust prediction zero 不从 CSTI 覆盖中消失；reference failure 不被误判为模型
   零分。
9. 结果中不包含原始 Tube，核心 scorer 不依赖绘图库。
10. 协议、schema、单元、适配、Case 集成和 Task 聚合测试全部通过；正式最大
    Case 的基准能够完成且不发生 OOM，wall time 与 peak RSS 已记录。
