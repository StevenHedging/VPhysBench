# 新实验情景 Evaluator 接入指南

## 1. 目的与前置条件

本文规定如何为一个新增实验情景实现、注册和验收 evaluator，供 AI 助手和真人共同
执行。数据导入、XLSX 字段整理、视频清洗和 Dataset 发布不在本文重复说明；开始前必须
先完成并通过 [数据集说明与使用手册](../datasets/README.md) 的数据验收。

Evaluator 的目标不是判断视频“看起来像不像”，而是回答三个可审计的问题：

1. 冻结 Case 声明的物理主体是否仍然存在，身份是否保持，是否出现新增、复制、合并或
   非法消失；
2. 匹配主体在共同物理时间轴上的状态如何偏离真实参考；
3. 该 scene 特有的物理过程是否与真实参考一致。

每个新 evaluator 必须满足两个硬门槛：

- Dataset 的全部正式 test Case 都能返回规范结果，错误预测只能得到结构化降级或 0 分，
  不能触发 evaluator error；
- 对每个具有 same-case GT 的 Case，将 GT 视频作为自身 prediction 时，主分必须为 1，
  或在确有不可避免的数值误差时极接近 1。

## 2. 推荐的公共框架

```text
Case entity manifest
        │
        ├── reference/condition 首帧：建立冻结主体声明与场景坐标系
        │
        ├── prediction 首帧：独立实例发现
        │                         │
        └──────── 带拒绝选项的主体绑定 ────────┐
                                               │
                         全程追踪 + residual 实例发现
                                               │
                         共同物理时间轴与 lifecycle
                                               │
                         scene-specific 状态距离
                                               │
                         主体分 + 物理分 + 完整性门
```

这个框架有意修正“GT 与生成视频首帧必然完全相同，因此可直接复制 GT mask”的假设。
I2V 模型可能对首帧进行 VAE 重建、缩放、padding、颜色改变或轻微形变，也可能删除、
复制或合并主体。Prediction 必须独立观察；GT 的 mask、轨迹或未来位置不得直接充当
prediction detector 的答案。

“首帧语义分割”也不是强制所有 scene 使用同一个模型。正确要求是：利用 manifest 中
的 `entity_class` 做可解释的实例发现，并输出 mask、位置、尺度、外观描述子和置信度。
可以使用 SAM2，也可以使用更适合该 scene 的圆/线/颜色/几何 detector；若采用多个
proposal source，必须去重并保留来源审计。

## 3. 必须冻结的契约

### 3.1 Entity manifest

先调用：

```python
from physbench.evaluation.common.entities import materialize_entity_manifest

manifest = materialize_entity_manifest(case)
```

不要从 prompt、文件名或背景颜色重新猜主体数量。Manifest 至少决定：

- `entity_id`、角色和类别；
- 物理属性绑定；
- `persistent`、`may_exit` 等 lifecycle；
- 多主体是否可交换；
- apparatus 与主体的区别。

若 Dataset 的 manifest 无法无歧义表达该 scene，先修订 Dataset schema 和导入逻辑，
不要在 evaluator 内写文件名特例。

### 3.2 共同时间轴

Reference 与 prediction 必须在同一时间点采样。使用 reference-bounded 时间轴和
`build_common_time_grid` 的物理时间权重，不按“第 k 帧对第 k 帧”比较不同 FPS 的
视频。Prediction 较短时，缺少的时间 cell 明确记为 unavailable/missing，不复制末帧，
也不外推轨迹。

主指标不得使用无约束 DTW 将明显提前、滞后或停滞的运动重新对齐成高分。若 scene
确实存在未知相位，可定义有物理边界的相位对齐，并把容许范围写入协议。

### 3.3 场景坐标系

坐标系只能由 condition/reference 的可观测装置或几何建立并冻结，例如：

- 单摆的 pivot、string axis；
- 碰撞轨道的纵轴；
- 圆周运动的圆心和半径；
- 平抛的图像水平轴、重力方向与发射点。

Prediction 中漂移或变形的装置不得重新定义原点、尺度或 ROI，否则模型可以通过移动
坐标系规避误差。

## 4. 标准实现步骤

### 步骤 1：先做全量数据审计

至少统计全部 Case 的分辨率、FPS、时长、主体数、首帧位置、运动方向、可见覆盖率、
末帧 lifecycle 和结构化标注字段。逐条查看首/中/末帧；数据量很大时可以高比例抽样，
但全部正式 test 必须覆盖。

先记录 detector 必须处理的真实变体，再定阈值。不能只围绕一条“最干净”的视频开发。

### 步骤 2：定义观察输出

每个主体 track 至少保留：

```text
xy[t]                    主体中心或 scene-specific 状态
mask[t]                  实例 mask
observed[t]              是否有真实观测
visibility[t]            visible / occluded / exited / missing
area_or_scale[t]          尺度
appearance_descriptor[t] 外观描述子
confidence[t]             观测置信度
source[t]                 proposal/segmenter 来源
```

同时保留每个时间 cell 的完整候选数量。匹配主体之外的 plausible candidate 不能静默
丢弃，它们是 duplicate/new entity 的证据。

### 步骤 3：独立发现并绑定首帧主体

Reference 与 prediction 分别运行 detector。绑定成本通常组合：

```text
class compatibility
+ normalized position distance
+ log scale difference
+ shape difference
+ appearance distance
+ topology/role compatibility
```

多主体使用一对一最小成本分配；同规格主体按 manifest 的 exchangeability 规则处理。
绑定必须允许 null/reject。没有可信对应主体时应记录 `missing`，不能把最近背景物体硬绑
成主体。

### 步骤 4：追踪和开放世界观察

已绑定主体用条件身份、短时运动和外观连续性追踪。每帧还要做 residual discovery，
并显式记录：

- missing/illegal disappearance；
- duplicate/new entity；
- merge/split；
- identity switch；
- legal exit；
- detector overflow 或低置信观测。

只允许对短内部 gap 做有界插值；插值 cell 必须有标记。Prediction 提前出画不能继承
GT 的 `may_exit` 资格，GT 推断出的合法 exit 时刻才是 reference lifecycle。

### 步骤 5：定义 scene-specific 距离

像素 IoU 只能作为分割诊断，不应是唯一主分。每个 scene 应先定义物理状态，再定义
归一化误差和有界相似度。例如：

```text
similarity(error; scale) = exp(-0.5 * (error / scale)^2)
```

尺度必须写进 protocol，不得按某个 baseline 的结果临时调节。组合权重之和应可校验，
所有分量和主分都必须落在 `[0, 1]`。

GT 是经验物理参考。理想方程拟合可作为重要诊断，但真实实验的噪声、透视和空气阻力
不应导致 GT-self 低于 1。可采用 reference-relative 方式：比较 prediction 与 GT 各自
的线性/二次拟合曲线和残差；二者完全相同时该分量为 1，同时单独报告理想定律拟合度。

### 步骤 6：统一输出

Task-facing 主指标固定使用：

```text
scene_subject_state_similarity
```

并至少输出：

- `<scene>_state_similarity`：scene-specific 物理分及分解；
- `physical_subject_similarity`：绑定、形状和外观；
- `object_centric_integrity`：主体数量、缺失、新增、ID switch；
- `physical_subject_mask_iou`：仅诊断；
- `entity_manifest`；
- `quality`、`provenance` 和逐帧审计 artifact。

建议 artifact 包含 `per_frame.csv`、轨迹图、主体相似度曲线和 `open_world_audit.json`。

## 5. 失败责任与 fail-closed 规则

| 失败来源 | 正式状态 | 分数 | 处理 |
| --- | --- | ---: | --- |
| Prediction 记录缺失/未完成 | `evaluated` | 0 | 结构化 degradation code |
| Prediction 视频缺失、损坏、过短 | `evaluated` | 0 或缺失 cell 计 0 | 不形成 evaluator error |
| Prediction 分割/追踪失败 | `evaluated` | 0 或保守缺失分 | 保留失败 provenance |
| Reference 媒体或标注不可用 | `unavailable` | N/A | Dataset/reference 责任，必须修数据或 observer |
| Evaluator 内部编程错误 | `error` | N/A | 不得伪装成 prediction 失败 |

Prediction 侧异常不应变成 N/A。Reference 侧失败也不能默默给 0，因为这会把评估器或
Dataset 的错误归咎于模型。正式发布前必须消灭 test 上的 reference unavailable。

## 6. 文件与注册位置

新增 scene 通常需要：

```text
src/physbench/evaluation/scenes/<scene>/
  __init__.py
  evaluator.py

src/physbench/evaluation/registry.py
configs/evaluation/protocols/scene_default_v<N>.json
schemas/v2/evaluation_protocol.schema.json
tests/test_<scene>_evaluator.py
```

协议是冻结身份。不要修改已经用于正式结果的旧协议；复制最新协议为新版本，只添加
新 type/config，并保持旧 scene 配置逐字段不变。Evaluator type 必须同时出现在 schema
与 `SceneEvaluatorRegistry.supported_evaluator_types()`，两者集合有自动测试。

## 7. 测试与验收矩阵

单元测试至少覆盖：

- 理想合成轨迹；
- GT observation 与自身比较严格为 1；
- 平移、速度错误、错误加速度和错误几何会降分；
- 主体缺失、复制、新增、提前出画和 ID switch；
- 空视频、坏编码、低 FPS、短视频和部分时间轴；
- detector 抛异常时 prediction fail-closed；
- registry 和 schema 路由。

真实验收必须覆盖：

1. 全部正式 test 的 GT-self；
2. 最好覆盖该 scene 的全部 Dataset Case；
3. 至少一个真实 baseline prediction，确认不是只有 GT shortcut 才能运行；
4. 汇总中 `errors=0`、`unavailable=0`、GT-self 最低分达到约定阈值；
5. 人工查看若干逐帧 artifact，确认 tracker 没有稳定跟错背景物体。

当前通用审计入口为：

```bash
PYTHONPATH=src python \
  scripts/audit_open_world_evaluator_v6.py \
  --dataset datasets/releases/12.0.0/dataset.json \
  --protocol scene_default_v10 \
  --scene parabolic_motion \
  --self-check \
  --output audits/<audit_id>
```

审计期间不得修改 evaluator、protocol 或 schema。报告中的 source-tree digest 必须显示
`stable_during_run=true`；修改后必须从头重跑。

## 8. 平抛 evaluator v1 作为示例

平抛的 manifest 声明一个 `may_exit` ball。Observer 用圆形几何和局部环形背景对比对
深浅背景保持极性无关，Prediction 首帧独立发现并通过位置、尺度、外观进行拒绝式绑定。
时间轴固定为 reference-bounded 32 FPS 公共采样，GT 真实可见尾部定义合法 lifecycle。

物理分由五项组成：

- 时间参数化二维轨迹；
- 水平近似匀速；
- 竖直近似匀加速；
- 二维抛物线几何；
- lifecycle 与主体数量完整性。

水平、竖直和抛物线拟合均采用 reference-relative 得分，理想定律拟合度另作诊断。因此
真实 GT 的透视和实验噪声仍会被展示，但 GT-self 的每个计分分量严格为 1。

## 9. 发布前检查清单

- [ ] 已阅读数据导入规范并审计全部正式 test；
- [ ] Manifest 无歧义，主体与 apparatus 分开；
- [ ] Prediction 首帧独立发现，不复制 GT mask；
- [ ] 绑定允许 reject/null，多主体分配符合 exchangeability；
- [ ] 时间轴基于物理时间，缺帧不外推；
- [ ] missing/new/duplicate/exit/ID switch 全部可审计；
- [ ] scene-specific 距离、尺度、权重写入新协议；
- [ ] Prediction 失败 fail-closed，Reference 失败不冒充模型低分；
- [ ] 合成、异常、registry/schema 测试通过；
- [ ] 全量 GT-self 无 error、无 unavailable，最低分达标；
- [ ] 至少一个真实 prediction 完成非退化评分；
- [ ] 审计 source-tree digest 在运行期间稳定；
- [ ] 旧协议和旧结果未被覆盖。
