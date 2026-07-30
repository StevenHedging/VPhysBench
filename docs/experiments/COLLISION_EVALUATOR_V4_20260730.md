# 碰撞评估器 v4：多帧角色观测与过程可视化

日期：2026-07-30

## 1. 结论

`scene_default_v4` 将一维碰撞 evaluator 从 `1.3` 升级到 `1.4`，其余四个 scene
继续使用 `1.3`。冻结 View B 的 32 个真实碰撞 reference 在完整
多帧检测、SAM2 分割、轨迹和接触事件链路下从 v3 的 12/32 提升到：

```json
{"cases": 32, "observable": 32, "failed": 0, "coverage": 1.0}
```

本轮只发布新的可选评估协议，不覆盖官方 v6 Task 固定的 `scene_default_v3`，也不
重写任何既有 v3 reevaluation。

## 2. 从 Jensen Eval2 借鉴和修正的内容

`/root/Jensen/Eval2/eval_video.py` 提供了有价值的工程思路：

- 多实例使用共同 seed frame；
- 在同一 SAM2 state 中传播持久实例 ID；
- 重叠像素由最高 SAM2 logit 决定唯一实例；
- 输出逐实例曲线、CSV 和叠加视频，便于人工复核。

Benchmark 没有直接复制其通用实例相似度。碰撞 Case 的三个球具有不可交换的物理
角色，因此 v4 做了以下适配和加固：

- 使用 `striker/target_1/target_2` 固定角色，不以自由匹配掩盖身份交换；
- 不要求 frame 0 颜色显著或三个球全部可稳定分割；
- 在多个早期帧上使用多阈值 Hough 圆证据，并以 temporal-median motion proposal
  作为 fallback；
- 允许非零共同 seed，并执行 SAM2 正向和反向传播；
- 传播后按 seed、连通分量面积和质心连续性保留单一角色组件；
- 保留 Benchmark 已有的真实时间轴归一化、三球状态、动量残差、恢复系数和
  主体位置/形状/外貌评分；
- prediction 观测失败继续记有限零分，reference 失败仍明确归到数据/评估侧。

## 3. 协议与实现

协议：

```text
configs/evaluation/protocols/scene_default_v4.json
```

协议 fingerprint：

```text
54cefd0a75dc927e783ba7bcd8c75719b6c8b619fc07dce055eb50f94ca60532
```

碰撞画布从 `640 × 360` 提高到 `960 × 540`。多帧 seed 搜索覆盖视频前 60%，最多
12 个候选帧。角色选择要求：

- striker 位于左侧并与 target pair 分离；
- `target_1/target_2` 相邻；
- 三球中心近似共线；
- 半径比例、可见性和间距落在协议边界内。

Hough 证据不可用时才尝试 temporal-median motion proposal。共同 seed 的三个
SAM2 prompt 各有五个正点和一个扩展 box。

实例传播使用：

```text
shared non-zero seed
→ forward propagation
→ reverse propagation
→ highest positive logit wins per pixel
→ role-seeded continuous component stabilization
```

共享 SAM2 适配器新增的互斥模式默认关闭，避免改变 v1–v3 的历史 frame-0 单向传播和
独立二值化行为。

## 4. 质量门与评分

v4 reference hard gate：

- 单个 mask 最少 12 像素；
- 单实例面积最多为画布的 2%；
- 三角色各自有效率至少 0.20；
- 轨迹运动跨度至少 30 像素。

32 个 View B reference 中最低角色有效率为：

```text
0.2567567568
collision_r2_medium_steel_medium_steel_medium_steel_v07184
```

因此 0.20 是由冻结数据审计支持的 reference 可识别下界，不是为 prediction 放宽的
得分捷径。prediction 只要不能形成足够轨迹就按 v3 记 `evaluated/0/degraded`；能形成
轨迹但覆盖率较低时：

```text
observation_reliability =
    min_role clip(prediction_valid_ratio / reference_valid_ratio, 0, 1)

state_score = raw_state_score × observation_reliability
```

主体位置、形状、外貌分和最终 case 权重不变。

## 5. 可视化与存储

每个 v4 碰撞 Case best-effort 生成：

```text
collision_observation.mp4
collision_instance_similarity.png
collision_event_timeline.png
collision_tracks.csv
collision_observation.json
```

默认外置根：

```text
/mnt/nvme1/physics_video_benchmark/evaluation_visualizations
```

仓库入口：

```text
visualizations
  -> /mnt/nvme1/physics_video_benchmark/evaluation_visualizations
```

视频是 2×2 面板：

1. reference 三角色 mask、质心和历史轨迹；
2. prediction 对应视图；
3. union mask 的 reference-only、prediction-only 和 intersection；
4. 逐帧 IoU、seed、接触帧与角色状态 dashboard。

正式 Case artifact 目录只保存
`collision_visualization_manifest.json`。该清单含每个外部文件的绝对路径、仓库链接
路径、大小、SHA-256 和 evaluator config digest。外部诊断不进入 sealed
AtomicRun/reevaluation manifest；生成失败不改变 case score。

## 6. 回归结果

### 6.1 Reference 可观测性

审计命令：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python \
  scripts/audit_collision_evaluator_v4.py \
  --device cuda \
  --output visualizations/scene_default_v4/\
collision_reference_observability_audit.json
```

报告：

```text
visualizations/scene_default_v4/collision_reference_observability_audit.json
```

报告 SHA-256：

```text
260a49430ca1e20e9e8f4925567ebb9ae1cdf1cdaa20e03cf541e7c082ad0d16
```

结果：

```text
32/32 observable
0 failed
coverage = 1.0
```

### 6.2 完整 identity

Case：

```text
collision_r2_large_steel_medium_steel_small_steel_v01313
```

reference 和 prediction 分别独立运行 81 帧媒体采样、多帧角色发现、双向 SAM2、
稳定化、轨迹、事件和主体评分。使用未经运行时改写的 `scene_default_v4`
（`sam2.device=auto`，实际解析为 CUDA）：

```text
status = evaluated
score = 0.9999999998013178
physics_state = 1.0
subject = 0.9999999996688631
observation_reliability = 1.0
reference role valid ratios = [1.0, 1.0, 0.9382716049]
prediction role valid ratios = [1.0, 1.0, 0.9382716049]
```

可视化入口：

```text
visualizations/scene_default_v4/
collision_r2_large_steel_medium_steel_small_steel_v01313/
identity_v4_official_20260730-898a747f58a2ec24/
```

该 bundle 的 evaluator config digest：

```text
729be6d42678e0952e46c94d87c62646a550feb9888c7b17ab63b4cf4671488a
```

### 6.3 自动化回归

新增 `tests/test_evaluation_protocol_v4.py`，覆盖：

- v3/v4 协议与 evaluator 版本隔离；
- schema 接受 `collision_1d_state_v3`；
- striker 不在 frame 0 时选择后续共同 seed；
- 非零 seed 的 SAM2 正向和反向传播；
- v4 多实例 mask 互斥；
- v3 frame-0 独立阈值行为不变；
- 语义角色交换不能得到 identity 分；
- 外置可视化与本地哈希清单。

新增测试 7/7 通过；v2、v3 和既有 scene evaluator 的相关回归 31/31 通过。完整仓库
测试为 232/232 通过。

## 7. 已知边界

- v4 修复的是 reference 可观测性和碰撞角色稳定性，不代表所有生成视频都会被可靠
  分割；异常 prediction 会得到低分或有限零分。
- SAM2 本机扩展缺失时会跳过可选的 mask hole filling 后处理；当前回归证明主传播
  和评分仍可运行，但部署环境和权重 identity 必须继续记录。
- 外置可视化是诊断资产，不是可替代 Case result 的权威分数来源。
- v4 尚未成为官方 Task 默认协议。正式比较必须在相同 v4 fingerprint 下重评所有
  Baseline 后再发布 leaderboard。
