# 五场景 Baseline 对照评测（2026-07-29）

## 1. 结论与结果边界

本轮完成了 quantity-embedding 实验所缺少的 generic、structured-text physics 和
direct-only Baseline 对照。所有模型预测均已生成：

| 任务视图 | Baseline 数 | 每个 Baseline 的视频 | 生成状态 |
| --- | ---: | ---: | --- |
| View A / `five_scene_finetune_eval_v5` | 2 | 66 / 66 | complete |
| View B / `five_scene_direct_eval_v5` | 4 | 214 / 214 | complete |

所有结果都由冻结的 `scene_default_v2` 协议产生，协议 fingerprint 为
`84733a984480b4ec8a66b2d31976bd72ddca0a284f525b9d66525436e88dfe88`。由于主体检测或
跟踪存在失败，六个新对照的 coverage 均不足 1，严格 Task score 全部为 `null`。
下文的 observed mean 只描述成功评测的 Case，不能当作正式排名分数。

最重要的解释限制：

- View A 与完整 View B 的分布和训练条件不同，不能直接比较两列分数；
- View B 的 `group_1`–`group_5` 是确定性均衡分组，不是 ID/OOD 标签；
- G15 的训练 trial 与 benchmark v3 来源重叠，因此两个 G15 结果均为
  `diagnostic_pretrained`，不得进入正式 leaderboard；
- coverage 不同会造成 observed-only selection bias，数值高低不能证明算法优劣；
- structured-text physics 在本轮三个 matched pair 中没有表现出稳定、跨分布的一致增益。

## 2. View A：支持 fine-tune 的 WAN2.2 对照

两组 v5 对照都从相同 WAN2.2-TI2V-5B base fresh fine-tune，不续训历史三场景 LoRA。
每组使用 121 个 unique train Case、290 个 scene-balanced metadata rows、10 epochs、
1450 optimizer steps，以及 training/inference seed 42。

| Baseline | 物理信息策略 | evaluated | coverage | observed macro mean | strict score |
| --- | --- | ---: | ---: | ---: | --- |
| WAN2.2 generic v3 | 忽略结构化 physics | 45 / 66 | 0.681818 | 0.371597 | `null` |
| WAN2.2 physics v3 | 追加结构化 physics 文本 | 44 / 66 | 0.666667 | 0.408185 | `null` |

作为同一 66-Case View A 上的历史参照，quantity-embedding run 在 alternate v2
reevaluation 中为 46/66、coverage 0.696970、observed macro mean 0.414271、strict
score `null`。它使用任务版本 v4 和较早的冻结 bundle；因此可以检查量级，但本轮不能
据此宣称 quantity-embedding 胜过 structured-text physics。

### 2.1 逐 scene

单元格为 `observed mean (evaluated / expected)`。

| scene | WAN generic | WAN physics |
| --- | ---: | ---: |
| pendulum | 0.540223 (13/13) | 0.591396 (13/13) |
| free_fall | 0.459475 (3/4) | 0.528552 (2/4) |
| collision_1d | 0.145959 (4/21) | 0.186233 (4/21) |
| inclined_plane_slide | 0.274555 (21/22) | 0.269246 (21/22) |
| uniform_circular_motion | 0.437775 (4/6) | 0.465498 (4/6) |

### 2.2 失败原因

| reason code | WAN generic | WAN physics |
| --- | ---: | ---: |
| `insufficient_instance_tracks` | 2 | 2 |
| `insufficient_valid_masks` | 8 | 9 |
| `collision_frame_zero_objects_missing` | 6 | 6 |
| `collision_striker_missing` | 4 | 4 |
| `insufficient_vertical_motion` | 1 | 1 |
| **error total** | **21** | **22** |

## 3. View B：direct-only Baseline 完整任务

| Baseline | 物理信息策略 | eligibility | evaluated | coverage | observed macro mean | strict score |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Cosmos3 generic | 忽略结构化 physics | regular direct-eval | 152/214 | 0.710280 | 0.264278 | `null` |
| Cosmos3 physics | 追加结构化 physics 文本 | regular direct-eval | 148/214 | 0.691589 | 0.306835 | `null` |
| G15 generic | 忽略结构化 physics | diagnostic pretrained | 154/214 | 0.719626 | 0.395411 | `null` |
| G15 physics | 追加结构化 physics 文本 | diagnostic pretrained | 150/214 | 0.700935 | 0.385329 | `null` |

### 3.1 完整 View B 的逐 scene 结果

单元格为 `observed mean (evaluated / expected)`。

| scene | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 0.444269 (40/40) | 0.416810 (40/40) | 0.504268 (39/40) | 0.525477 (39/40) |
| free_fall | 0.279006 (8/11) | 0.474926 (2/11) | 0.400371 (9/11) | 0.307556 (8/11) |
| collision_1d | 0.219374 (10/32) | 0.215279 (10/32) | 0.278901 (8/32) | 0.295579 (9/32) |
| inclined_plane_slide | 0.268060 (86/95) | 0.291134 (88/95) | 0.295508 (85/95) | 0.287915 (80/95) |
| uniform_circular_motion | 0.110682 (8/36) | 0.136027 (8/36) | 0.498009 (13/36) | 0.510118 (14/36) |

### 3.2 失败原因

| reason code | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: |
| `circular_disk_not_found` | 18 | 18 | 19 | 18 |
| `insufficient_instance_tracks` | 10 | 10 | 4 | 4 |
| `insufficient_valid_masks` | 16 | 22 | 18 | 22 |
| `collision_frame_zero_objects_missing` | 7 | 7 | 9 | 8 |
| `collision_striker_missing` | 5 | 5 | 5 | 5 |
| `collision_target_pair_missing` | 1 | 1 | 1 | 1 |
| `insufficient_vertical_motion` | 3 | 1 | 1 | 3 |
| `insufficient_incline_motion` | 2 | 2 | 3 | 3 |
| **error total** | **62** | **66** | **60** | **64** |

## 4. View B 在 View A 66-Case 测试集上的对齐诊断

direct-only Baseline 不支持 View A 的训练阶段，因此没有伪造 fine-tune。这里从每个
完整 View B `case_results.jsonl` 中按冻结的 `views/view_a.json` 精确抽取同一 66 个
test Case，并恢复其 `test_id` / `test_ood1` 标签。分数仍来自同一次 View B 生成，
只是统计切片。

| Baseline | evaluated | coverage | aligned observed macro mean | strict score |
| --- | ---: | ---: | ---: | --- |
| Cosmos3 generic | 43/66 | 0.651515 | 0.341301 | `null` |
| Cosmos3 physics | 43/66 | 0.651515 | 0.323598 | `null` |
| G15 generic | 43/66 | 0.651515 | 0.421138 | `null` |
| G15 physics | 44/66 | 0.666667 | 0.399152 | `null` |

对齐 observed macro mean 使用与 View A 相同的层次：先求 partition 中成功 Case 的
均值，再求 scene 的 partition 宏平均，最后求五个 scene 的宏平均。没有成功 Case 的
partition 不会被伪造为零；这也意味着它仍然只是 observed-only 诊断。

### 4.1 对齐逐 scene

| scene | Cosmos generic | Cosmos physics | G15 generic | G15 physics |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 0.484087 (13/13) | 0.446950 (13/13) | 0.568154 (13/13) | 0.558346 (13/13) |
| free_fall | 0.529403 (2/4) | 0.547386 (1/4) | 0.464561 (3/4) | 0.419003 (3/4) |
| collision_1d | 0.369682 (7/21) | 0.255808 (7/21) | 0.197104 (4/21) | 0.194690 (5/21) |
| inclined_plane_slide | 0.221019 (19/22) | 0.264823 (20/22) | 0.291225 (19/22) | 0.277808 (19/22) |
| uniform_circular_motion | 0.102313 (2/6) | 0.103021 (2/6) | 0.584646 (4/6) | 0.545911 (4/6) |

## 5. 可复现资产

矩阵与 AtomicRun：

```text
runs_v2/quantity_controls_viewa_v5_seed42_20260729.matrix.json
runs_v2/quantity_controls_viewa_v5_seed42_20260729__*/
runs_v2/quantity_controls_viewb_v5_seed42_20260729.matrix.json
runs_v2/quantity_controls_viewb_v5_seed42_20260729__*/
```

每个 AtomicRun 都保存 `run.json`、`frozen_cases.jsonl`、`predictions.jsonl`、
`evaluation/case_results.jsonl`、`evaluation/case_metrics.jsonl` 和
`evaluation/summary.json`。View A 的两个 run 还保存各自 fresh fine-tune 的 checkpoint、
loss 曲线和 8-worker 推理汇总。

本报告只读取冻结结果，没有重写 run 或原有 v2 evaluation。后续评估协议变化必须使用
新的版本化 protocol 和 reevaluation variant，不得覆盖这里的 canonical v2 结果。
