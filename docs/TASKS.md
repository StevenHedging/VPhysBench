# Task v1

Task 是模型无关、可冻结的工作负载声明。发行版只接受 `schema_version: "1.0"`，规范位于
`schemas/task.schema.json`。

## 官方任务

- `tasks/official/six_scene_direct_eval_v1.json`
  - Task ID：`six_scene_direct_eval_v1`
  - 训练：无
  - 评估：六场景全部 775 个 case
- `tasks/official/six_scene_train_six_scene_eval_v1.json`
  - Task ID：`six_scene_train_six_scene_eval_v1`
  - 训练：六场景 679 个 View A train case
  - 评估：六场景 96 个 View A ID test case

二者都绑定 Dataset `physics_video_seven_scene_v14`、推理种子 42 和协议
`scene_default_v1`。

`push_bottle` 的 127 个训练 case 和 14 个 test case 仍保留在 Dataset 中，但不被任何
官方 Task v1 选择。

## 字段

所有 Task 都包含：

- `task_id`、`family`、`dataset_id`；
- `selection`：只声明数据选择，不声明模型输入或训练实现；
- `seeds.training` 与 `seeds.inference`；
- `evaluation.protocol` 与 `evaluation.reporting`。

`direct_eval` 的 selection 使用 `evaluation_scene_ids`、`groups` 和可选 `case_ids`。
`finetune_eval` 使用彼此独立的 `training_scene_ids`、`evaluation_scene_ids` 与
`test_regimes`。Dataset view 由 family 唯一推导，因此 v1 不再重复保存
`dataset_view`。

Task 禁止模型名称、checkpoint、prompt 改写、物理量注入方式、runner 命令和训练超参。
这些均属于 Baseline bundle 与 DataAdapter。

## 编译与覆盖

planner 先将 Task 展开为 canonical plan。计划中的 `training_scene_ids` 和
`train_case_ids` 描述训练侧；`scene_ids` 和 `jobs` 描述评估侧。每个 job 固定
`case_id`、`scene_id`、partition 和 inference seed。

Baseline 内可以编写并注册它自己的训练/推理脚本，但只能消费 canonical plan。通用编排
层调用 `TaskBuilder.compile(dataset, task, canonical_plan)`；Baseline 编译器没有重新规划
Task 的入口。

只编译而不运行：

```bash
physbench task-build \
  --dataset datasets/releases/14.0.0/dataset.json \
  --task tasks/official/six_scene_direct_eval_v1.json \
  --baseline baselines/my_model \
  --output task_instance.json
```

对多个 Baseline 运行同一 Task：

```bash
physbench matrix-run \
  --dataset datasets/releases/14.0.0/dataset.json \
  --task tasks/official/six_scene_direct_eval_v1.json \
  --baseline baselines/model_a \
  --baseline baselines/model_b \
  --matrix-id comparison \
  --output-root run
```

矩阵会比较 canonical plan 签名；训练场景、训练 case、评估场景、jobs 或种子不同都会
拒绝配对。计划或 dry run 不是已完成的 benchmark 结果。
