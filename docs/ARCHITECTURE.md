# Architecture

VPhysBench 把可比较性、模型实现和评分权限分给五个所有者：

```text
Dataset → Task planner → canonical plan → Baseline compiler/runtime → Evaluator
                                      ↘ AtomicRun audit trail ↗
```

## 所有权

- Dataset 拥有 case、输入与参考资产、物理量、训练/测试视图和不可变数据摘要。
- Task 拥有工作负载语义：训练与评估场景、case/group 选择、种子、协议和报告策略。
- Benchmark planner 是选择权威。它把 Dataset 与 Task 展开为 canonical plan；所有
  Baseline 对同一输入必须得到相同的训练 case 和推理 job。
- Baseline 拥有模型身份、输入策略、DataAdapter、训练/推理脚本和执行依赖。它只能把
  已冻结计划编译为 BaselineTaskInstance，不能重新选择 case 或替换协议。
- Evaluator 独占参考资产和评分注释，并按 Evaluation v1 产生 case 结果与 Task 汇总。
- AtomicRun 冻结上述身份、摘要、计划、适配记录、预测、日志和评估产物。

## Task 与 Baseline 脚本的边界

Task 需要保留，因为它是跨 Baseline 的公共实验单位，也是复现、配对比较、覆盖率和审计
的依据。允许每个 Baseline 在注册目录内提供它能执行的训练/推理脚本；这些脚本属于
实现层，不等价于 Task。若把数据选择也放入脚本，不同 Baseline 就可以无意或有意地运行
不同样本，最终分数失去可比性。

实际调用顺序是：

1. `load_task` 校验模型无关的 Task v1；
2. `plan_atomic_task` 由基准展开 Dataset view，生成 canonical plan；
3. 通用编排层把 plan 交给 Baseline 的 `TaskBuilder.compile`；
4. DataAdapter 只为计划内 case 构造模型原生输入；
5. Trainer/Predictor 只执行封印后的 BaselineTaskInstance；
6. Evaluator 根据 canonical jobs 检查覆盖并评分。

`TaskBuilder` 不再暴露自行调用 planner 的 `build` 方法。这保证“任务选择”在上游一次性
完成，而 Baseline 仍可自由实现真正影响模型行为的代码。

## 场景集合解耦

canonical plan 明确保存两个集合：

- `training_scene_ids`：Baseline 训练能力必须覆盖的场景；
- `scene_ids`：推理、聚合和计分场景。

七场景训练、五场景评估不再依赖含糊的单一 `scene_ids`。命令行 `--scene-id` 只缩小
评估集合，不会静默改变官方七场景训练集合。Baseline 的 `supported_scenes` 同时校验这
两个集合。

## 发行边界

发行树不含已注册 Baseline bundle、模型实现/配置/权重或运行输出。保留
`baseline_api` 和 `baseline_runtime` 是为了让外部 Baseline 能按同一个协议接入，并不
代表仓库集成了任何算法。Dataset 资产由冻结的 Hub revision 提供，运行输出只能写入
`run/<run_id>/`。
