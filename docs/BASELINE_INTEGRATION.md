# Baseline 接入

> v2 的唯一任务编译入口是 Baseline 内置的 `TaskBuilder`。它把不可变的
> `DatasetSnapshot + TaskSpec` 编译为当前 Baseline 可运行、带摘要封印的
> `BaselineTaskInstance`。统一 `DataAdapter` 是 TaskBuilder 的私有协作者，不是与
> TaskBuilder 并列的 Benchmark 顶层部件。详细设计见
> [TaskBuilder 架构](TASK_BUILDER_ARCHITECTURE.md)和
> [统一 Data Adapter 架构](DATA_ADAPTER_ARCHITECTURE.md)。

Benchmark 的公共要素仍只有 `Dataset`、`Task` 和 `Baseline`。TaskBuilder 属于
Baseline 的实现，不引入第四种领域对象。

## v2 编译与执行契约

```text
dataset 目录 + task.json + baseline.json
             │  文件 facade 负责加载、校验和快照化
             ▼
DatasetSnapshot + TaskSpec
             │
             ├─ Benchmark planner：唯一负责 View A/B 与 train/ID/OOD 选择
             ▼
CanonicalTaskPlan
             │
             ├─ Baseline.TaskBuilder
             │    ├─ 能力与场景兼容性检查
             │    ├─ 内部 DataAdapter：空间/时间/范式/文本/物理适配
             │    ├─ 训练与推理 job 编译
             │    ├─ artifact/cache 绑定
             │    └─ operation DAG 组装
             ▼
sealed BaselineTaskInstance
             │  BaselinePlugin.run_task(instance=...)
             ▼
Trainer / Predictor → Predictions → Benchmark Metrics
```

这个边界有三条硬约束：

1. TaskBuilder 不自行重算数据划分。模型无关的 planner 先生成
   `CanonicalTaskPlan`，因此不同 Baseline 不会悄悄选择不同 train/ID/OOD case。
2. `build()` 必须确定性且无训练/推理副作用。相同 Dataset、Task 和 Baseline
   配置应产生相同实例摘要。
3. Trainer/Predictor 只消费已经封印的实例；执行前校验实例摘要、Baseline 摘要和
   TaskBuilder 指纹，不能重新读取并解释原始 task 文件。

对应的最小插件形态是：

```python
class MyBaselinePlugin(BaselinePlugin):
    task_builder: MyTaskBuilder

    def run_task(self, *, instance, run_dir, execute, stop_after_training):
        instance.verify()
        # Trainer / Predictor 只根据 instance 执行
        ...
```

### TaskBuilder 输入

Python 接口接收已经加载并校验的：

- `DatasetSnapshot`：无提示词的数据描述、case、View A/B 索引、scene schema、
  asset root 与数据摘要；
- `TaskSpec`：任务族、conditioning、选择规则、seed 和任务摘要；
- `BaselineBundle` 由 TaskBuilder 构造时持有，不作为 Dataset 或 Task 的内容泄漏。

CLI 提供等价的文件 facade：

```bash
PYTHONPATH=src python -m physbench task-build \
  --dataset datasets/physics_v1/dataset.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/task_instance.json
```

`task-build` 只做加载、规划、编译和封印，不训练、不推理。

### TaskBuilder 输出

`BaselineTaskInstance` 的公共 envelope 至少冻结：

- Dataset / Task / Baseline / TaskBuilder / DataAdapter 指纹；
- Benchmark 生成的 canonical plan；
- 本次被选中的源 case 快照；
- 可审计 adaptation records；
- 可选 training spec 和 inference jobs；
- `train → infer → evaluate` 或 `infer → evaluate` operation DAG；
- 内容寻址缓存绑定；
- 对 Benchmark 不透明的 `baseline_payload` 与 `native_inputs`；
- `instance_digest`，用于发现落盘或内存中的篡改。

Task 1 的推理 job 通过符号引用 `artifact://train/model` 依赖本实例的训练产物；
Task 2 使用 `baseline://frozen_model`，从而避免把某台机器的临时 checkpoint 路径
硬编码进通用任务语义。

### Baseline bundle

v2 的 `components` 至少包含：

```json
{
  "components": {
    "task_builder": {
      "type": "my_model_task_builder_v1",
      "config": {
        "data_adapter": {
          "spatial": {},
          "temporal": {},
          "paradigm": {},
          "text": {},
          "physics": {}
        }
      }
    },
    "trainer": {"type": "my_trainer_v1", "config": {}},
    "predictor": {"type": "my_predictor_v1", "config": {}}
  }
}
```

顶层 `data_adapter` 和 `condition_adapter` 会被 v2 registry 拒绝。物理适配也不被
统一规定为提示词：WAN 可以把结构化参数写入文本，其他 Baseline 可以生成 token、
tensor 或 control stream；Benchmark 只审计适配记录，把 `native_inputs` 视为不透明值。

### 执行与产物

`atomic-run` 不绕过 TaskBuilder。它先构建并验证实例，再冻结实例的完整 manifest 和
可读投影，最后才交给 Baseline 执行：

```text
runs_v2/<run_id>/
├── frozen/                         # Dataset / Task / Baseline 快照
├── task_builder.json               # Builder 类型、指纹和内部适配器描述
├── data_adapter.json               # 便于审计的内部 DataAdapter 描述
├── task_instance/
│   ├── manifest.json               # sealed BaselineTaskInstance
│   ├── canonical_plan.json
│   ├── adaptations.jsonl
│   ├── training.json
│   ├── inference_jobs.jsonl
│   ├── execution_graph.json
│   ├── cache_bindings.json
│   └── baseline_payload.json
├── artifacts/
├── predictions.jsonl
└── evaluation/
```

不加 `--execute` 会完成编译、冻结和模型侧计划；加 `--execute` 才实际训练/推理。

```bash
PYTHONPATH=src python -m physbench atomic-run \
  --dataset datasets/physics_v1/dataset.json \
  --task tasks/official/direct_eval_generic.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

`generic` 和 `physics` 是两份独立 AtomicTask。对 `finetune_eval`，二者各自产生一个
BaselineTaskInstance 和独立模型产物；`matrix-run` 只负责验证二者共享数据划分、
case、seed 等配对条件，不会让它们共享训练后的 Adapter。

## v1 兼容输入契约

以下内容仅描述保留的 v1 `run` 接口。新 Baseline 不应以此替代 TaskBuilder。

v1 baseline 配置声明 adapter、输入视图和训练能力。核心框架不假设模型是 T2V 还是 I2V。

推理 job 是 JSON 对象，至少包含：

```json
{
  "job_id": "...",
  "case_id": "...",
  "scene_id": "pendulum",
  "prompt_profile_id": "physics_natural",
  "input_view": "i2v",
  "model_input": {
    "prompt": "...",
    "first_frame": "..."
  },
  "output_video": "predictions/<prompt_profile_id>/<job_id>.mp4"
}
```

训练 job 提供 train case ID、完整 case 记录、冻结 PromptProfile 和已解析提示词的路径。
外部模型负责自己的媒体预处理，但必须使用 `resolved_prompts.jsonl` 中选定的训练
profile，并把生成结果写回 `predictions.jsonl`，字段遵循 prediction schema。

在 v1 中，一个 run 只训练或加载一次模型/Adapter。同一 case 可以展开为 `generic` 与
`physics_natural` 两个推理 job；它们共享 checkpoint、首帧、seed 和推理超参，仅
`model_input.prompt` 与输出路径不同。

## v1 Adapter

- `dummy`：不生成视频，只输出占位 prediction，用于验证整个 Benchmark。
- `command`：把冻结 job JSON 传给外部命令。命令执行默认关闭，需调用方显式授权。
- `wan22_lora`：原生适配 WAN2.2-TI2V-5B + LoRA，负责只读媒体归一化、Task 1 微调、Task 2 冻结 checkpoint 和 TI2V 推理；详见 `WAN22_LORA_BASELINE.md`。
- 原生 Python 模型：继承 `BaselineAdapter` 并注册到 `baselines/registry.py`。

训练 checkpoint、日志和预处理缓存应放在 run 目录的 `artifacts/` 下，不能回写原始数据目录。
