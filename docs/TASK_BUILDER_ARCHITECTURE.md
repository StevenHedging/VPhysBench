# Baseline-owned TaskBuilder 架构

## 1. 目标

每个 Baseline 都内置一个 `TaskBuilder`。它接收 Dataset 与 Task 配置，把模型无关的
实验定义编译为该 Baseline 可以直接执行、审计和复现的任务实例：

```text
TaskSpec → CanonicalTaskPlan → BaselineTaskInstance → AtomicRun
```

TaskBuilder 解决的是“相同 Benchmark 任务如何落到不同模型原生接口”这一编译问题。
它不是新的 Benchmark 领域要素，而是 Baseline 内部的强制组件。

设计遵循五个不变量：

1. **选择权属于核心**：View A/B、train/ID/OOD case 和 seed 由 Benchmark 生成
   `CanonicalTaskPlan`，Baseline 不得重新解释；
2. **适配权属于 Baseline**：空间、时间、输入范式、文本与物理注入均由目标 Baseline
   决定；
3. **构建无副作用**：TaskBuilder 只编译 JSON 任务，不编码、训练或推理；
4. **执行只认实例**：Trainer/Predictor 的执行入口是封印后的
   `BaselineTaskInstance`，不再临时读取原始 Task 来改变语义；
5. **结果可校验**：Dataset、Task、Baseline、Builder、Adapter 和实例都有稳定 digest
   或 fingerprint。

## 2. 三层契约

### 2.1 TaskSpec：用户意图

TaskSpec 只表达模型无关语义：

- `family`: `finetune_eval` 或 `direct_eval`；
- `conditioning`: `generic` 或 `physics`；
- `dataset_view`: View A 或 View B；
- scene/group/case/partition 选择；
- training/inference seed；
- 可选 OOD2 组合。

TaskSpec 不包含 WAN prompt、分辨率、帧数、LoRA 参数或模型路径。

### 2.2 CanonicalTaskPlan：核心选择结果

`plan_atomic_task(task, dataset)` 由 Benchmark 核心执行，冻结：

```text
dataset_id + dataset_digest
task_id + family + conditioning
scene_ids
train_case_ids
training_seed
jobs[job_id, case_id, scene_id, partition, conditioning, seed]
```

所有 Baseline 对同一 Dataset/Task 都从相同计划出发。这样可以公正比较模型，也避免
某个 Baseline 在适配时偷偷筛掉困难 case。

### 2.3 BaselineTaskInstance：模型可执行任务

目标 Baseline 的 TaskBuilder 将 canonical plan 与自身静态 bundle 编译为原生任务
实例。该实例同时包含跨 Baseline 可读的公共 envelope 和只允许目标 Baseline 解释的
`baseline_payload`。

## 3. 所有权与依赖方向

| 对象 | 所有者 | 可以依赖 | 不应依赖 |
|---|---|---|---|
| DatasetSnapshot | Dataset | 资产、结构化事实、View 索引 | prompt、WAN、LoRA |
| TaskSpec | Task | Dataset ID、任务语义 | 模型输入格式、Adapter |
| CanonicalTaskPlan | Benchmark 核心 | DatasetSnapshot、TaskSpec | 具体 Baseline |
| TaskBuilder | Baseline | bundle、canonical plan、私有 DataAdapter | 修改 Dataset/View |
| BaselineTaskInstance | 编译产物 | 三个 snapshot 与原生输入 | 未冻结的运行时选择 |
| BaselinePlugin | Baseline | 封印实例 | 重新选择 case 或 conditioning |
| Evaluator | Benchmark 核心 | Dataset case、prediction、metric config | WAN 私有兼容文件 |

依赖方向保持单向：

```text
Dataset ─┐
         ├─> Core Planner ─> CanonicalTaskPlan ─┐
Task ────┘                                      ├─> Baseline TaskBuilder
Baseline bundle ────────────────────────────────┘
                                                     │
                                                     ▼
                                            BaselineTaskInstance
                                                     │
                                                     ▼
                                             Baseline execution
                                                     │
                                                     ▼
                                                Evaluation
```

## 4. 接口

核心抽象位于 `src/physbench/baseline_api/interfaces.py`。

### 4.1 TaskBuilder

```python
class TaskBuilder(ABC):
    bundle: BaselineBundle
    data_adapter: DataAdapter

    @property
    def fingerprint(self) -> str: ...

    def describe(self) -> dict: ...

    def build(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
    ) -> BaselineTaskInstance:
        canonical_plan = plan_atomic_task(task, dataset)
        return self.compile(dataset, task, canonical_plan)

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance: ...
```

`build()` 是统一模板方法：核心 planner 总是在编译前运行。子类实现 `compile()`，
但不能替换 planner。

### 4.2 BaselinePlugin

```python
class BaselinePlugin(ABC):
    bundle: BaselineBundle
    task_builder: TaskBuilder

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict, list[dict]]: ...
```

执行器必须先验证 instance digest、目标 Baseline digest 与 TaskBuilder fingerprint。
它只解析自己的 `baseline_payload` 和 `native_inputs`。

### 4.3 文件级 facade

`build_task_instance()` 位于 `src/physbench/orchestration/atomic_runner.py`，负责从三个
配置路径加载 snapshot，定位 Baseline plugin，调用其 TaskBuilder 并验证实例。这是
CLI 和其他调用方的稳定文件级入口。

## 5. 编译阶段

一个合格的 TaskBuilder 按以下次序工作：

1. **Compatibility check**
   - family 与 conditioning 是否受支持；
   - canonical plan 中的 scene 是否受支持；
   - `direct_eval` 所需冻结 checkpoint 是否存在。
2. **Source binding**
   - 只收集 canonical plan 中 train/eval 涉及的 case；
   - 记录 Dataset asset root，但不复制或修改原始资产。
3. **Case adaptation**
   - 分 train/eval role 调用私有 DataAdapter；
   - 记录 `adaptation_id`、五阶段审计和 opaque `native_inputs`。
4. **Training compilation**
   - `finetune_eval` 生成 trainer spec、seed、case/adaptation 引用；
   - `direct_eval` 的 `training` 明确为 `null`。
5. **Inference compilation**
   - 为 canonical job 绑定 adaptation、model ref 与 native input；
   - 不改变 job ID、partition 或 seed。
6. **Graph and cache binding**
   - 生成 operation DAG；
   - 绑定由媒体策略与 Dataset digest 决定的不可变缓存。
7. **Seal**
   - 写入所有 identity/fingerprint；
   - canonical JSON 求 SHA-256，得到不可变实例。

## 6. 实例 envelope

当前 `schema_version=2.1`，schema 位于
`schemas/v2/baseline_task_instance.schema.json`：

```text
BaselineTaskInstance
├── schema_version
├── instance_id
├── instance_digest
├── identity
│   ├── dataset {dataset_id, digest}
│   ├── task {task_id, digest}
│   ├── baseline {baseline_id, digest}
│   ├── task_builder {type, fingerprint}
│   ├── data_adapter {fingerprint, materialization_fingerprint}
│   └── canonical_plan_digest
├── semantics {family, conditioning, scene_ids}
├── canonical_plan
├── source {asset_root, cases}
├── adaptations[]
├── training | null
├── inference {predictor, jobs[]}
├── execution_graph {operations[]}
├── cache_bindings[]
└── baseline_payload
```

公共 envelope 应保持稳定；未来 Baseline 的专有字段应放入 `baseline_payload` 或
`native_inputs`，而不是不断扩展核心 schema。

## 7. 不可变封印与防篡改

`BaselineTaskInstance.seal()` 的规则是：

1. 移除已有 `instance_digest`；
2. 以 UTF-8、键排序、无多余空白的 canonical JSON 序列化；
3. 对该字节串计算 SHA-256；
4. 将 digest 写回文档，再保存 canonical JSON。

`value` 属性每次从 canonical JSON 反序列化，返回新的对象；外部代码修改该对象不会
改变实例本体。`from_document()` 用于加载磁盘文档，并拒绝 recorded digest 与重新
计算结果不一致的文件；`verify()` 在执行前再次校验。

该机制防止“审核的是 A，执行的却是被内存代码静默改写后的 B”。它不取代资产文件
本身的逐文件哈希；未来可在 Dataset 层补充逐资产 SHA-256。

## 8. Operation DAG 与符号模型引用

TaskBuilder 显式生成执行依赖，而不是依赖目录是否恰好存在：

```text
finetune_eval

┌───────┐   artifact://train/model   ┌───────┐          ┌──────────┐
│ train │ ─────────────────────────> │ infer │ ───────> │ evaluate │
└───────┘                            └───────┘          └──────────┘

direct_eval

baseline://frozen_model              ┌───────┐          ┌──────────┐
───────────────────────────────────> │ infer │ ───────> │ evaluate │
                                     └───────┘          └──────────┘
```

两种保留引用的含义：

- `artifact://train/model`：同一实例中 `train` operation 声明的模型输出；
- `baseline://frozen_model`：Baseline bundle 中配置的冻结基础模型或 Adapter。

符号引用使 TaskBuilder 保持无副作用，也让构建出的实例不依赖尚未创建的 run 路径。
Baseline executor 负责在执行期把符号引用解析为真实 artifact。

## 9. WAN2.2 + LoRA 实现

WAN 实现位于 `src/physbench/baseline_plugins/wan22.py`：

- `Wan22TaskBuilder`：兼容性检查、适配、train/infer spec、DAG、缓存与封印；
- `Wan22DataAdapter`：五阶段 case 适配；
- `Wan22BaselinePlugin.run_task()`：验证实例并运行 WAN 私有执行逻辑。

静态配置目录为：

```text
baselines/wan22_lora/
├── baseline.json
└── task_builder/
    └── data_adapter/
        └── profiles/
            ├── generic.json
            └── physics.json
```

WAN TaskBuilder fingerprint 覆盖：

- TaskBuilder type；
- 完整 DataAdapter fingerprint；
- Trainer 配置；
- Predictor 配置；
- 模型配置。

runtime GPU 资源仍由 Baseline bundle snapshot 冻结，但不进入 TaskBuilder 编译器
fingerprint；完整 Baseline digest 会捕获其变化。

WAN 当前通过兼容桥复用已验证的 DiffSynth 训练/推理实现。桥接过程中生成的
`frozen_cases.jsonl` 与 `resolved_prompts.jsonl` 是插件私有投影；运行器和评测器只
依赖 Dataset snapshot、任务实例和标准 prediction。

## 10. 实际产物

`atomic-run` 先构建实例，再创建 run 目录并冻结：

```text
<run_dir>/
├── frozen/
│   ├── dataset.json
│   ├── cases.jsonl
│   ├── views.json
│   ├── task.json
│   └── baseline.json
├── plan.json
├── task_builder.json
├── data_adapter.json
├── component_fingerprints.json
├── task_instance/
│   ├── manifest.json
│   ├── canonical_plan.json
│   ├── adaptations.jsonl
│   ├── training.json
│   ├── inference_jobs.jsonl
│   ├── execution_graph.json
│   ├── cache_bindings.json
│   └── baseline_payload.json
├── training/
├── jobs/
├── predictions.jsonl
└── evaluation/
```

`manifest.json` 是完整、带 digest 的权威实例；其余 `task_instance/` 文件是便于人工
审阅和工具消费的确定性投影。

## 11. 使用方式

只编译和导出实例：

```bash
PYTHONPATH=src python3 -m physbench task-build \
  --dataset datasets/physics_v1/dataset.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output /tmp/wan22_task_instance.json
```

构建、冻结、执行和评测：

```bash
PYTHONPATH=src python3 -m physbench atomic-run \
  --dataset datasets/physics_v1/dataset.json \
  --task tasks/official/finetune_eval_physics.json \
  --baseline baselines/wan22_lora/baseline.json \
  --output-root runs_v2
```

`atomic-run` 默认 dry-run；加入 `--execute` 才允许训练和推理。建议工作流是：

1. 用 `task-build` 审核 selection、adaptation、native input、DAG 和 digest；
2. 用默认 dry-run 的 `atomic-run` 审核 run 级投影；
3. 最后用 `--execute` 启动真实计算。

## 12. 新 Baseline 的接入准则

新增 Baseline 时应：

1. 在 `baseline.json` 顶层 `components` 中声明
   `task_builder`、`trainer`、`predictor`；
2. 把 DataAdapter 放入 TaskBuilder，而不是重新增加顶层
   `data_adapter`/`condition_adapter`；
3. 实现确定、无副作用的 `TaskBuilder.compile()`；
4. 保留 canonical job 的 case、partition 和 seed；
5. 将模型专有输入放入 opaque `native_inputs`；
6. 将模型专有任务字段放入 `baseline_payload`；
7. 通过 DAG 和符号引用连接操作，不在构建阶段创建 checkpoint；
8. 实现只消费并验证 `BaselineTaskInstance` 的 `run_task()`；
9. 测试同输入编译 digest 稳定、实例篡改会失败、两类 family 的 DAG 正确；
10. 测试 generic 不可读取结构化物理标注，physics 只经声明的 Baseline 私有通道注入。

这样 Dataset、Task 与 Baseline 可以分别演进：新增数据不要求修改模型代码，新增 Task
选择不要求修改 Dataset，新增 Baseline 也不要求污染核心任务或数据 schema。
