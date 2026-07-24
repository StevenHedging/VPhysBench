# 统一 DataAdapter 架构

## 1. 定位与边界

Benchmark 的领域对象仍然只有 Dataset、Task、Baseline。Dataset 提供不可变资产和
结构化事实，Task 负责模型无关的数据选择与实验语义，Baseline 独占把这些事实转换为
模型原生输入的知识。

核心框架面向 Baseline 的任务构建入口是 `TaskBuilder`。`DataAdapter` 是
TaskBuilder 内部的 case 级适配器，而不是 Baseline 顶层组件：

```text
DatasetSnapshot + TaskSpec
             │
             ▼
   Baseline-owned TaskBuilder
      │
      ├── Benchmark CanonicalTaskPlan
      │
      └── private DataAdapter
             ├── spatial
             ├── temporal
             ├── paradigm
             ├── text
             └── physics
                         │
                         ▼
             BaselineTaskInstance
               └── opaque native_inputs
```

这种封装保留了五阶段的独立可测试性，又避免运行器直接编排 DataAdapter、Trainer 和
Predictor。运行器只要求 Baseline 提供 TaskBuilder，并执行其封印后的任务实例。

`native_inputs` 对 Benchmark 是 opaque 的。核心框架只冻结、审计和传递它，不要求
其中必须存在 `prompt`、`image` 或某个统一的物理条件字段。

## 2. 五个内部阶段

| 阶段 | 职责 | 不负责 |
|---|---|---|
| spatial | 分辨率、宽高比、分桶、padding/crop 策略 | 修改 Dataset 原始资产 |
| temporal | 物理时间、FPS、帧数、采样窗口 | 编造不存在的时间段 |
| paradigm | 构造 T2V/I2V/V2V 等模型需要的输入资产 | 规定所有模型必须使用首帧 |
| text | 描述场景和物理过程，不写入详细物理参数 | 读取 `case.physics` |
| physics | 将结构化物理信息注入模型支持的原生通道 | 假设注入通道必然是文本 |

这些阶段是 DataAdapter 的内部组合单元，不是 Benchmark 顶层插件。TaskBuilder 为
train/eval role 调用 `adapt_case(case, conditioning, role=...)`，得到可审计的
adaptation record；再把其中的 `native_inputs` 固化到训练规格或推理 job。

媒体真正的 resize、重采样或首帧提取可以推迟到 Baseline runtime。Dataset 中的源视频
不因某个 Baseline 的输入规格而改变。

## 3. Generic 与 Physics

两种 Task 使用同一类 DataAdapter：

- `generic`：执行 spatial、temporal、paradigm、text；physics 阶段明确禁用；
- `physics`：先执行完全相同的四个阶段，再执行 Baseline 私有 physics 阶段。

text 阶段接收一个已移除 `case.physics` 的对象，形成运行时信息屏障；physics 阶段是
唯一允许读取结构化物理标注的阶段。两种第一类任务仍是两个独立 AtomicRun，因此训练
得到独立 Adapter。

TaskBuilder 会把 conditioning 和 DataAdapter 指纹同时写入任务实例。执行器不会在
实例构建后临时更换 prompt profile 或物理注入方式。

## 4. 物理注入不是统一 Prompt

DataAdapter 的 `adapt_case()` 返回 baseline-native 的 opaque `native_inputs`。
不同 Baseline 可以采用完全不同的注入方式：

```text
WAN2.2       physics -> natural language -> native_inputs.text.prompt
Token model  physics -> quantized tokens -> native_inputs.physics_tokens
ControlNet   physics -> control tensor   -> native_inputs.control
Multimodal   physics -> encoder context  -> native_inputs.cross_attention
Simulator    physics -> state vector     -> native_inputs.initial_state
```

这些键只属于对应 Baseline。Benchmark 不建立“所有模型通用的 physics prompt”或
“所有模型通用的 physics tensor”，从而避免最低公分母接口限制未来架构。

## 5. WAN2.2 实现

WAN 配置实际位于：

```text
baselines/wan22_lora/
├── baseline.json
└── task_builder/
    └── data_adapter/
        └── profiles/
            ├── generic.json
            └── physics.json
```

`baseline.json` 的 `components.task_builder.config.data_adapter` 保存五阶段配置。当前
WAN 物理策略是 `append_structured_values_to_text`：

- generic profile 生成无详细参数的物理过程描述；
- physics profile 在相同过程描述后追加带单位和固定精度的结构化物理量；
- paradigm 阶段提供首帧，首选 `assets.first_frame`，否则从参考视频 frame 0 提取；
- 最终 WAN 原生输入是首帧和文本。

最后一点只是 WAN 私有实现，不是 Benchmark 公共契约。

## 6. 指纹与共享缓存

每个 DataAdapter 记录：

- `fingerprint`：五个阶段、模板和策略的完整指纹；
- `stage_fingerprints`：每个阶段的独立指纹；
- `materialization_fingerprint`：spatial、temporal、paradigm 的组合指纹。

共享媒体缓存使用：

```text
cache/baselines/<baseline_id>/<materialization_fingerprint>/<dataset_digest>/
```

修改文本模板或物理注入方法会改变完整 DataAdapter 指纹以及 TaskBuilder/任务实例
摘要，但不会重新编码相同视频。修改 FPS、分辨率或输入范式会改变
`materialization_fingerprint`，从而建立新的媒体缓存。

缓存绑定由 TaskBuilder 写入 `BaselineTaskInstance.cache_bindings`；它是可重建派生物，
不是 Dataset 的一部分。

## 7. 在 BaselineTaskInstance 中的产物

DataAdapter 的结果先成为任务实例的一部分，而不是直接触发模型运行：

```text
BaselineTaskInstance
├── identity.data_adapter
│   ├── fingerprint
│   └── materialization_fingerprint
├── adaptations[]
│   ├── adaptation_id
│   ├── five-stage audit
│   └── native_inputs
├── training.adaptation_ids[]       # finetune_eval
├── inference.jobs[]
│   ├── adaptation_id
│   └── native_inputs
└── cache_bindings[]
```

同一 case 在 train 与 eval role 下拥有不同的 `adaptation_id`，使训练输入和评测输入
都可独立审计。WAN 当前使用：

```text
<case_id>::<role>::<conditioning>
```

## 8. Run 产物

AtomicRun 会把适配相关内容冻结为：

```text
task_builder.json                         # TaskBuilder 与内置 DataAdapter 描述
data_adapter.json                         # 五阶段配置、指纹与缓存策略
component_fingerprints.json               # TaskBuilder/instance/adapter 摘要
task_instance/
├── manifest.json                         # 完整封印实例
├── adaptations.jsonl                     # 五阶段计划、native_inputs 与参数审计
├── inference_jobs.jsonl                  # 模型原生推理 job
└── cache_bindings.json                   # 不可变派生缓存绑定
```

WAN 后端还会生成：

```text
adaptations/case_adaptations.jsonl
data_adapter_cache_binding.json
resolved_prompts.jsonl
```

这些是 WAN 执行与旧 DiffSynth 入口的私有投影，不是新的 Benchmark 公共组件。核心
评测器不依赖其中的 prompt 或 WAN 私有 case 结构。

DataAdapter 如何被编译进完整任务实例，见
[TaskBuilder 架构](TASK_BUILDER_ARCHITECTURE.md)。
