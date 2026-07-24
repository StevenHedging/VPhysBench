# 提示词条件设计

> 本文记录 v1 PromptProfile 机制。v2 已将通用文本和物理注入作为 Baseline
> DataAdapter 的内部阶段；只有 WAN 的 physics 阶段仍选择文本注入，其他 Baseline
> 不需要使用提示词。参见
> [统一 Data Adapter 架构](DATA_ADAPTER_ARCHITECTURE.md)。

## 设计原则

视频、物理真值与提示词是三个独立层：

```text
Case（资产 + physical_parameters）
                  │
                  ├── PromptProfile: generic
                  └── PromptProfile: physics_natural
                                  │
                                  ▼
                     Baseline model_input.prompt
```

case manifest 不再决定一次实验实际使用的提示词。Task 只选择 profile，runner 再根据
case 的 `scene_id` 和 `physical_parameters` 确定性地生成最终文本。因此更换提示词
不会复制视频、修改划分或创建另一份数据集。

## 当前仅有的两类 PromptProfile

| ID | 内容 | 用途 |
|---|---|---|
| `generic` | 场景、初始事件、固定相机和自然运动；不含数值物理参数 | 检查首帧与通用文本本身的能力 |
| `physics_natural` | 与 generic 相同的基本语义，再用自然语言写入标注物理量 | 检查显式物理条件是否改善生成 |

两类 profile 均覆盖单摆、自由落体、一维对心碰撞、匀减速滑动和匀速圆周运动。
当前真实数据中的前三类已通过完整解析校验，后两类会在数据导入时按 scene 契约检查
参数与单位。

## 训练与评测

视图 A 的默认设置为：

```json
{
  "prompts": {
    "train_profile": "physics_natural",
    "eval_profiles": ["generic", "physics_natural"]
  }
}
```

这表示只使用 `physics_natural` 训练一个 Adapter。训练结束后，同一个 Adapter 对每个
评测 case 运行两次，二者共享首帧、seed、生成帧数、分辨率、scheduler 与其余推理
超参，只替换提示词。它不是两次微调。

视图 B 不训练，只把同一冻结模型按两类 profile 分别评测。命令行可用重复参数选择
profile：

```bash
--eval-prompt-profile generic \
--eval-prompt-profile physics_natural
```

## 可复现产物

每次 run 包含：

- `frozen_prompt_profiles.json`：实际模板、模板 SHA-256 和 profile 选择；
- `resolved_prompts.jsonl`：每个 case 的最终提示词、引用参数和文本 SHA-256；
- `jobs/*.json`：含 `prompt_profile_id` 和真正传给 baseline 的 prompt；
- `predictions/<prompt_profile_id>/`：隔离两类视频，防止覆盖；
- `summary.json` / `report.md`：按 scene、partition、profile 分组，并提供总体
  `prompt_breakdown`。

历史 case 中重复保存的 `text.prompt`、`t2v.prompt`、`i2v.prompt` 和
`ti2v.prompt` 暂不批量删除，以免破坏旧工具；新 runner 不再把它们视为权威条件。
