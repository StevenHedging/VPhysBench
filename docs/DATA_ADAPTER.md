# DataAdapter 与 Baseline 输入策略

## 1. 定位

DataAdapter 属于 Baseline。它把冻结 Case 转换成模型原生输入：

```text
conditionable Case + role + Baseline adapter recipe
→ native_inputs + input_contract + adaptation audit
```

Dataset 保存原始 prompt、媒体与结构化事实；Task 只决定训练/评测 case 和 seed；
adapter 决定模型是否使用物理信息以及如何表示。统一接口是：

```python
adapt_case(case, *, role)
```

`role` 为 `train` 或 `eval`，不再传 Task 层的物理开关。

## 2. Adapter 接收的 Case

Compiler 给所有 Baseline 相同的 `conditionable_case_v1`：

```text
case_id / scene_id
text.prompt
appearance / temporal / ood
允许作为生成输入的 assets
physics（全部正式标量与时序quantity）
```

它不会给 adapter：

- `reference_video`、`physics_reference_video`、`source_video`；
- provenance、source locator 或 alignment evidence；
- provenance中的派生、辅助或审计量；
- evaluator reference。

Adapter 本身不接收训练 target。Compiler 只在 sealed runtime source 的训练 case 中
另加 `supervised_targets` 供 trainer 使用；eval adapter 与 predictor 看不到该 target。

所有 Baseline 看到同一份可条件化 Case，保证未来的 text、token、trajectory、mask、
flow 等策略使用同一接口。某个 Baseline 是否消费 physics，由 manifest 与输出 contract
共同约束。

## 3. `input_policy`

Baseline schema 5.0 必须声明：

```json
{
  "input_policy": {
    "schema_version": "1.0",
    "case_view": "conditionable_case_v1",
    "text": {
      "source": "case.text.prompt",
      "usage": "required"
    },
    "physics": {
      "source": "case.physics",
      "usage": "ignored",
      "representations": []
    }
  }
}
```

文本源固定是非空 `case.text.prompt`。物理策略：

### `ignored`

- `representations` 必须为空；
- `used_parameters` 必须是 `{}`；
- `physics_channels` 必须为空；
- 标准 adapter 的 `physics_transform` 必须为 `{"type": "none"}`。

### `optional`

- manifest 必须列出至少一种 representation；
- 每条 Case 可不使用物理；
- 一旦使用，`used_parameters` 与 physics channel 必须同时、精确出现。

### `required`

- manifest 必须列出至少一种 representation；
- 每条 adaptation 至少使用一个正式参数；
- channel 中登记的参数集合必须与 `used_parameters` 完全一致。

该策略是 Baseline identity 的一部分。改变 usage、representation 或 transform，需要新的
Baseline ID 或明确的 Baseline 版本升级，不能通过 Task 或运行时 flag 临时切换。

## 4. 标准 adapter

`StandardDataAdapter` 支持：

```text
standard_t2v_v1
standard_i2v_v1
standard_v2v_v1
```

它按五个阶段生成可审计输入：

1. `spatial`：scene 对应的宽高或模型 shape token；
2. `temporal`：FPS、帧数与 `4n+1` 等模型约束；
3. `paradigm`：T2V/I2V/V2V 媒体角色；
4. `text`：读取 `case.text.prompt`；
5. `physics`：按固定 policy 忽略或转换正式物理量。

当前内置物理转换：

```json
{
  "type": "append_structured_text_v1",
  "template_set": "six_scene_physics_clauses_v2"
}
```

模板位于：

```text
src/physbench/baseline_plugins/resources/six_scene_physics_clauses_v2.json
```

Renderer按scene白名单读取含`value`的独立标量quantity，验证单位和稳定symbol，按声明
精度格式化，并追加到`case.text.prompt`后。时序quantity不会被自动求均值、峰值、插值或
序列化；要使用完整时序必须由Python adapter声明相应representation。Renderer同时记录原prompt digest、最终prompt
digest、字段、原值、单位、symbol和渲染值。当前速度值是非负大小，运动方向来自原始
Case prompt。Driver只消费已经封印的`native_inputs.text.prompt`，不得再次拼接。

V1模板保留用于历史Bundle复现；当前active manifest全部指向V2资源，不能静默回退。

## 5. 自定义 Python adapter

非文本物理注入使用 Bundle-local adapter：

```json
{
  "adapter": {
    "kind": "python",
    "entrypoint": "adapter.py",
    "config": {},
    "cache_policy": "content_addressed_immutable"
  }
}
```

`adapter.py` 必须导出：

```python
def create_adapter(bundle):
    return MyDataAdapter(bundle)
```

返回对象实现 `DataAdapter`，至少提供：

- `adapt_case(case, *, role)`；
- `describe()`；
- `fingerprint`；
- `materialization_fingerprint`；
- 可选 `dependency_paths()`。

可声明的 representation 是开放字符串，例如：

```text
structured_text
numeric_tokens
trajectory
mask
optical_flow
force_field
state_sequence
proxy_video
control_video
```

大型表示必须放在不可变 artifact/cache 中，`native_inputs` 只保存
`artifact://sha256/<digest>` 或 `cache://sha256/<digest>`。Channel 同时登记
`content_sha256`、`producer_fingerprint` 和 `source_digest`；driver 解析实体时应复验
实际字节。

## 6. `input_contract`

每条 adaptation 必须输出：

```text
input_contract
├── schema_version
├── generation_mode
├── text
│   ├── required=true
│   └── binding
├── media_channels[]
├── physics_channels[]
└── asset_access[]
```

Contract 只描述模型输入如何绑定，不规定 `native_inputs` 内部形状。Compiler 会验证：

- 文本 binding 指向非空字符串；
- generation mode 与 Baseline capability 一致；
- media channel 与 asset whitelist 一致；
- physics representation 已声明；
- used parameter是正式quantity，且值、单位、symbol未被改写；
- 大型 control 使用 artifact reference；
- producer fingerprint 属于当前 adapter/materializer/TaskBuilder。

## 7. 媒体范式

| mode | 输入媒体 |
| --- | --- |
| `t2v` | 无媒体 |
| `i2v` | 一个或多个图像 channel |
| `v2v` | 一个或多个视频 channel |
| `hybrid` | 同时有图像和视频 |

标准 I2V 使用 Dataset 的 `assets.first_frame`。缺失首帧应回到 Dataset provenance 流程
补齐，不能从 reference 临时提取，也不能把 GT 首帧拼到生成视频。

V2V 标准 channel ID 为 `conditioning_video`。这里的 conditioning 仅表示媒体在
V2V 模型中的输入角色，不是“是否注入结构化物理信息”的 Task 属性。视频必须来自显式
独立资产（默认 `assets.input_video`）或经审计的 derived artifact；禁止绑定
reference、physics reference 或 source video，也会检查内容 digest 别名。

Dataset 14.0.0 的 916 个 Case 当前都没有 `assets.input_video`，也没有独立裁剪并授权的
条件前缀视频。因此 managed V2V 只是预留接口，不能用于当前官方 Task；DataAdapter 不会
从 reference/source/physics-reference video 临时裁剪或推导该输入。

## 8. Fingerprint 与 cache

完整 adapter fingerprint 覆盖：

```text
spatial + temporal + paradigm + text + physics
```

Materialization fingerprint 只覆盖：

```text
spatial + temporal + paradigm
```

因此同模型的 generic/physics Baseline：

- canonical plan 相同；
- first-frame 与 generation shape 相同；
- media materialization fingerprint 可相同；
- 完整 adapter/TaskBuilder/TaskInstance fingerprint 不同。

文本或物理模板变化不会无意义地重建媒体 cache，但一定会改变完整输入身份。

Cache key 至少包含 source asset digest、materialization implementation、空间/时间配置
与输入范式。Cache 是 immutable；相同 key 出现不同字节时必须报错，不能覆盖。

## 9. 统一 I2V 媒体契约

所有 managed I2V Baseline 只声明每个 scene 的 `width`、`height`，以及 `fps` 和
固定/可变帧数。`StandardDataAdapter` 自动生成一份
`native_inputs.media_contract`；Baseline 不再分别声明 contain、margin fill 或 evaluator
裁剪策略。

公共 I2V Driver 使用同一个实现完成以下操作：

1. 将 Dataset 首帧等比、居中 contain 到模型 canvas；
2. 使用边缘像素延展临时 margin；
3. 在模型调用前验证 conditioning image 已精确等于 canvas，保证模型内部 resize 为
   same-size no-op；
4. 生成后验证 prediction 的 canvas、起始时间、FPS 和帧数规则；
5. 将不合格输出标为 `protocol_error`，不交给 scene evaluator。

Prediction record 由 managed runtime 绑定编译时封印的同一份 `media_contract`，Driver
不能覆盖。评估器只移除可由 centered contain 唯一确定的 prediction margin，保留 GT
完整物理视野；不允许事后 letterbox、非等比拉伸、crop-to-fill 或内容配准。

新增普通 I2V 模型时，优先使用 scaffold 生成的 `StandardI2VCLIDriver`。模型脚本只需
接收公共 Driver 生成的 `--image`、按 job spec 的 canvas/FPS/帧数输出视频；不需要为
该模型增加 evaluator 分支。

## 10. 审计输出

每条 adaptation 至少记录：

```text
adaptation_id
case_id / scene_id / role
source_prompt_sha256 / prompt_sha256
text_transform_id
used_parameters
spatial / temporal / paradigm / text / physics stages
input_contract
native_inputs
adapter fingerprint
materialization fingerprint
```

这些记录进入 `task_instance/adaptations.jsonl`，AtomicRun 同时冻结
`data_adapter.json`、`task_builder.json` 和 component fingerprints，使 Dataset、
Task、adapter 与模型执行差异可以分开审计。
