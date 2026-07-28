# DataAdapter 与条件隔离

## 1. 定位

DataAdapter recipe 属于 Baseline，不属于 Dataset。它将冻结 case 转换为模型原生
输入，同时使媒体派生与 generic/physics 条件使用可审计。

schema v4 managed/submission Bundle 通过 loader 注入 `DataAdapter`：普通 Baseline
使用内置 `StandardDataAdapter`，结构化控制 Baseline 可提供 Bundle-local Python
adapter。schema v3 command Bundle 继续通过 `CommandDataAdapterProxy` 调用模型族
实现。Benchmark 不把模型专有 runner payload 写入 Dataset。

```text
frozen case + frozen job + adapter config
→ native_inputs + adaptation audit + immutable cache artifacts
```

## 2. 五阶段模型

### 空间适配

- 选择 scene 对应宽高 bucket；
- 保持宽高比；
- resize 后 pad；
- 保存 source/target 尺寸、scale、offset 和插值方式；
- 不覆盖 Dataset 首帧或 reference。

### 时间适配

- 声明模型 FPS、目标帧数和 `4n+1` 等合法长度；
- 保存生成覆盖区间与变换 recipe；
- 只处理已授权的条件媒体，不读取 evaluator reference；
- 不把 GT 帧注入生成结果。

reference 解码、公共 timeline 和 reference-bounded sampling 属于 Evaluator，不属于
DataAdapter。Adapter 只需保证 prediction 的最后时间戳覆盖任务要求的物理区间。

### 输入范式与审计契约

- T2V：文本；
- I2V：文本与首帧；
- V2V：文本与独立条件视频；
- hybrid：文本，同时包含图像和视频。

所有模式都必须输出 `input_contract`，声明 generation mode、非空文本 binding、媒体
channel、physics channel 和资产访问白名单。I2V 必须使用 `assets.first_frame`；不再
从 reference 第 0 帧补首帧。V2V 必须使用显式条件资产（推荐
`assets.input_video`），禁止使用 `reference_video`、`physics_reference_video` 或
`source_video`。

### 文本适配

文本 profile 由 Baseline 选择并进入完整依赖指纹。当前 WAN 与 Cosmos 为保证输入对照
公平，共用 `five_scene_i2v_v1`；模型专属 profile 也可以放在自己的 Bundle 内。profile
负责 scene 描述与物理白名单，不改变 Dataset 事实。

### 物理注入

`generic|physics` 是信息访问策略，不是注入方法。公共 compiler 不向 generic adapter
提供结构化 `case.physics`；仅 physics conditioning 可读取结构化标注。注入必须：

- 只使用 profile 白名单字段；
- 保留数值与单位；
- 记录字段路径和渲染结果；
- 写入 Baseline 声明的 native target；
- 实际使用至少一个 `annotated=true` 参数；
- 不修改公共 job 或 Dataset。

representation 是开放字符串，例如 `structured_text`、`numeric_tokens`、
`trajectory`、`mask`、`optical_flow`、`force_field` 或 `proxy_video`。大型控制内容
必须用 `artifact://sha256/<digest>` / `cache://sha256/<digest>` URI 引用，
TaskInstance 只冻结 binding 与 provenance。核心验证声明格式和 active producer；
custom driver 负责解析实体、复验实际字节 SHA-256，并验证其自报 `source_digest`。

## 3. Generic 隔离

generic 分支的公共契约不是“最后 prompt 没出现数字”，而是 compiler 不提供结构化
physics 或 GT/reference/provenance。

测试要求：

1. generic adapter 输入没有 `physics` key；
2. physics adapter 只看到 `annotated=true` 的字段；
3. 内置共享媒体 recipe 在 generic/physics 两臂使用相同 materialization key；
4. 每条 adaptation 的字段、数值、单位、channel 和 binding 都可审计。

Python adapter/driver 是受信任 Bundle 代码。case ID 和资产路径可能带有语义，因此当前
不承诺对恶意扩展的严格 non-interference；第三方盲测需要另加 opaque handle、无语义
资产别名和进程隔离。

## 4. 内容寻址 cache

Cache key 至少包含：

```text
source asset SHA-256
+ materialization implementation digest
+ spatial config
+ temporal config
+ input paradigm config
```

文本差异不能使媒体 materialization 失效。缓存目录是 immutable；同 key 内容不一致
必须报错，不能覆盖。

完整 DataAdapter fingerprint 覆盖五个阶段；materialization fingerprint 只覆盖空间、
时间和输入范式阶段。schema v4 Bundle-local Python 文件自动进入 portable digest；
其他 Bundle-local 文件用 `fingerprint_paths`，Bundle 外共享实现/profile 用
`dependency_paths()`。因此文本变化仍会使
TaskBuilder/TaskInstance 身份变化，但不会无意义地重建媒体 cache。

## 5. WAN2.2 当前配置

`baselines/wan22_lora/baseline.json` 和 managed G15 recipe 使用：

| scene | bucket |
| --- | --- |
| pendulum | 480 × 832 |
| free_fall | 480 × 832 |
| uniform_circular_motion | 480 × 832 |
| collision_1d | 832 × 480 |
| inclined_plane_slide | 832 × 480 |

时间规格：

- 24 FPS；
- 最多 121 帧；
- 至少 5 帧；
- 保留合法 `4n+1` 帧数；
- 使用物理时间前缀。

输入范式为 I2V。physics profile 把结构化值追加到
`native_inputs.text.prompt`；generic profile 明确禁止详细物理字段。

## 6. Cosmos3-Nano 当前配置

Cosmos 同样使用 Dataset `assets.first_frame`，但不先生成 WAN 宽高像素副本，而是把
源资产与 Cosmos-native shape token 交给其预处理器：

| scene | resolution | aspect ratio |
| --- | ---: | --- |
| pendulum | 480p | `9,16` |
| free_fall | 480p | `9,16` |
| collision_1d | 480p | `16,9` |
| inclined_plane_slide | 480p | `16,9` |
| uniform_circular_motion | 480p | `4,3` |

时间规格固定为 24 FPS、121 帧，满足 `4n+1`。这不要求 GT 与生成视频同分辨率或同
帧数；统一 timeline、letterbox 和 reference-bounded sampling 属于 evaluator。
Cosmos generic/physics 仍使用同一 first-frame 资产和 generation shape，仅 prompt
stage 不同。

## 7. 审计输出

每个 adaptation 至少记录：

```text
adaptation_id
case_id
conditioning
implementation type/version
source asset binding
spatial transform
temporal transform
text profile
physics fields used
prompt digest / used parameters
adapter and materialization fingerprints
```

TaskInstance 保存审计摘要和完整 artifact 路径，AtomicRun 再冻结实例指纹。这样可以
区分 Dataset 差异、Task 差异、Adapter 差异和模型推理差异。
