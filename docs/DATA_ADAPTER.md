# DataAdapter 与条件隔离

## 1. 定位

DataAdapter 属于 Baseline，不属于 Dataset 或 Benchmark 核心。它将冻结 case 转换为
模型原生输入，同时保证媒体派生可重现、generic/physics 条件隔离可验证。

在 Bundle v3 中，核心持有 `CommandDataAdapterProxy`；实际实现位于 Baseline 目录，
通过 `physbench-baseline-v1/adapt_case` 返回 adaptation record。核心只验证接口、
fingerprint 和隔离不变量，不解释模型原生 payload。

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

- 按物理时间读取视频；
- 处理模型 FPS 和 `4n+1` 等帧数约束；
- 保存 source indices、时间戳和截取区间；
- 不把 GT 帧注入生成结果。

### 输入范式

- T2V：文本；
- I2V：文本与首帧；
- TI2V：文本、首帧及模型原生附加条件。

首帧优先使用 `assets.first_frame`；缺失时只能从 canonical reference 第 0 帧确定性提取。

### 文本适配

文本 profile 位于 Baseline bundle 内。它负责 scene 描述、模型风格和负面条件，不改变
Dataset 事实。

### 物理注入

仅 physics conditioning 可读取结构化 `case.physics`。注入必须：

- 只使用 profile 白名单字段；
- 保留数值与单位；
- 记录字段路径和渲染结果；
- 写入 Baseline 声明的 native target；
- 不修改公共 job 或 Dataset。

## 3. Generic 隔离

generic 分支的安全边界不是“最后 prompt 没出现数字”，而是适配阶段不能访问 physics。

测试要求：

1. 给 generic adapter 传入可监控 case 视图；
2. 若访问 `physics` 立即失败；
3. generic 与 physics 使用相同媒体 cache key；
4. 两者只在条件 artifact/fingerprint 上不同。

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
时间和输入范式阶段。Bundle portable digest 另行覆盖 endpoint、完整实现和 profiles，
因此文本变化仍会使 TaskBuilder/TaskInstance 身份变化，但不会无意义地重建媒体 cache。

## 5. WAN2.2 当前配置

`baselines/wan22_lora/baseline.json` 使用：

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

## 6. 审计输出

每个 adaptation 至少记录：

```text
adaptation_id
case_id
job_id
conditioning
implementation type/version
source asset binding
spatial transform
temporal transform
text profile
physics fields used
native output digest
cache keys
```

TaskInstance 保存审计摘要和完整 artifact 路径，AtomicRun 再冻结实例指纹。这样可以
区分 Dataset 差异、Task 差异、Adapter 差异和模型推理差异。
