# DataAdapter 与条件隔离

## 1. 定位

DataAdapter 属于 Baseline，不属于 Dataset 或 Benchmark 核心。它将冻结 case 转换为
模型原生输入，同时保证媒体派生可重现、generic/physics 条件隔离可验证。

在 Bundle v3 中，核心持有 `CommandDataAdapterProxy`；实际实现位于 Bundle，或位于
同模型族共用且被显式指纹化的支持模块。它通过
`physbench-baseline-v1/adapt_case` 返回 adaptation record。核心只验证接口、
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

“可解码 reference 帧数”和“生成视频需要覆盖的时间戳数”必须分开：前者向下取合法
帧数以禁止伪造 source frame，后者向上取合法帧数以保证 prediction 的最后时间戳覆盖
评估区间。只比较容器 duration 会产生一帧偏差。

### 输入范式

- T2V：文本；
- I2V：文本与首帧；
- TI2V：文本、首帧及模型原生附加条件。

首帧优先使用 `assets.first_frame`；缺失时只能从 canonical reference 第 0 帧确定性提取。

### 文本适配

文本 profile 由 Baseline 选择并进入完整依赖指纹。当前 WAN 与 Cosmos 为保证输入对照
公平，共用 `five_scene_i2v_v1`；模型专属 profile 也可以放在自己的 Bundle 内。profile
负责 scene 描述与物理白名单，不改变 Dataset 事实。

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
时间和输入范式阶段。Bundle-local 文件进入 portable digest，共享实现/profile 进入
TaskBuilder runtime dependency fingerprint。因此文本变化仍会使
TaskBuilder/TaskInstance 身份变化，但不会无意义地重建媒体 cache。

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
