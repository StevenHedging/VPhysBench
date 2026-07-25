# 数据契约

当前最新 release 是 `datasets/physics_video/releases/3.0.0/`，使用
`schema_version=2.0` 的 prompt-free Dataset 契约。“v2”表示数据契约版本，不等于
release 号；2.0.0 作为原三场景不可变快照保留。下方 Case manifest 小节描述的是
仍被兼容的 v1 清单。

## Case manifest

主清单使用 JSON Lines：每行一个独立 case，字段由 `schemas/case.schema.json` 定义。

关键字段：

- `case_id`、`scene_id`：稳定且全局唯一；不要把随机种子写入 case ID。
- `view_a_split`：`train`、`test_id` 或 `test_ood1`。
- `physical_parameters`：参数名映射到 `{value, unit, annotated}`，保留单位。
- `temporal`：可选的 reference 时间标注。`encoded_to_physical_speed=8` 表示该 reference 仍是 8 倍慢放，baseline 必须在自己的适配缓存中恢复物理时间；`1.0` 表示 reference 已是真实时间。当前官方自由落体 reference 已在数据侧恢复，因此为 `1.0`，原慢放源由 `assets.source_video` 单独保存。
- `ood.level` 与 `ood.factors`：ID 为 `id + []`；OOD1 必须至少一个 appearance factor。
- `provenance.parent_case_id`：合成 OOD1 的父 ID case。
- `assets.reference_video`：这个外观条件下真实拍摄的完整视频。
- `assets.physics_reference_video`：用于运动/轨迹 GT；OOD1 可指向父 case 的真实视频。
- `has_real_reference_video`：严格表示是否存在与当前外观一致的真实延续视频。
- `input_views`：baseline 可直接消费的 T2V/I2V/TI2V 媒体视图。I2V/TI2V
  只要求首帧，不要求在 case 内保存 prompt。

## 提示词与视频解耦

物理参数属于 case 真值，提示词属于 run 级实验条件。新运行不会把 case 内的
`text.prompt` 或 `input_views.*.prompt` 作为权威输入，而是根据
`physical_parameters` 和冻结的 PromptProfile 生成文本。旧字段暂时保留用于兼容历史
manifest，后续导入数据可以不再重复写入它们。

目前只注册两类提示词：

- `generic`：仅描述实验场景、初始事件和相机约束，不注入数值物理参数；
- `physics_natural`：与 `generic` 共享场景语义，并用自然语言注入该 case 已标注的物理参数。

每次运行都会保存 `frozen_prompt_profiles.json`（模板、配置哈希和选择）以及
`resolved_prompts.jsonl`（每个 case 的最终文本、所用参数和文本哈希），避免修改
PromptProfile 后无法复现实验。

## OOD1 合成约束

合成 OOD1 只创建首帧，必须：

1. 记录生成工具、版本/模型、prompt、seed 和父 case；
2. 与父 case 的物理参数完全一致；
3. `has_real_reference_video=false`，除非之后确实补拍同条件视频；
4. 不把合成图片或父 case 视频标成当前外观的真实参考视频；
5. appearance factor 必须来自 scene 配置允许集合。

## 导入建议

权威资产统一存放在 `datasets/physics_video/assets/`。Dataset release 必须通过
`assets.lock.json` 冻结所有被引用资产的路径、大小和 SHA-256。运行前使用：

```bash
PYTHONPATH=src python3 -m physbench validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-assets

# 发布前或怀疑资产损坏时执行完整字节校验
PYTHONPATH=src python3 -m physbench validate-dataset \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --check-asset-hashes
```
