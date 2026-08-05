# 平抛运动参数表整理报告

生成时间：2026-07-29T07:41:16.595345+00:00

## 整理结论

- 现有平抛case：128条；
- Excel成功映射：97条，其中训练
  70条、测试27条；
- 没有参数表行：31条，其中训练
  26条、测试5条；
- Excel映射后重复：0；Excel中不存在的case：0；
- 原始Excel、G30和six-scene-v1均未修改；
- 新入口：`manifests/six_scene_v2_cases.jsonl`。

## 来源工作簿

- `2.平抛运动实验记录表.xlsx`：20条，SHA256 `425c777858dcdcda30c9065ea80bde85cee9f478690666008057f9c04775c2d0`
- `3.平抛运动实验记录表.xlsx`：47条，SHA256 `1ac90477cadca0e1b8bdefd7c1773a8e9c1d4a9bf5b6325bbef6829fcd1d9e3b`
- `4.平抛运动实验记录表.xlsx`：30条，SHA256 `1b3ace94af5cd391e1bd873178f712bd4834755de43f3b57c8581830c343fb21`

## 速度归一化

工作簿时间列标题是`block_time_ms`，但数据为约0.01--0.04。
按毫秒使用会得到659--1520 m/s。结合球直径、缓存公式和实验量级，本次把原值解释为
秒，并重新计算：

`v = ball_diameter_m / block_time_s`

归一化速度共97条，范围
0.659432--1.520205 m/s，均值
1.021238 m/s。源值、缓存值和除数1000均保存在导入审计中。

## 未覆盖case

`["parabolic_img_0543", "parabolic_img_0544", "parabolic_img_0545", "parabolic_img_0546", "parabolic_img_0547", "parabolic_img_0548", "parabolic_img_0549", "parabolic_img_0550", "parabolic_img_0551", "parabolic_img_0552", "parabolic_img_0553", "parabolic_img_0554", "parabolic_img_0555", "parabolic_img_0556", "parabolic_img_0557", "parabolic_img_0558", "parabolic_img_0560", "parabolic_img_0561", "parabolic_img_0562", "parabolic_img_0563", "parabolic_img_0564", "parabolic_img_0565", "parabolic_img_0566", "parabolic_img_0567", "parabolic_img_0568", "parabolic_img_0569", "parabolic_img_0570", "parabolic_img_0659", "parabolic_img_0660", "parabolic_img_0661", "parabolic_img_0662"]`

这些case保留在v2总catalog中，但标记为
`missing_uploaded_workbook_row`，不得用于需要初速度的参数因果结论。

## 已确认的源表勘误

- 用户确认：3号表与4号表使用的都是“大钢球”；3号表“特大钢球”为标注错误。v2已将3号表47条统一规范为“大钢球”，同时保留源值和修正审计。
- 用户确认：3号表T024的视频编号应为IMG_0590，而非源表中的IMG_0560。v2映射到parabolic_img_0590，同时保留源值和修正审计。

## 源数据异常

- 时间列标题为毫秒，但0.01--0.04的源数值必须按秒解释；Excel缓存速度因此大1000倍，v2已重算且保留全部原值。
- 23.7mm和4.16g对应源表密度约0.597g/cm^3，与钢材不符；质量按源值保留但标记unverified，不用于真空平抛轨迹公式。
- 光电门位于出口前1.2cm；测得速度是近发射速度proxy，并非出口瞬时速度的无误差真值。
- 仍有31条case没有上传表格行，保持显式missing，未按相邻编号推断参数。

## 输出

- `source_docs/parabolic_motion_parameters_imported_v2.csv`
- `manifests/parabolic_parameter_import_v2.jsonl`
- `manifests/six_scene_v2_cases.jsonl`
- `manifests/six_scene_v2_all.csv`
- `manifests/six_scene_v2_train.csv`
- `manifests/six_scene_v2_test.csv`
- `manifests/parabolic_parameterized_v2_train.csv`
- `manifests/parabolic_parameterized_v2_test.csv`
- `manifests/parabolic_parameterized_v2_test_inputs.jsonl`
- `manifests/six_scene_v2_audit.json`

`parabolic_parameterized_v2_test_inputs.jsonl`不含参考视频路径、测试视频帧数或时长；
它只登记首帧、独立物理参数和prompt。正式推理仍必须另行预注册一个全局固定、
非case特定的预测帧数。

## 最终验证

- v1 `six_scene_cases.jsonl` SHA256仍为
  `8213dd59765f7ff22cbc83d812660b91688dc14ad11f7f7a869a9e1b2f090312`；
- v2共397个唯一case，train 298、test
  99，train/test overlap为0；
- 参数完整子集为train 70、test
  27，交集为0；
- 安全测试输入未包含参考视频、时长、参考帧数、case特定`num_frames`或事件终点；
- 参数导入没有打开测试参考视频，也没有修改G30、v1或冻结划分。
