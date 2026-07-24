# Assets

真实数据按 `assets/<scene_id>/<case_id>/` 组织：

```text
<case_id>/
├── reference.mov|mp4   # Benchmark 使用的真实时间参考视频
├── source_slowmo.mp4   # 仅自由落体：逐字节保存的原始 8 倍慢放源
└── first_frame.png     # 由 reference 派生的 I2V/TI2V 输入
```

文件哈希、来源压缩包成员、原始媒体探测值和时间尺度标注见
`../manifests/import_audit.jsonl`。baseline 所需的尺寸、FPS、帧数和时间尺度变换必须写入
各自的运行缓存，不得回写本目录。自由落体是明确的数据侧例外：`reference.mp4` 已通过
无重编码的两遍时间戳 remux 恢复真实时间，manifest 因而标注速度因子 `1.0`；原始慢放
字节仍完整保存在 `source_slowmo.mp4`。
