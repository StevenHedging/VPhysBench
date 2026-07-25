# Assets

真实数据按 `assets/<scene_id>/<case_id>/` 组织：

```text
<case_id>/
├── source/
│   ├── reference.mov          # 碰撞：逐字节保存的原始视频
│   ├── source_slowmo.mp4      # 自由落体：原始 8 倍慢放源
│   └── first_frame_source.png # 碰撞对齐前首帧证据
└── canonical/
    ├── reference.mp4          # Benchmark 使用的真实时间参考视频
    └── first_frame.png|jpg    # 由 reference 派生或审核通过的首帧
```

文件哈希、来源压缩包成员、原始媒体探测值和时间尺度标注见
`../provenance/imports/import_audit.jsonl`；release 使用的文件由
`../releases/<version>/assets.lock.json` 封印。baseline 所需的尺寸、FPS、帧数和时间尺度变换必须写入
各自的运行缓存，不得回写本目录。自由落体是明确的数据侧例外：`reference.mp4` 已通过
无重编码的两遍时间戳 remux 恢复真实时间，manifest 因而标注速度因子 `1.0`；原始慢放
字节仍完整保存在 `source/source_slowmo.mp4`。
