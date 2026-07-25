# Canonical Assets

资产按 `assets/<scene_id>/<case_id>/` 组织：

```text
<case_id>/
├── source/                 # 可选，原始逐字节文件
└── canonical/
    ├── reference.mp4       # Benchmark 物理时间 reference
    └── first_frame.png     # 与 reference frame 0 一致
```

批量压缩来源只在 `assets/source_archives/<batch>/` 保存一次。Case 使用
`assets.source_archive` 与 `provenance.source_locator.member` 定位原始成员。

Canonical 规则：

- 文件路径相对 `datasets/physics_video/`；
- reference 容器时间戳表达 case `temporal` 声明的物理时间；
- first frame 与 reference 解码帧 0 一致；
- 斜面 reference 从审核后的启动附近帧开始；
- 自由落体 source 慢放字节保留，canonical reference 已恢复物理时间；
- 文件被 `releases/3.0.0/assets.lock.json` 封印；
- 任何模型侧媒体转换不得回写本目录。

来源与审核信息位于 `../provenance/`，运行时只使用 release 冻结的路径。
