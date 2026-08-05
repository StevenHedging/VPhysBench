# Canonical Assets

当前6.0.0沿用的资产按
`assets/<scene_id>/<descriptive_physical_case_directory>/` 组织：

```text
<descriptive_physical_case_directory>/
├── source/                 # 可选，原始逐字节文件
└── canonical/
    ├── reference.mp4       # Benchmark 物理时间 reference
    └── first_frame.png     # 与 reference frame 0 一致
```

批量压缩来源只在 `provenance/source_archives/<batch>/` 保存一次；
`assets/source_archives` 是为冻结release保留的兼容链接。Case 使用
`assets.source_archive` 与 `provenance.source_locator.member` 定位原始成员。

Canonical 规则：

- 文件路径相对 `datasets/`；
- 当前目录名简要编码 scene 的主要结构化物理量，不编码背景、颜色或采集环境；
- `case_id`继续作为稳定身份，目录名与case ID的绑定记录在6.0.0
  `asset_directory_mapping.json`；
- 6.0.0与5.1.0引用同一冻结媒体，不重复占用媒体payload；
- reference 容器时间戳表达 case `temporal` 声明的物理时间；
- first frame 与 reference 解码帧 0 一致；
- 斜面 reference 从审核后的启动附近帧开始；
- 自由落体 source 慢放字节保留，canonical reference 已恢复物理时间；
- 当前引用文件被`releases/6.0.0/assets.lock.json`封印；
- 任何模型侧媒体转换不得回写本目录。

来源与审核信息位于 `../provenance/`，运行时只使用 release 冻结的路径。
