# Canonical Assets

当前资产按以下结构组织：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
├── physics.json             # 与Case绑定的结构化物理标注
├── source/                  # 可选，原始逐字节文件
└── canonical/
    ├── reference.mp4|mov    # Benchmark物理时间reference
    ├── first_frame.png      # 与reference frame 0一致
    └── masks/               # 逐主体manifest、PNG和NPZ
```

`physics.json`必须恰好包含`schema_version`、`case_id`、`scene_id`和`physics`。它由
10.0.0 Case的`assets.physics_annotation`引用并进入资产锁；Loader强制校验其身份及
`physics`与内联`case.physics`完全一致。媒体通常被Git忽略，Case根部的
`physics.json`是例外，它属于受版本控制的Dataset元数据。

Canonical规则：

- 文件路径相对`datasets/`；
- 目录名简要编码scene的主要结构化物理量，不编码背景、颜色或采集环境；
- `case_id`是稳定身份，目录名不是身份API；
- reference容器时间戳表达Case `temporal`声明的物理时间；
- first frame与reference解码帧0一致；
- canonical视频只做必要的事件窗口和空间裁剪，不为模型改FPS或抽帧；
- 所有当前引用文件由`releases/10.0.0/assets.lock.json`封印；
- 任何模型侧媒体转换不得回写本目录。

批量压缩来源只在`provenance/source_archives/<batch>/`保存一次；
`assets/source_archives`是冻结Release的兼容链接。来源与审核信息位于
`../provenance/`。
