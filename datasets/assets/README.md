# Canonical Assets

当前资产按以下结构组织：

```text
assets/<scene_id>/<descriptive_physical_case_directory>/
├── caption.json             # 唯一的当前文本描述
├── physics.json             # 唯一的当前符号化结构化物理标注
├── source/                  # 可选，原始逐字节文件
└── canonical/
    ├── reference.mp4|mov    # Benchmark物理时间reference
    ├── first_frame.png      # 与reference frame 0一致
    └── masks/               # 逐主体manifest、PNG和NPZ
```

`caption.json`保存Case/Scene身份、caption、语言和标注来源；`physics.json`保存
Case/Scene身份及`physics`。12.0.0的轻量Case索引通过`assets.caption`和
`assets.physics_annotation`引用二者，Loader读取后物化`case.text`与`case.physics`。
媒体通常被Git忽略，Case根部的两个JSON属于受版本控制的Dataset元数据。

Canonical规则：

- 文件路径相对`datasets/`；
- 目录名简要编码scene的主要结构化物理量，不编码背景、颜色或采集环境；
- `case_id`是稳定身份，目录名不是身份API；
- reference容器时间戳表达Case `temporal`声明的物理时间；
- first frame与reference解码帧0一致；
- canonical视频只做必要的事件窗口和空间裁剪，不为模型改FPS或抽帧；
- 当前Release不维护资产锁或文件哈希；
- 每个Case只能有一个`caption.json`和一个`physics.json`，禁止新增版本后缀副本；
- 任何模型侧媒体转换不得回写本目录。

批量压缩来源只在`provenance/source_archives/<batch>/`保存一次；
`assets/source_archives`是冻结Release的兼容链接。来源与审核信息位于
`../provenance/`。
