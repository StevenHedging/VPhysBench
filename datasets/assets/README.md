# Canonical V14 assets

Dataset 14.0.0 的大型资产由 `physbench dataset pull` 从
`datasets/huggingface.json` 锁定的公开 Hugging Face commit 直接下载。
此目录是只读 Dataset；模型转换、预测、日志和额外可视化必须写入
`run/<run_id>/` 或外部缓存。

每个 Case 的 canonical 结构为：

```text
assets/<scene_id>/<case_directory>/
├── caption.json
├── physics.json
└── canonical/
    ├── first_frame.png
    ├── reference.mp4
    ├── masks/
    │   ├── manifest.json
    │   ├── 01.npz
    │   └── 01.png
    └── reference_observation/
        ├── manifest.json
        ├── entities/
        └── visualization/
```

`masks/` 保存原始、稳定并经复核的首帧主体 mask。
`reference_observation/` 是从 GT 视频离线派生的版本化轨迹与 mask tube，
可以重新生成；evaluator 在评分时读取这些冻结观测，不对 GT 重新分割。

`datasets/releases/14.0.0/assets.lock.json` 是正式资产成员和哈希权威。
Case 索引中的 `assets.*` 路径均相对于 `datasets/`；Baseline 只能看到
Task 和 DataAdapter 明确授权的生成输入，不能读取 reference 或 evaluator 资产。
