# Splits

由 `physbench split` 生成的视图 A/B 冻结划分。不要手工覆盖已用于正式报告的 split。

- `view_a.json`：采用 case 的官方 train / test_id / test_ood1 标注。
- `view_b_seed42_g5.json`：各 scene 内使用 seed 42 确定性打乱并尽量等分为 5 组。
