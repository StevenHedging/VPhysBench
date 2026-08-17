# Dataset 文档入口

Dataset 构成、原子 Case 定义、`physics.json`、`caption.json`、mask 规则，以及新 Scene
从原始视频和 XLSX 接入的完整流程，统一维护在
[数据集说明与使用手册](../datasets/README.md)。

本文件不再重复数据规范，避免出现多个互相冲突的入口。

## Direct asset delivery

Git 发行版只保存 Dataset 14.0.0 元数据和 40 位不可变 Hugging Face commit。
`physbench dataset pull` 从该 commit 显式下载 `assets/**`，直接恢复到
本地 `datasets/assets/`。中断后重跑会复用已完成文件；下载后会检查
Dataset 声明的资产是否就绪。
