# Dataset 文档入口

Dataset 构成、原子 Case 定义、`physics.json`、`caption.json`、mask 规则，以及新 Scene
从原始视频和 XLSX 接入的完整流程，统一维护在
[数据集说明与使用手册](../datasets/README.md)。

本文件不再重复数据规范，避免出现多个互相冲突的入口。

## Distribution v1

Git 发行版只保存 Dataset 13.0.0 元数据和 40 位不可变 Hugging Face commit。
`physbench dataset pull` 从该 commit 显式下载
`distribution/v1/manifest.json` 及其声明的分片，不扫描或逐个解析远端叶文件。
Manifest 固定 Dataset identity、release、digest、文件清单，以及每个分片和文件的
大小与 SHA-256。

下载缓存和解压 staging 位于 `datasets/.vphysbench/`，按 revision 隔离并被 Git
忽略。中断后重跑会复用已验证分片。绝对路径、父目录穿越、符号链接、重复或未声明
成员、大小/哈希不符都会 fail closed；完整 staged Dataset 与 digest 验证通过后，
下载器才发布 `datasets/assets/`。
