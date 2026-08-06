# VPhysBench 可移植性重命名设计

## 目标

将仓库工作目录从 `physics_video_benchmark/` 重命名为 `VPhysBench/`，并让社区发布所需的活动代码、配置、命令和文档不依赖当前服务器的绝对路径。重命名不得改写 Dataset 历史 provenance，不得重写已有 `run/` 生成物，也不得改变稳定的 Python 导入和命令行接口。

## 名称与兼容接口

- GitHub 仓库与展示名称统一使用 `VPhysBench`。
- Python distribution 名称从 `physics-video-benchmark` 调整为 `vphysbench`。
- Python 包名保持 `physbench`。
- CLI 命令保持 `physbench`。
- Git remote 保持 `StevenHedging/VPhysBench.git`，不执行 push。

目录名、发行名称属于项目品牌；`physbench` 包名和 CLI 属于已存在的程序接口。两者分开处理可完成品牌迁移，同时避免无必要的 API 破坏。

## 路径政策

### 活动发布面

下列活动发布面不得包含当前机器的 `/root/...`、`/mnt/...` 或旧仓库根 `physics_video_benchmark`：

- `README.md`
- `src/`
- `scripts/`
- `tasks/official/` 与 `tasks/smoke/`
- 当前正式 Task 使用的 evaluation protocol
- 面向用户的当前操作、评估、基线接入与场景开发文档
- tracked portable baseline manifests 与 `baseline.local.example.json`

仓库内文件使用相对于仓库根或其所属 manifest 的相对路径。文档命令默认从仓库根执行，输出示例使用 `run/`、`results/`、`evaluation_audits/` 等仓库相对位置。

### 外部部署依赖

模型权重、第三方框架、独立 Python 环境和 Hugging Face cache 不属于仓库。它们继续通过 Git 忽略的 `baseline.local.json` 注入，不进入 portable manifest 或 GitHub。tracked `baseline.local.example.json` 使用相对占位示例，不记录任何开发者机器路径。

现有本机 `baseline.local.json` 可保留其真实绝对部署路径，因为它们是未跟踪的机器本地状态；只要其中没有旧 Benchmark 根路径，仓库目录改名不要求修改它们。

### 历史与冻结材料

以下内容保留其原始路径字符串：

- `datasets/provenance/` 中的源媒体位置、导入暂存位置和审计记录；
- `docs/experiments/` 中用于复述既有实验环境的路径；
- 当前正式 Task 不再使用的冻结旧 evaluation protocol；
- 既有 `run/`、`results/` 与 cache 生成物。

这些字符串是历史事实、冻结身份或不可变运行证据，不是当前安装说明。机械替换会伪造 provenance，或改变旧协议和运行记录的摘要。

## 实施结构

### 1. 可移植性门禁

新增仓库级测试，枚举明确的活动发布面并拒绝：

- 当前服务器前缀 `/root/` 与 `/mnt/`；
- 旧目录 token `physics_video_benchmark`；
- 根 README 中旧项目标题或旧目录结构名称；
- `pyproject.toml` 中旧 distribution 名称。

测试使用显式 include 列表和历史 exclude 列表，避免把 provenance 原始事实误判为活动配置。测试还验证 Git root basename 为 `VPhysBench`，因此必须在改名前先出现预期失败。

### 2. 品牌与活动文档迁移

将根 README 标题、目录树、安装命令和面向用户的当前文档改为 `VPhysBench` 及相对命令。删除不必要的 `cd /absolute/path`；确需说明进入仓库时使用 `cd VPhysBench` 或 `<repo-root>`，不绑定用户名和挂载点。

将活动审计输出示例改到仓库相对目录。历史实验文档及旧协议路径不参与替换。

### 3. Baseline 示例可移植性

portable manifests 继续使用已有相对缓存路径。所有 tracked `baseline.local.example.json` 把 `/absolute/path/to/...` 改为相对的 `../external/...` 示例，并在基线接入文档中说明这些值由使用者按本机部署填写，文件不会进入 Git。

本次不新增通用环境变量插值器：运行时目前把本地部署值作为文件系统路径直接使用，引入一套新的插值语法会扩大接口和测试范围，且不是仓库改名所必需。

### 4. 物理目录重命名

在活动文本适配完成并通过除 basename 外的门禁后，将 `/root/Steven/physics_video_benchmark` 原子重命名为 `/root/Steven/VPhysBench`。`.git/`、Dataset、`run/`、cache 和 ignored local overrides 随目录整体移动，不复制大文件。

重命名后所有命令从新路径执行。Git remote、branch、status 和 ignore 行为必须保持不变；旧目录必须不存在。

## 错误处理与安全

- 重命名前确认目标 `/root/Steven/VPhysBench` 不存在，避免覆盖已有目录。
- 不删除或覆盖任何 Dataset、`run/`、cache、模型权重和 local override。
- 不追踪 ignored `run/`、cache 或 `baseline.local.json`。
- 不推送 GitHub，不修改 remote URL。
- 若目录重命名失败，保持旧目录可用并停止后续命令。
- 若 scoped path gate 失败，报告具体文件，不通过扩大历史 exclude 来掩盖活动路径问题。

## 验证

重命名完成后验证：

1. `git rev-parse --show-toplevel` 返回 `/root/Steven/VPhysBench`。
2. `/root/Steven/physics_video_benchmark` 不存在。
3. 活动发布面路径门禁通过，历史保留区仍保持原始 provenance。
4. Git remote 仍指向 `StevenHedging/VPhysBench.git`。
5. Dataset v12 validator 通过且 `media_changes=0`。
6. baseline discovery 仍列出全部 portable baseline identity。
7. run-root、CLI、artifact、runner 与 integrated-baseline 专项测试通过，仅允许未配置真实模型的显式 skip。
8. `git diff --check` 通过，Git status 不包含任何生成物或 local override。

全量测试仍需运行并如实报告；仓库已知的旧 evaluator、缺失资产和过期协议测试失败不应被误报为本次重命名回归，也不在本设计中顺带修复。

## 完成条件

- 项目物理目录与公开品牌均为 `VPhysBench`。
- 活动发布面不含机器绝对路径或旧目录名。
- 历史 provenance 与冻结材料保持审计真实性。
- 稳定接口仍为 `import physbench` 和 `physbench` CLI。
- Dataset、baseline discovery 和迁移专项验证结果与改名前一致。
