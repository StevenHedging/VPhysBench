# Per-Object Mask NPZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 原地把 Physics Video Dataset 9.0.0 的 797 个 Case 级汇总 NPZ 拆成 1,205 个逐对象 NPZ，并同步更新全部索引、生成器和验证契约。

**Architecture:** `mask_storage.py` 负责逐对象 `[1,H,W]` 文件和 schema 1.2 manifest；`upgrade_mask_storage.py` 兼容读取当前汇总格式和迁移后的逐对象格式，先全量预检再逐 Case 原子发布。发布构建器为每个 instance 和 Case asset role 建立独立路径，验证器逐文件交叉检查 NPZ、PNG、物理对象映射和资产锁。

**Tech Stack:** Python 3.10、NumPy、OpenCV、`unittest`、现有 `physbench` Dataset loader/资产锁工具。

## Global Constraints

- 直接更新 `datasets/physics_video/releases/9.0.0`，不新建版本。
- 创建 1,205 个 `XX.npz`，删除 797 个 `masks.npz`。
- 每个 `XX.npz` 的 `masks` 为 `uint8 [1,H,W]`、值域 `{0,1}`。
- 每个 NPZ 只包含 `masks`、`mask_ids`、`object_ids`、`frame_index`，并支持 `allow_pickle=False`。
- 保留 1,205 张 `{0,255}` PNG，不改变任何 Mask 像素、几何、顺序或物理对象映射。
- 两个跳过 Case 继续没有 Mask 目录；8.0.0 的 2,032 个锁定资产保持不变。
- 最终 9.0.0 必须有 5,239 个锁定资产和 1,205 个逐对象 NPZ。

---

### Task 1: 逐对象存储模块

**Files:**
- Modify: `datasets/physics_video/releases/9.0.0/mask_storage.py`
- Modify: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Produces: `write_mask_bundle(mask_directory: Path, masks: list[np.ndarray], instances: list[dict[str, Any]]) -> dict[str, Any]`
- Produces: `load_mask_npz(path: Path) -> dict[str, np.ndarray]`
- Produces: `upgrade_manifest_storage(manifest: dict[str, Any]) -> dict[str, Any]`

- [ ] **Step 1: Rewrite bundle tests for per-object archives**

Change the two-mask fixture to require `01.npz` and `02.npz`, no `masks.npz`; assert each archive has exact keys, `masks.shape == (1,H,W)`, one mask/object ID, scalar frame zero, and equality with its PNG. Require storage layout `1HW`, model `asset_pattern`, schema 1.2, per-instance `npz_asset`, and absence of `npz_index`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest discover -s tests -p 'test_first_frame_mask_storage_v9.py' -v
```

Expected: FAIL because the writer still creates one `masks.npz` and schema 1.1 metadata.

- [ ] **Step 3: Implement per-object atomic writing**

Keep bundle-wide shape/cardinality/overlap validation, but write one ordered archive per instance with arrays `masks=stack[index:index+1]`, `mask_ids=[mask_id]`, `object_ids=[object_id]`, `frame_index=int64(0)`. Return storage metadata with `asset_pattern=".../{mask_id}.npz"` and layout `1HW`; upgrade manifests to 1.2, set each instance's `npz_asset`, and remove `npz_index`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all storage/bundle tests PASS.

- [ ] **Step 5: Commit**

```bash
git add datasets/physics_video/releases/9.0.0/mask_storage.py tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: store one mask archive per object"
```

### Task 2: 幂等迁移和 SAM2 生成器

**Files:**
- Modify: `datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py`
- Modify: `datasets/physics_video/releases/9.0.0/generate_first_frame_masks.py`
- Modify: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Consumes: Task 1 writer/loader/manifest functions.
- Produces: preflight operations that use either one legacy `[O,H,W]` summary or a complete set of `[1,H,W]` object archives.
- Produces: upgrade report schema 1.2 with `npz_files=1205` and `removed_summary_npz_files=797` on the current release.

- [ ] **Step 1: Write failing migration tests**

Update the temporary release fixture to start with a real legacy `masks.npz`. Assert read-only preflight accepts it; materialization creates `01.npz`/`02.npz`, deletes `masks.npz` only after validating both new files, writes a schema 1.2 manifest, and is byte/semantic stable on a second run without the legacy file. Corrupt input must leave the legacy file and PNGs untouched.

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 test command. Expected: migration tests FAIL on paths, schema and deletion behavior.

- [ ] **Step 3: Implement dual-input preflight and safe migration**

For each complete Case, derive all `XX.npz` paths from instances. If `masks.npz` exists, validate its ordered planes; otherwise require and validate every object archive. During materialization, write all object files, read them back, update manifest atomically, then unlink that Case's summary archive. Report actual object-file and removed-summary counts.

- [ ] **Step 4: Update the SAM2 generator**

Call the new writer signature, emit schema 1.2 manifests/instances, return the list of instance NPZ assets, and set report schema 1.2. Do not modify localization, segmentation or subject ordering logic.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the focused test command and compile all three release-local scripts. Expected: PASS with no warnings.

- [ ] **Step 6: Commit**

```bash
git add datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py datasets/physics_video/releases/9.0.0/generate_first_frame_masks.py tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: migrate mask archives by object"
```

### Task 3: 9.0.0 索引与验证契约

**Files:**
- Modify: `datasets/physics_video/releases/9.0.0/build_release.py`
- Modify: `datasets/physics_video/releases/9.0.0/validate_release.py`
- Modify: `datasets/physics_video/releases/9.0.0/README.md`
- Modify: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Produces: Case roles `first_frame_subject_mask_npz_XX`.
- Produces: instance field `npz_asset`; removes Case `first_frame_masks_npz`, record `npz_asset`, and instance `npz_index`.
- Produces: `masks.jsonl`/mask annotation schema 1.2.

- [ ] **Step 1: Rewrite builder tests and verify RED**

Require complete fixture Cases to expose two `first_frame_subject_mask_npz_XX` paths, instances to carry matching `npz_asset`, and all summary fields/indices to be absent. Require skipped roles to be null and schema 1.2.

- [ ] **Step 2: Update builder and README**

Derive `XX.npz` next to `XX.png`, attach per-object Case roles, remove summary roles, and emit audit counts `npz_file_count=1205`, `legacy_summary_npz_file_count=0`. Document per-object training reads and `[1,H,W]`.

- [ ] **Step 3: Update independent validator**

For every complete Case, require schema 1.2/storage pattern, validate each instance's Case role/path/archive/IDs and compare `archive["masks"][0]` to PNG. Require 1,205 NPZ, no `masks.npz`, 5,239 locked assets, and null skipped roles. Preserve geometry, ordering, overlap, base hash and loader checks.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run focused tests plus `py_compile`; expected all PASS.

- [ ] **Step 5: Commit**

```bash
git add datasets/physics_video/releases/9.0.0/build_release.py datasets/physics_video/releases/9.0.0/validate_release.py datasets/physics_video/releases/9.0.0/README.md tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: index per-object mask archives"
```

### Task 4: 全量迁移、重建发布和提交

**Files:**
- Generate ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/[0-9][0-9].npz`
- Delete ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/masks.npz`
- Rewrite ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/manifest.json`
- Modify: all tracked 9.0.0 indexes, lock and reports derived by the release tools.

- [ ] **Step 1: Run read-only whole-release preflight**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py --check
```

Expected: 799 selected, 797 complete, 2 skipped, 1,205 masks valid, zero writes.

- [ ] **Step 2: Materialize and remove summaries**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py --materialize --report datasets/physics_video/releases/9.0.0/mask_storage_upgrade_report.json
```

Expected: 1,205 object NPZ created and 797 summary NPZ removed.

- [ ] **Step 3: Rebuild indexes and asset lock**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/build_release.py
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/build_dataset_asset_lock.py --dataset datasets/physics_video/releases/9.0.0/dataset.json
```

Expected: 5,239 locked assets and a new dataset digest.

- [ ] **Step 4: Run independent full-release validation**

```bash
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/validate_release.py
```

Expected: 797 complete, 2 skipped, 1,205 NPZ, 1,205 PNG, no summaries, 5,239 assets, all checks PASS.

- [ ] **Step 5: Commit tracked release changes**

```bash
git add datasets/physics_video/releases/9.0.0
git commit -m "data: split first-frame masks into object archives"
```

### Task 5: Fresh final verification

**Files:**
- Verify all Task 1–4 outputs.

- [ ] **Step 1: Run focused and related regression tests**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest discover -s tests -p 'test_first_frame_mask_storage_v9.py' -v
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest tests.test_dataset_maintenance_scripts tests.test_dataset_contract_v4 tests.test_six_scene_dataset_v8 -v
```

- [ ] **Step 2: Re-run full-release validation and filesystem counts**

Require exactly 1,205 `[0-9][0-9].npz`, zero `masks.npz`, 1,205 PNG, 797 manifests, and pixel equality in representative one-/multi-object Cases.

- [ ] **Step 3: Verify clean tracked state**

Run `git diff --check`, `git status --short`, inspect final commits, and confirm both skipped Case directories remain absent.
