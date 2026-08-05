# First-Frame Mask NPZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 原地更新 Physics Video Dataset 9.0.0，使每个已完成 Case 同时拥有 `{0,1}` 的模型 NPZ 和 `{0,255}` 的逐主体可视化 PNG。

**Architecture:** 新增一个 release-local 的 Mask 存储模块，集中负责二值归一化、安全写入和 NPZ 契约；现有 SAM2 生成器和一次性升级脚本共同调用它。发布构建器只负责把 NPZ 加入 Case/统一索引，独立验证器对 NPZ、PNG、manifest、物理对象映射和资产锁进行全量交叉校验。

**Tech Stack:** Python 3.10、NumPy、OpenCV、`unittest`、现有 `physbench` Dataset loader 和资产锁构建器。

## Global Constraints

- 直接原地更新 `datasets/physics_video/releases/9.0.0`，不创建新版本。
- 仅处理现有 797 个完整 Case 和 1,205 张 Mask；两个跳过 Case保持跳过。
- 不重新运行 SAM2，不改变任何 Mask 的非零像素集合、编号或物理对象映射。
- NPZ 的 `masks` 为 `uint8 [O,H,W]`、值域 `{0,1}`；PNG 为单通道 `uint8 [H,W]`、值域 `{0,255}`。
- NPZ 不包含 object dtype，必须可由 `numpy.load(..., allow_pickle=False)` 读取。
- 8.0.0 的视频、首帧、文本、物理标注、Scene、View 和 2,032 个锁定资产不得变化。

---

### Task 1: Release-local Mask 存储契约

**Files:**
- Create: `datasets/physics_video/releases/9.0.0/mask_storage.py`
- Create: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Produces: `normalize_binary_mask(mask: np.ndarray) -> np.ndarray`
- Produces: `read_binary_png(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray`
- Produces: `write_mask_bundle(mask_directory: Path, masks: list[np.ndarray], instances: list[dict[str, Any]], npz_asset: str) -> dict[str, Any]`; 返回可直接写入 manifest 的 `storage` 对象。
- Produces: `load_mask_npz(path: Path) -> dict[str, np.ndarray]`
- Produces: `upgrade_manifest_storage(manifest: dict[str, Any], npz_asset: str) -> dict[str, Any]`

- [ ] **Step 1: Write failing normalization tests**

Add tests that import `mask_storage.py` by path and assert that `{0,1}` and `{0,255}` inputs normalize to identical `{0,1}` `uint8` arrays, while RGB, empty, and `{0,2}` inputs raise `ValueError`.

- [ ] **Step 2: Run the normalization tests and verify RED**

Run:

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest discover -s tests -p 'test_first_frame_mask_storage_v9.py' -v
```

Expected: FAIL because `mask_storage.py` and its public functions do not exist.

- [ ] **Step 3: Implement binary normalization and PNG reading**

Implement strict dimensionality, dtype, value, non-empty and optional shape checks. Return a C-contiguous `uint8` array containing only 0 and 1.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same discovery command. Expected: normalization tests PASS.

- [ ] **Step 5: Write failing bundle tests**

Using a temporary directory and two small masks, assert the exact NPZ key set
`{"masks", "mask_ids", "object_ids", "frame_index"}`, `uint8 [O,H,W]`, Unicode IDs,
scalar `int64(0)`, `allow_pickle=False` loading, `{0,255}` PNG output, byte-stable NPZ
rewrites, continuous mask IDs, and `npz_index` fields in the upgraded manifest.

- [ ] **Step 6: Run the bundle tests and verify RED**

Run the focused test command. Expected: FAIL because bundle writing/loading and manifest upgrade are not implemented.

- [ ] **Step 7: Implement atomic bundle writing and manifest upgrade**

Validate all masks and instances before writing. Write each NPZ/PNG to a same-directory temporary file and publish with `os.replace`; use `np.savez_compressed` with ordered keys and Unicode arrays. Set manifest `schema_version="1.1"`, remove the ambiguous old top-level `dtype`/`values`, add the exact storage object below, and set each instance's zero-based `npz_index`.

```json
{
  "model": {
    "asset": "assets/<scene>/<case>/canonical/masks/masks.npz",
    "array_key": "masks",
    "layout": "OHW",
    "dtype": "uint8",
    "values": [0, 1]
  },
  "visualization": {
    "asset_pattern": "assets/<scene>/<case>/canonical/masks/{mask_id}.png",
    "dtype": "uint8",
    "values": [0, 255]
  }
}
```

- [ ] **Step 8: Run all storage tests and verify GREEN**

Run the focused test command. Expected: all tests PASS with no warnings.

- [ ] **Step 9: Commit the storage module and tests**

```bash
git add datasets/physics_video/releases/9.0.0/mask_storage.py tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: add dual-format first-frame mask storage"
```

### Task 2: Existing-asset upgrade and future SAM2 generation

**Files:**
- Create: `datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py`
- Modify: `datasets/physics_video/releases/9.0.0/generate_first_frame_masks.py`
- Modify: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Consumes: Task 1 storage functions.
- Produces: `preflight_release(release_root: Path = RELEASE_ROOT, physics_video_root: Path = PHYSICS_VIDEO_ROOT) -> list[dict[str, Any]]`
- Produces: `upgrade_release(*, release_root: Path = RELEASE_ROOT, physics_video_root: Path = PHYSICS_VIDEO_ROOT, materialize: bool, report_path: Path | None) -> dict[str, Any]`
- Produces: generator output with manifest schema 1.1, `npz_asset`, and per-instance `npz_index`.

- [ ] **Step 1: Write failing upgrade tests**

Build a temporary complete Case bundle and assert that preflight accepts old `{0,1}` PNGs, rejects mapping/cardinality/value errors before writes, materialization creates the approved directory structure, and a second materialization preserves NPZ/PNG semantics. Add a source-level generator contract test that requires use of `write_mask_bundle`, manifest schema 1.1, and `npz_asset`.

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 discovery command. Expected: FAIL because the upgrade module and generator integration are absent.

- [ ] **Step 3: Implement two-pass release upgrade**

Read current `masks.jsonl` and Case manifests, preflight all 799 records without writing, then materialize only the 797 complete records through `write_mask_bundle`. Preserve all localization, segmentation, geometry, physics keys and object IDs; create no directory for skipped records. Emit `mask_storage_upgrade_report.json` with counts and skipped IDs.

- [ ] **Step 4: Update the SAM2 generator**

Add `npz_index` while constructing instances, describe the two storage representations in schema 1.1 manifests, call `write_mask_bundle` under `--materialize`, return `npz_asset`, and emit report schema 1.1. Do not change localization or SAM2 inference code.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the focused test command. Expected: all tests PASS.

- [ ] **Step 6: Commit upgrade/generator support**

```bash
git add datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py datasets/physics_video/releases/9.0.0/generate_first_frame_masks.py tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: materialize model mask archives"
```

### Task 3: Release metadata and full validator contracts

**Files:**
- Modify: `datasets/physics_video/releases/9.0.0/build_release.py`
- Modify: `datasets/physics_video/releases/9.0.0/validate_release.py`
- Modify: `datasets/physics_video/releases/9.0.0/README.md`
- Modify: `tests/test_first_frame_mask_storage_v9.py`

**Interfaces:**
- Consumes: schema 1.1 manifests and `masks.npz` assets from Task 2.
- Produces: `cases.jsonl` field `assets.first_frame_masks_npz`.
- Produces: `masks.jsonl` schema 1.1 field `npz_asset` and per-instance `npz_index`.
- Produces: validation report counts `npz_file_count=797`, `mask_file_count=1205`, `asset_file_count=4831`.

- [ ] **Step 1: Write failing release-contract tests**

Assert the builder source and a small transformed fixture expose `first_frame_masks_npz`, `npz_asset`, schema 1.1 and contiguous zero-based `npz_index`; assert skipped records expose null assets. Add validator unit assertions for exact NPZ keys/dtypes/IDs, PNG `{0,255}`, and pixel equality between `(png > 0)` and `masks[k]`.

- [ ] **Step 2: Run focused tests and verify RED**

Run the focused test command. Expected: FAIL because builder/validator still implement the old PNG-only contract.

- [ ] **Step 3: Update release builder and documentation**

Deep-copy report instances, add `npz_index`, derive `canonical/masks/masks.npz`, add the Case and record fields, set mask annotation schemas to 1.1, and add NPZ/PNG counts and roles to the audit. Document NPZ as the training source and PNG as visualization-only.

- [ ] **Step 4: Update independent full-release validation**

Exclude `first_frame_masks_npz` when comparing Case facts to 8.0.0. For complete records, cross-check Case/record/manifest paths and schema, load NPZ with `allow_pickle=False`, validate exact arrays, compare every PNG to its NPZ plane, and preserve all geometry/order/exclusivity checks. For skipped records, require every Mask asset to be null or absent.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the focused test command. Expected: all tests PASS.

- [ ] **Step 6: Commit metadata/validation code**

```bash
git add datasets/physics_video/releases/9.0.0/build_release.py datasets/physics_video/releases/9.0.0/validate_release.py datasets/physics_video/releases/9.0.0/README.md tests/test_first_frame_mask_storage_v9.py
git commit -m "feat: index and validate mask npz assets"
```

### Task 4: Materialize all 9.0.0 assets and rebuild the frozen release

**Files:**
- Generate ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/masks.npz`
- Rewrite ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/[0-9][0-9].png`
- Rewrite ignored assets: `datasets/physics_video/assets/*/*/canonical/masks/manifest.json`
- Modify: `datasets/physics_video/releases/9.0.0/cases.jsonl`
- Modify: `datasets/physics_video/releases/9.0.0/masks.jsonl`
- Modify: `datasets/physics_video/releases/9.0.0/dataset.json`
- Modify: `datasets/physics_video/releases/9.0.0/mask_release_audit.json`
- Create: `datasets/physics_video/releases/9.0.0/mask_storage_upgrade_report.json`
- Modify: `datasets/physics_video/releases/9.0.0/assets.lock.json`
- Modify: `datasets/physics_video/releases/9.0.0/release.json`

**Interfaces:**
- Consumes: Tasks 1–3 scripts.
- Produces: fully indexed and hash-locked 9.0.0 dual-format asset tree.

- [ ] **Step 1: Run whole-release preflight without writes**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py --check
```

Expected: 799 selected, 797 complete, 2 skipped, 1,205 binary inputs valid, zero writes.

- [ ] **Step 2: Materialize NPZ, visualization PNG, and manifests**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/upgrade_mask_storage.py --materialize --report datasets/physics_video/releases/9.0.0/mask_storage_upgrade_report.json
```

Expected: 797 NPZ and 1,205 PNG materialized; the same two Case IDs remain skipped.

- [ ] **Step 3: Rebuild Case and mask indexes**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/build_release.py
```

Expected: `cases=799 complete=797 skipped=2 masks=1205 npz=797`.

- [ ] **Step 4: Rebuild asset lock and release digest transactionally**

```bash
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/build_dataset_asset_lock.py --dataset datasets/physics_video/releases/9.0.0/dataset.json
```

Expected: 4,831 locked files and a newly computed 9.0.0 dataset digest.

- [ ] **Step 5: Run the independent full-release validator**

```bash
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python datasets/physics_video/releases/9.0.0/validate_release.py
```

Expected: 799 Case, 797 NPZ, 1,205 PNG, 2 skipped, 4,831 assets, and all checks pass.

- [ ] **Step 6: Commit the rebuilt tracked release metadata**

```bash
git add datasets/physics_video/releases/9.0.0
git commit -m "data: add model-readable first-frame mask archives"
```

### Task 5: Final regression and artifact verification

**Files:**
- Verify all files changed by Tasks 1–4.

**Interfaces:**
- Produces: final evidence that code, release metadata, local ignored assets and Git state agree.

- [ ] **Step 1: Run focused Mask tests fresh**

```bash
/mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest discover -s tests -p 'test_first_frame_mask_storage_v9.py' -v
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run existing Dataset maintenance regressions**

```bash
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest tests.test_dataset_maintenance_scripts tests.test_dataset_contract_v4 tests.test_six_scene_dataset_v8 -v
```

Expected: all existing tests PASS.

- [ ] **Step 3: Re-run full-release validation fresh**

Run the Task 4 validator command again. Expected: the same counts and digest as the first successful validation.

- [ ] **Step 4: Verify filesystem counts and representative arrays**

Count `masks.npz`, `[0-9][0-9].png`, and manifests under the asset tree; load a one-object and a multi-object archive with `allow_pickle=False`; confirm exact shapes, values and mappings. Expected counts: 797 NPZ, 1,205 PNG, 797 manifests.

- [ ] **Step 5: Verify tracked scope and clean status**

Run `git diff --check`, inspect commits since `2e54050`, and confirm `git status --short` is empty. Confirm the two skipped Case paths still have no `canonical/masks/` directory.
