# Flatten Datasets Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the authoritative Dataset roots from `datasets/physics_video/{assets,provenance,releases}` to `datasets/{assets,provenance,releases}` without changing Case physics annotations or media bytes, and align all active downstream path consumers.

**Architecture:** The three roots move together on the same filesystem, so every release descriptor can retain `asset_root: "../.."` and every Case can retain its existing `assets/...` and `provenance/...` relative paths. `src/physbench/data_layout.py` becomes the single runtime source for the flat roots. Raw source archives move physically under provenance while a tracked compatibility symlink at `datasets/assets/source_archives` preserves all frozen release paths and digests.

**Tech Stack:** Python 3.10, `unittest`, JSON/JSONL Dataset manifests, Git, POSIX filesystem links.

## Global Constraints

- Do not change, reorder, extract, or normalize any `physics` object or quantity.
- Do not transcode, copy, or rewrite video, image, mask, ZIP, XLSX, or NPZ payloads.
- Preserve the byte content of every existing release metadata file unless it contains an active documentation or executable path literal that must be updated.
- Preserve 8.0.0 and 9.0.0 asset-lock verification.
- Do not rewrite frozen `runs/`, `runs_v2/`, `results/`, or historical provenance audit records.
- Work in the current checkout as previously authorized; do not create a media-duplicating worktree.

---

### Task 1: Define and test the flat runtime layout

**Files:**
- Create: `tests/test_flat_dataset_layout.py`
- Modify: `src/physbench/data_layout.py`

**Interfaces:**
- Consumes: repository root resolved from `src/physbench/data_layout.py`.
- Produces: `DATASETS_ROOT`, `PHYSICS_VIDEO_ROOT` compatibility alias, `PHYSICS_VIDEO_ASSETS`, `PHYSICS_VIDEO_PROVENANCE`, all historical release constants, and `V9_DATASET` rooted directly below `datasets/`.

- [ ] **Step 1: Write the failing layout test**

```python
def test_authoritative_roots_are_direct_children_of_datasets():
    assert PHYSICS_VIDEO_ROOT == ROOT / "datasets"
    assert PHYSICS_VIDEO_ASSETS == ROOT / "datasets/assets"
    assert PHYSICS_VIDEO_PROVENANCE == ROOT / "datasets/provenance"
    assert V9_DATASET == ROOT / "datasets/releases/9.0.0/dataset.json"
```

- [ ] **Step 2: Run the test and confirm it fails because constants and directories still use `datasets/physics_video`**

Run: `PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest tests.test_flat_dataset_layout -v`

- [ ] **Step 3: Change `data_layout.py` to use the flat root and add `V9_DATASET` without changing the existing `LATEST_DATASET` release selection**

```python
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PHYSICS_VIDEO_ROOT = DATASETS_ROOT
PHYSICS_VIDEO_ASSETS = DATASETS_ROOT / "assets"
PHYSICS_VIDEO_PROVENANCE = DATASETS_ROOT / "provenance"
RELEASES_ROOT = DATASETS_ROOT / "releases"
```

- [ ] **Step 4: Move the three roots and rerun the test**

Expected: the flat-root assertions and both current release existence checks pass.

### Task 2: Preserve ignored media and frozen release resolution

**Files:**
- Modify: `.gitignore`
- Modify: `.gitattributes`
- Move: `datasets/physics_video/assets` to `datasets/assets`
- Move: `datasets/physics_video/provenance` to `datasets/provenance`
- Move: `datasets/physics_video/releases` to `datasets/releases`
- Move: `datasets/assets/source_archives` to `datasets/provenance/source_archives`
- Create: `datasets/assets/source_archives` symlink to `../provenance/source_archives`

**Interfaces:**
- Consumes: release-relative `assets/...` and `provenance/...` paths.
- Produces: identical path resolution from the new `datasets/` asset root without copying payload bytes.

- [ ] **Step 1: Extend the failing test to require no authoritative `datasets/physics_video` directory and to require the source-archive compatibility link**
- [ ] **Step 2: Update ignore and attribute rules before moving media**
- [ ] **Step 3: Move directories on the same filesystem and create the compatibility link**
- [ ] **Step 4: Verify that 8.0.0 and 9.0.0 load with `check_assets=True`**

Run:

```bash
PYTHONPATH=src /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -c \
'from physbench.datasets import load_dataset; from physbench.data_layout import V8_DATASET,V9_DATASET; [load_dataset(p, check_assets=True) for p in (V8_DATASET,V9_DATASET)]'
```

### Task 3: Align active downstream consumers and documentation

**Files:**
- Modify: active path literals in `scripts/`, `tests/`, `README.md`, `Makefile`, `docs/`, and baseline READMEs.
- Modify: `datasets/README.md` and `datasets/assets/README.md`.
- Delete after merge: `datasets/physics_video/README.md`.

**Interfaces:**
- Consumes: `datasets/releases/<version>/dataset.json`, `datasets/assets`, and `datasets/provenance`.
- Produces: runnable maintenance commands and documentation that use only the flat authoritative layout.

- [ ] **Step 1: Replace active executable and test path literals**
- [ ] **Step 2: Update current user-facing documentation while leaving frozen run/result artifacts and historical provenance evidence unchanged**
- [ ] **Step 3: Run the flat-layout tests and relevant Dataset tests**
- [ ] **Step 4: Compare pre/post physics fingerprints and release verification**

The physics fingerprint is the canonical SHA-256 of each release's ordered `(case_id, physics)` sequence. Expected: the pre-migration and post-migration fingerprints are identical for 8.0.0 and 9.0.0.

### Task 4: Final verification and commit

**Files:**
- Review all changed paths with `git status` and `git diff --stat`.

- [ ] **Step 1: Verify no large media payload appears as untracked or staged content**
- [ ] **Step 2: Run `datasets/releases/9.0.0/validate_release.py`**
- [ ] **Step 3: Run the full unit suite and compare failures with the recorded pre-migration baseline of 21 failures, 64 errors, and 12 skips; no new layout-caused failures are allowed**
- [ ] **Step 4: Commit the verified migration as one structural change**
