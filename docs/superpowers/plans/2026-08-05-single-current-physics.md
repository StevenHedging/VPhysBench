# Single Current Physics Annotation Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish Dataset 12.0.0 as the only active runtime Release, with one corrected `physics.json` per Case and no version-suffixed or legacy physics documents.

**Architecture:** Derive V12 metadata from the frozen V11 Case facts, replace only the Case-local physics asset path and Dataset/Task identities, and atomically materialize V11 physics content under `physics.json`. Retire old runtime Release directories after V12 passes full hash validation; Baselines and evaluators continue consuming inline schema-5 physics unchanged.

**Tech Stack:** Python 3.10, `unittest`, JSON/JSONL, SHA-256 asset locks, POSIX atomic rename, Git.

## Global Constraints

- Preserve all 799 Case IDs, six Scene IDs, ordered membership, Views, prompts, inline physics, media roles, media bytes, masks, appearance, temporal, alignment, and source provenance facts from V11.
- Final assets contain exactly 799 tracked `physics.json` and zero tracked `physics.v11.json`.
- Do not modify any video, image, mask, archive, workbook, or source media.
- V12 runtime Release contains exactly seven entries: `README.md`, `dataset.json`, `release.json`, `cases.jsonl`, `assets.lock.json`, `scenes/`, and `views/`.
- V12 remains Dataset/Case schema 5.0 and physics-document schema 2.0.
- Official plan counts remain 582 train Cases, 76 finetune jobs, and 658 direct jobs.
- Runtime Release directories 1.0.0 through 11.0.0 are removed only after V12 full-hash validation succeeds.

---

### Task 1: Specify the V12 single-file contract

**Files:**
- Create: `tests/test_single_current_physics_v12.py`
- Create: `scripts/build_dataset_v12.py`

**Interfaces:**
- Consumes: V11 ordered Cases, Scenes, Views, asset lock, and release identity.
- Produces: `migrate_case(case)`, `prepare_v12(cases, datasets_root)`, and deterministic schema-2 `physics.json` bytes.

- [ ] **Step 1: Write failing tests** asserting 799 ordered Case identities, identical inline physics/prompts/Views, path replacement from `physics.v11.json` to `physics.json`, deterministic document bytes, and the exact 5,286/494/715 coverage relation.
- [ ] **Step 2: Run** `PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest tests.test_single_current_physics_v12 -v` and require import failure because the builder does not exist.
- [ ] **Step 3: Implement pure migration helpers** that deep-copy V11, change only Dataset/Case identity and `assets.physics_annotation`, and generate documents from V11 inline physics.
- [ ] **Step 4: Re-run the Task 1 test and require all pure-function tests pass.**
- [ ] **Step 5: Commit** with `feat(dataset): prepare single-file physics release`.

### Task 2: Publish and independently validate V12

**Files:**
- Modify: `scripts/build_dataset_v12.py`
- Create: `scripts/validate_dataset_v12.py`
- Modify: `tests/test_single_current_physics_v12.py`
- Create: `datasets/releases/12.0.0/**`
- Replace: `datasets/assets/*/*/physics.json`
- Delete: `datasets/assets/*/*/physics.v11.json`
- Create: `datasets/provenance/releases/12.0.0/{migration.json,validation.json}`

**Interfaces:**
- Produces: `build_release(check=False)` and `validate_v12(dataset_path, write_report=False)`.

- [ ] **Step 1: Add failing Release tests** for the seven-entry tree, 799 `physics.json`, zero versioned documents, 6,038 lock entries, unchanged 5,239 non-physics asset records, and unchanged media hashes.
- [ ] **Step 2: Implement staged construction**: preflight all candidates, stage documents, atomically replace Case-local physics files, stage Release metadata, rebuild the lock, and run a complete Loader hash check before publication.
- [ ] **Step 3: Implement an independent validator** that checks identities, documents, quantity/prompt rules, Views, media invariance, asset hashes, and migration counts without calling builder helpers.
- [ ] **Step 4: Publish V12 and write validation evidence.** Expected summary: `cases=799 physics=799 locked_assets=6038 media_changes=0`.
- [ ] **Step 5: Run the V12 tests and validator; require zero failures.**
- [ ] **Step 6: Commit only metadata changes; confirm no staged media extension.** Commit `data: publish single-file physics release v12`.

### Task 3: Retire old runtime Releases and switch defaults

**Files:**
- Delete: `datasets/releases/{1.0.0,...,11.0.0}/**`
- Modify: `src/physbench/data_layout.py`
- Modify: `tasks/official/five_scene_direct_eval.json`
- Modify: `tasks/official/five_scene_finetune_eval.json`
- Modify: current Dataset/Task/Baseline tests

**Interfaces:**
- Produces: `V12_RELEASE_ROOT`, `V12_DATASET`, and `LATEST_DATASET=V12_DATASET` with official `_v12` Task identities.

- [ ] **Step 1: Write failing default tests** requiring V12 identity, one active Release directory, zero active references to retired Dataset IDs, and unchanged plan counts.
- [ ] **Step 2: Switch current constants and official Tasks to V12.** Historical constants may remain only where a clearly archived script needs Git-history reproduction; no active module or test may import them.
- [ ] **Step 3: Delete runtime Release directories 1.0.0 through 11.0.0 after resolving and printing their exact paths.** Do not delete anything under `datasets/assets` except the two physics metadata filenames covered by Task 2.
- [ ] **Step 4: Run current Dataset, TaskBuilder, integrated Baseline, entity-manifest, and maintenance tests; require zero failures.**
- [ ] **Step 5: Commit** with `refactor(dataset): retire legacy runtime releases`.

### Task 4: Synchronize current documentation and maintenance surface

**Files:**
- Modify: `README.md`, `Makefile`, `datasets/*.md`, `docs/*.md`, active Baseline READMEs
- Modify or remove: V10/V11 builders, validators, and historical tests that require retired runtime paths

**Interfaces:**
- Produces: one documented current entry, `datasets/releases/12.0.0/dataset.json`, and one Case-local filename, `physics.json`.

- [ ] **Step 1: Search active text and code** for retired Release paths/IDs, `physics.v11.json`, `V10_DATASET`, and `V11_DATASET`; classify explicit Git-history records separately.
- [ ] **Step 2: Update current documentation** to explain V12 schema-5 quantities and the single-file Case layout without presenting retired Releases as loadable.
- [ ] **Step 3: Remove or quarantine maintenance code/tests whose only function is loading deleted runtime Releases.** Preserve historical experiment reports as immutable prose.
- [ ] **Step 4: Run `git diff --check`, JSON parsing, Python compilation, and current documentation-path searches.**
- [ ] **Step 5: Commit** with `docs(dataset): document single current release`.

### Task 5: Final verification

**Files:**
- Verify only; fix scoped defects with a failing test first.

**Interfaces:**
- Produces: fresh completion evidence.

- [ ] **Step 1: Run `scripts/validate_dataset_v12.py` with all 6,038 SHA-256 checks.**
- [ ] **Step 2: Run the current release gate** covering Loader/schema, single-file migration, Dataset/Task, all active Baselines, entity manifest, and maintenance tests.
- [ ] **Step 3: Verify repository scope:** 799 `physics.json`, zero `physics.v11.json`, one runtime Release, no media changes since `478f1d5`, clean diff checks, and all changed Python files compile.
- [ ] **Step 4: Run the full unit suite and classify any remaining failures as current defects or explicitly retired historical coverage.**
- [ ] **Step 5: Run V12 validation and the current release gate again after the final commit; report commit IDs, Dataset/asset digests, counts, evidence paths, and worktree status.**
