# VPhysBench Portability Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Rename the complete working repository to `VPhysBench` and remove machine-specific absolute paths from its active publication surface without rewriting historical provenance or generated artifacts.

**Architecture:** A repository contract test defines the portable active surface and explicitly excludes historical/frozen evidence. Text, package metadata, baseline examples, and scaffold output are migrated before the physical directory is atomically renamed; the same contract then verifies the new basename. Existing `physbench` imports and CLI remain stable.

**Tech Stack:** Python 3.11, `unittest`, `tomllib`, Git, Markdown, JSON, POSIX filesystem rename.

## Global Constraints

- The physical repository directory and public display name become `VPhysBench`.
- The Python distribution name becomes `vphysbench`; package and CLI names remain `physbench`.
- Active tracked files must not contain `/root/`, `/mnt/`, `/absolute/`, or the old repository token `physics_video_benchmark`.
- Preserve original strings in `datasets/provenance/`, `docs/experiments/`, frozen protocols `scene_default_v4.json` through `scene_default_v7.json`, and generated `run/`, `results/`, and `cache/` content.
- Preserve ignored `baseline.local.json` deployment state; do not stage it.
- Do not copy or rewrite Dataset media, run artifacts, caches, model weights, or source archives.
- Delete `docs/superpowers/specs/2026-08-06-vphysbench-portability-rename-design.md` only after successful migration verification.
- Do not push or modify the Git remote.

---

### Task 1: Establish failing portability contracts

**Files:**
- Create: `tests/test_repository_portability.py`
- Modify: `tests/test_managed_baselines.py`

**Interfaces:**
- Consumes: Git tracked-path inventory and `create_baseline_scaffold(...) -> Path`
- Produces: `RepositoryPortabilityTest` and relative scaffold-example regression coverage

- [x] **Step 1: Add the repository portability test**

Create `tests/test_repository_portability.py` with a tracked-text scanner that excludes `tests/`, `docs/superpowers/`, `datasets/provenance/`, `docs/experiments/`, and frozen protocol files v4-v7. Decode only UTF-8 tracked files and assert that every remaining file lacks `/root/`, `/mnt/`, `/absolute/`, and `physics_video_benchmark`. Add separate assertions that `ROOT.name == "VPhysBench"`, README starts with `# VPhysBench`, and `tomllib.loads(pyproject)["project"]["name"] == "vphysbench"`.

```python
FORBIDDEN = ("/root/", "/mnt/", "/absolute/", "physics_video_benchmark")
FROZEN_PROTOCOLS = {
    f"configs/evaluation/protocols/scene_default_v{version}.json"
    for version in range(4, 8)
}
```

- [x] **Step 2: Add the scaffold portability assertion**

In `ManagedBaselineTests.test_scaffolds_pass_registry_validation`, load each generated `baseline.local.example.json` and assert:

```python
self.assertFalse(Path(example["model"]["checkpoint"]).is_absolute())
if backend == "submission":
    self.assertFalse(
        Path(example["runtime"]["submission_manifest"]).is_absolute()
    )
```

- [x] **Step 3: Run both tests and verify the intended red state**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_repository_portability \
  tests.test_managed_baselines.ManagedBaselineTests.test_scaffolds_pass_registry_validation
```

Expected: repository assertions list current machine paths, old branding, and old basename; scaffold assertions report absolute generated examples.

### Task 2: Migrate brand, active documentation, and baseline examples

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `baselines/causal_forcing_pp_2step_i2v/README.md`
- Modify: `baselines/cosmos3_nano_i2v/README.md`
- Modify: `baselines/wan22_g15_sparse_motion/README.md`
- Modify: `baselines/wan22_lora/README.md`
- Modify: `baselines/wan22_quantity_embedding/README.md`
- Modify: `baselines/causal_forcing_pp_2step_i2v/baseline.local.example.json`
- Modify: `baselines/cosmos3_nano_i2v/baseline.local.example.json`
- Modify: `baselines/wan22_g15_sparse_motion/baseline.local.example.json`
- Modify: `baselines/wan22_lora/baseline.local.example.json`
- Modify: `baselines/wan22_quantity_embedding/baseline.local.example.json`
- Modify: `docs/BASELINE_INTEGRATION.md`
- Modify: `docs/COSMOS3.md`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/NEW_SCENE_EVALUATOR.md`
- Modify: `docs/OBJECT_CENTRIC_EVALUATION.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/WAN22.md`
- Modify: `docs/WAN22_QUANTITY_EMBEDDING.md`

**Interfaces:**
- Consumes: forbidden-token policy from Task 1
- Produces: portable user-facing commands and `vphysbench` distribution metadata

- [x] **Step 1: Migrate public branding**

Change the root README heading to `# VPhysBench`, its repository tree root to `VPhysBench/`, and the `[project].name` value in `pyproject.toml` to `vphysbench`. Preserve `physbench = "physbench.cli:main"` and the `src/physbench` package.

- [x] **Step 2: Replace environment-bound Python commands**

In the listed README and current docs, replace `/root/miniconda3/envs/phybench/bin/python` with `python`, its `pip` with `python -m pip`, and absolute `physbench` executables with `physbench`. Replace old absolute `cd` commands with `cd VPhysBench` where repository entry is instructional; otherwise remove redundant `cd` lines.

- [x] **Step 3: Replace active audit-output and external-checkout examples**

Use repository-relative output paths such as `evaluation_audits/<audit_id>` and `evaluator_audits/<audit_id>`. Replace the SAM2 checkout example with `../sam2` and other generic `/absolute/...` documentation examples with `../external/...` values.

- [x] **Step 4: Make all five local override examples relative**

Change each `/absolute/path/to/...` value in the five tracked `baseline.local.example.json` files to a purpose-specific path below `../external/`, retaining every JSON field and `null` value.

- [x] **Step 5: Run the portability test and inspect remaining failures**

Run the Task 1 test command. Expected: only the physical basename assertion and scaffold-generated example assertions remain red.

### Task 3: Make generated baseline scaffolds portable

**Files:**
- Modify: `src/physbench/baseline_runtime/scaffold.py`
- Test: `tests/test_managed_baselines.py`

**Interfaces:**
- Consumes: scaffold portability assertion from Task 1
- Produces: relative `baseline.local.example.json` values from `create_baseline_scaffold`

- [x] **Step 1: Change scaffold placeholders**

Set the generated model checkpoint to `../external/checkpoint` and the submission manifest to `../external/submission.jsonl`. Do not change portable `baseline.json` or runtime resolution semantics.

- [x] **Step 2: Verify scaffold green and repository basename still red**

Run the Task 1 command. Expected: scaffold test passes; repository test fails only because the physical directory is still named `physics_video_benchmark`.

### Task 4: Rename the physical working repository

**Files:**
- Move directory: `/root/Steven/physics_video_benchmark -> /root/Steven/VPhysBench`

**Interfaces:**
- Consumes: portable tracked tree from Tasks 1-3
- Produces: Git root `/root/Steven/VPhysBench`

- [x] **Step 1: Run rename preflight**

Verify the source is the current Git root, `/root/Steven/VPhysBench` does not exist, branch is `main`, and status contains no staged or untracked generated artifacts. The pre-existing untracked physical-response-loss plan must remain untracked and preserved.

- [x] **Step 2: Atomically rename the repository directory**

Run from `/root/Steven`:

```bash
mv -- /root/Steven/physics_video_benchmark /root/Steven/VPhysBench
```

- [x] **Step 3: Verify the moved Git workspace**

From `/root/Steven/VPhysBench`, confirm `git rev-parse --show-toplevel`, `git remote -v`, branch, `run/`, Dataset, cache, ignored local overrides, and relative Dataset source-archive symlink. Confirm the old directory is absent.

- [x] **Step 4: Run the portability tests green**

Run the Task 1 command from the new root. Expected: all selected tests pass.

### Task 5: Verify Benchmark behavior and remove the approved design

**Files:**
- Delete: `docs/superpowers/specs/2026-08-06-vphysbench-portability-rename-design.md`
- Preserve: `docs/superpowers/plans/2026-08-06-vphysbench-portability-rename.md`

**Interfaces:**
- Consumes: renamed workspace and green portability contracts
- Produces: verified migration commit without the temporary design document

- [x] **Step 1: Validate Dataset v12**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python scripts/validate_dataset_v12.py
```

Expected: `status=valid`, 799 cases, 3,818 quantities, and `media_changes=0`.

- [x] **Step 2: Verify baseline discovery and focused runtime contracts**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python -m physbench baseline list

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_repository_portability \
  tests.test_visualization_runtime_policy \
  tests.test_artifacts \
  tests.test_run_root \
  tests.test_integrated_baselines \
  tests.test_runner \
  tests.test_managed_baselines.ManagedBaselineTests.test_scaffolds_pass_registry_validation
```

Expected: nine baseline identities; mandatory focused tests pass with only explicitly unconfigured real deployments skipped.

- [x] **Step 3: Run the complete test suite and classify results**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest discover -s tests -v
```

Compare with the known pre-migration baseline of 531 tests, 17 failures, 14 errors, and 13 skips. Any new failure touching portability, root resolution, Dataset validation, baseline discovery, run root, or scaffold output blocks completion.

- [x] **Step 4: Delete the approved design document**

Use `apply_patch` to delete `docs/superpowers/specs/2026-08-06-vphysbench-portability-rename-design.md`, then run `git diff --check` and the portability test again.

- [x] **Step 5: Stage only migration files and commit**

Explicitly stage the files listed in Tasks 1-5 and this plan. Verify no `run/`, Dataset media, cache, `baseline.local.json`, or unrelated untracked plan is staged. Commit:

```bash
git commit -m "refactor: rename benchmark to VPhysBench"
```

- [x] **Step 6: Report final state**

Report the new filesystem root, commit, preserved interfaces, scoped path-gate result, Dataset and focused-test evidence, exact full-suite status, and any remaining GitHub publication blockers. Do not push.
