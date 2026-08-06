# Benchmark `run/` Path Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run/` the only active Benchmark output root without modifying frozen generated artifacts.

**Architecture:** Change the CLI boundary, repository ignore contract, tracked README, tests, and actionable documentation from the deleted `runs/`/`runs_v2/` names to `run/`. Preserve explicit `--output-root` behavior and all generated artifacts below `run/`; remove the stale ignored WAN local override instead of pretending its deleted checkpoint still exists.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, Git, Markdown, JSON Dataset validation.

## Global Constraints

- `run/` is the only supported top-level run-output root.
- Do not create `runs/` or `runs_v2/` compatibility symlinks.
- Do not rewrite frozen files below generated run directories.
- Preserve explicit `--output-root` behavior.
- Keep `run/README.md` tracked and ignore every other generated child of `run/`.
- Remove `baselines/wan22_lora/baseline.local.json` because its checkpoint was permanently deleted.
- Delete `docs/superpowers/specs/2026-08-06-run-path-migration-design.md` after successful migration, as requested by the user.
- Do not push to GitHub or change the remote.

---

### Task 1: Establish failing path-contract tests

**Files:**
- Modify: `tests/test_visualization_runtime_policy.py`
- Modify: `tests/test_artifacts.py`
- Create: `tests/test_run_root.py`

**Interfaces:**
- Consumes: `physbench.cli.build_parser() -> argparse.ArgumentParser`
- Produces: regression coverage for CLI defaults and repository output-root policy

- [ ] **Step 1: Add CLI default assertions**

Extend `VisualizationRuntimePolicyTest.test_atomic_cli_defaults_off_and_explicit_flag_enables` after parsing the existing `base` and `matrix` argument lists:

```python
self.assertEqual("run", parser.parse_args(base).output_root)
self.assertEqual("run", parser.parse_args(matrix).output_root)
```

- [ ] **Step 2: Update the artifact fixture to the canonical spelling**

Change:

```python
run_dir = root / "runs_v2" / "run"
```

to:

```python
run_dir = root / "run" / "fixture"
```

- [ ] **Step 3: Add a repository-layout test**

Create `tests/test_run_root.py`:

```python
from __future__ import annotations

import subprocess
import unittest

from _paths import ROOT


class RunRootContractTest(unittest.TestCase):
    def test_only_run_root_is_active(self) -> None:
        generated = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "run/example/generated.json",
            ],
            cwd=ROOT,
            check=False,
        )
        readme = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "run/README.md",
            ],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(0, generated.returncode)
        self.assertEqual(1, readme.returncode)
        self.assertTrue((ROOT / "run" / "README.md").is_file())
        self.assertFalse((ROOT / "runs").exists())
        self.assertFalse((ROOT / "runs_v2").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the tests and verify the intended red state**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_visualization_runtime_policy \
  tests.test_artifacts \
  tests.test_run_root
```

Expected: failures report `runs_v2` CLI defaults and missing `run/*` ignore rules.

### Task 2: Migrate active runtime and Git layout

**Files:**
- Modify: `src/physbench/cli.py`
- Modify: `.gitignore`
- Rename/Modify: `runs_v2/README.md -> run/README.md`

**Interfaces:**
- Consumes: Task 1 CLI and repository-layout tests
- Produces: canonical default `output_root == "run"` and safe Git tracking policy

- [ ] **Step 1: Change both CLI defaults**

In `build_parser()`, change the `atomic-run` and `matrix-run` declarations to:

```python
atomic.add_argument("--output-root", default="run")
matrix.add_argument("--output-root", default="run")
```

- [ ] **Step 2: Replace obsolete ignore rules**

Make the output section of `.gitignore` exactly:

```gitignore
run/*
!run/README.md
results/*
!results/README.md
```

- [ ] **Step 3: Update the tracked run-root README**

Retain the physical `run/README.md` and add an opening sentence:

```markdown
`run/` 是当前唯一的 AtomicRun 与 matrix 输出根目录。
```

Do not edit any other file below `run/`.

- [ ] **Step 4: Run the focused tests and verify green**

Run the Task 1 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Verify Git ignores generated run content**

Run:

```bash
git check-ignore -v run/quantity_controls_viewa_v5_seed42_20260729.matrix.json
git status --short --untracked-files=normal
```

Expected: the matrix is ignored; status does not enumerate generated children of `run/`.

### Task 3: Migrate actionable tracked documentation

**Files:**
- Modify: `README.md`
- Modify: `baselines/*/README.md` files containing `runs_v2`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/BASELINE_INTEGRATION.md`
- Modify: `docs/COSMOS3.md`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/OBJECT_CENTRIC_EVALUATION.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/WAN22.md`
- Modify: `docs/WAN22_QUANTITY_EMBEDDING.md`
- Modify: tracked `docs/experiments/*.md` files containing `runs_v2`
- Modify: `docs/superpowers/plans/2026-08-05-flatten-datasets-layout.md`

**Interfaces:**
- Consumes: canonical `run/` contract from Task 2
- Produces: commands and artifact layouts that resolve to the active directory

- [ ] **Step 1: Replace tracked `runs_v2` path tokens**

For every tracked file returned by `git grep -l 'runs_v2'`, replace path token
`runs_v2` with `run`. Do not search or edit generated files below `run/`.

- [ ] **Step 2: Remove deleted legacy-root instructions**

Update the run-retention section of `docs/OPERATIONS.md` so it describes only
`run/` and `results/`. Update the historical flatten-layout plan sentence to
protect frozen `run/`, `results/`, and provenance records without naming the
deleted `runs/` root.

- [ ] **Step 3: Update the root structure diagram**

The final line in the root README tree becomes:

```text
└── run/                     # 当前 AtomicRun 与 matrix 输出
```

- [ ] **Step 4: Run tracked-text gates**

Run:

```bash
git grep -n 'runs_v2' -- ':!docs/superpowers/specs/2026-08-06-run-path-migration-design.md'
git grep -nE '(^|[^A-Za-z0-9_])runs/' -- \
  ':!docs/superpowers/specs/2026-08-06-run-path-migration-design.md'
```

Expected: both commands return no matches.

### Task 4: Remove the stale optional WAN deployment

**Files:**
- Delete: `baselines/wan22_lora/baseline.local.json` (Git-ignored local state)
- Preserve: `baselines/wan22_lora/baseline.local.example.json`
- Preserve: `baselines/wan22_lora/baseline.json`
- Preserve: `baselines/wan22_lora/physics.baseline.json`

**Interfaces:**
- Consumes: existing `unittest.skipUnless` deployment guards in `tests/test_integrated_baselines.py`
- Produces: honest unconfigured state instead of a path to a deleted checkpoint

- [ ] **Step 1: Confirm the failure before deletion**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_integrated_baselines
```

Expected: two errors cite the missing checkpoint below the deleted `runs/`.

- [ ] **Step 2: Delete only the stale local override**

Delete `baselines/wan22_lora/baseline.local.json`. Do not modify the portable
manifest or example.

- [ ] **Step 3: Verify portable discovery and guarded tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python -m physbench baseline list
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_integrated_baselines
```

Expected: WAN identities remain listed; mandatory tests pass and the two
real-deployment tests skip because the local deployment is absent.

### Task 5: Verify the complete migration and remove the design document

**Files:**
- Delete: `docs/superpowers/specs/2026-08-06-run-path-migration-design.md`
- Preserve: every generated file below `run/` other than its README

**Interfaces:**
- Consumes: Tasks 1-4
- Produces: verified internally consistent Benchmark and final GitHub-readiness report

- [ ] **Step 1: Run Dataset validation**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /root/miniconda3/envs/phybench/bin/python scripts/validate_dataset_v12.py
```

Expected: `status=valid`, 799 cases, 3,818 quantities, and `media_changes=0`.

- [ ] **Step 2: Run focused runtime tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest -v \
  tests.test_visualization_runtime_policy \
  tests.test_artifacts \
  tests.test_run_root \
  tests.test_integrated_baselines \
  tests.test_runner
```

Expected: mandatory tests pass; only explicitly unconfigured deployment tests skip.

- [ ] **Step 3: Run the complete suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. \
  /root/miniconda3/envs/phybench/bin/python -m unittest discover -s tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 4: Audit filesystem and tracked paths**

```bash
test -d run
test ! -e runs
test ! -e runs_v2
git check-ignore run/quantity_controls_viewa_v5_seed42_20260729.matrix.json
git diff --check
git status --short --untracked-files=normal
```

Expected: only `run/` exists, generated content is ignored, and no whitespace errors exist.

- [ ] **Step 5: Delete the approved design document**

Delete `docs/superpowers/specs/2026-08-06-run-path-migration-design.md` only
after Steps 1-4 succeed, then run `git diff --check` again.

- [ ] **Step 6: Commit the migration without generated artifacts**

Stage explicit tracked source, test, documentation, ignore, README move, design
deletion, and this plan. Confirm no generated file below `run/` is staged, then
commit:

```bash
git commit -m "refactor: migrate benchmark outputs to run root"
```

- [ ] **Step 7: Report publication readiness accurately**

Report that run-path consistency is complete but GitHub community readiness is
still blocked by the missing license, 2,395 untracked runtime assets, absent
public media download flow, absent CI/lockfile, and machine-local installation
instructions.
