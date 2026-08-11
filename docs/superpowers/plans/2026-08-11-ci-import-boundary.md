# CI Import Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the lightweight GitHub release check by removing eager evaluator imports from Task/protocol interfaces and giving CI capability-specific test targets.

**Architecture:** A focused `task_compiler` module owns Dataset/Task/Baseline compilation without importing evaluator code. Package initializers expose lightweight functions directly and resolve heavy runtime functions lazily. CI verifies the lightweight and evaluation capabilities separately.

**Tech Stack:** Python 3.11, `unittest`, GitHub Actions, Make.

## Global Constraints

- Keep Task v1, Evaluation v1, Dataset 13.0.0, and all scoring behavior unchanged.
- Do not add a registered baseline, model runtime, weight, local configuration, or run output.
- The `hub` environment must not require OpenCV, NumPy, SciPy, PyTorch, or SAM 2 for metadata, planning, compilation, or release checks.

---

### Task 1: Protect the lightweight public import contract

**Files:**
- Create: `tests/test_lightweight_import_boundaries.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `physbench.evaluation.load_evaluation_protocol` and `physbench.orchestration.compile_task_instance`.
- Produces: a subprocess regression test that fails if those imports touch heavy optional modules.

- [ ] **Step 1: Write the failing test**

  Start a fresh Python subprocess with a `MetaPathFinder` that raises
  `ModuleNotFoundError` for `cv2`, `numpy`, `scipy`, `torch`, and `sam2`. Import
  the two lightweight APIs, load `scene_default_v1`, and assert that none of
  the blocked module roots appears in `sys.modules`.

- [ ] **Step 2: Run the test to verify RED**

  Run:
  `PYTHONPATH=src:tests:. python -m unittest tests.test_lightweight_import_boundaries -v`

  Expected: failure at `cv2` from the current eager package import chain.

- [ ] **Step 3: Add the test to `test-interface`**

  Define `test-interface` as the lightweight release suite and keep `test` as
  its compatibility alias.

### Task 2: Extract the Task compiler from AtomicRun execution

**Files:**
- Create: `src/physbench/orchestration/task_compiler.py`
- Modify: `src/physbench/orchestration/atomic_runner.py`
- Modify: `src/physbench/orchestration/__init__.py`

**Interfaces:**
- Produces: `compile_task_instance(plugin, dataset, task) -> BaselineTaskInstance` and `build_task_instance(...)->BaselineTaskInstance`.
- Consumes: Dataset loader, Task loader/planner, Baseline bundle/plugin API.

- [ ] **Step 1: Move only compiler responsibilities**

  Move `compile_task_instance` and `build_task_instance` unchanged into the
  focused module. Import them into `atomic_runner` for internal use.

- [ ] **Step 2: Make orchestration runtime exports lazy**

  Export compiler functions directly. Implement module `__getattr__` mappings
  for `run_atomic`, `run_matrix`, `reevaluate_atomic`, and
  `reevaluate_atomic_variant`, importing their owner module only when requested.

- [ ] **Step 3: Verify GREEN**

  Run the new boundary test and `tests.test_managed_baselines`.

### Task 3: Make Evaluation protocol loading independent

**Files:**
- Modify: `src/physbench/evaluation/__init__.py`
- Modify: `src/physbench/orchestration/atomic_runner.py`

**Interfaces:**
- Direct export: `load_evaluation_protocol`.
- Lazy exports: `evaluate_task`, `aggregate_task_results`.

- [ ] **Step 1: Add lazy evaluator exports**

  Keep `load_evaluation_protocol` directly imported from `protocols`; resolve
  evaluator functions from `task_evaluator` only on attribute access.

- [ ] **Step 2: Defer evaluation activation**

  Import `evaluate_task` inside the AtomicRun evaluation phase and wrap a
  missing optional dependency in an error that directs users to install
  `.[scene-evaluation]`.

- [ ] **Step 3: Run the lightweight suite**

  Run `make test-interface` and confirm all tests pass without evaluator extras.

### Task 4: Split GitHub Actions capability checks

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Makefile`
- Modify: `docs/GETTING_STARTED.md`
- Modify: `README.md`

**Interfaces:**
- Produces: required lightweight `interface-and-release` job and explicit
  evaluator contract target/job.

- [ ] **Step 1: Point the release job at `make test-interface`**

  Preserve smoke, release audit, and archive verification as independent steps.

- [ ] **Step 2: Add the evaluator job**

  Install the scene evaluator extra and run a CPU fixture subset through
  `make test-evaluation`. Do not download the private Dataset in CI.

- [ ] **Step 3: Update installation guidance**

  State clearly which commands require `.[hub]` and which require
  `.[scene-evaluation]`.

- [ ] **Step 4: Verify and publish**

  Recreate a clean hub-only Python 3.11 environment, run all release targets,
  run the evaluator target in the existing full environment, audit the diff,
  commit, push `2026-08-11`, and inspect the new GitHub Actions result.
