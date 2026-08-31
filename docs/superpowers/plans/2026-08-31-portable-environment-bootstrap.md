# Portable Environment Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command, self-locating VPhysBench environment bootstrap with deterministic planning and certify that the tracked checkout runs after relocation.

**Architecture:** A tiny shell launcher discovers the checkout and selects Python; `physbench.bootstrap` owns deterministic command planning and execution. The existing doctor remains the readiness authority, with a new Dataset-independent runtime level, while release tests execute the exported checkout from a path containing spaces.

**Tech Stack:** Bash, Python 3.11/3.12 standard library, `venv`, pip constraints, unittest, Git archive.

**Spec:** `docs/superpowers/specs/2026-08-31-portable-environment-bootstrap-design.md`

## Global Constraints

- Metadata bootstrap requires Python 3.11 or newer.
- Evaluation bootstrap accepts Python 3.12 or newer, installs PyTorch 2.10.0 from the CUDA 12.8 wheel index, and is CI-certified on Python 3.12 only.
- Dataset media, checkpoints, credentials, Baselines, caches, and Runs must remain untracked.
- Every repository-owned path must derive from the checkout or an explicit option.
- Bootstrap must be idempotent, side-effect-free under `--dry-run`, and must not download Dataset assets or checkpoints.
- Existing `doctor --level metadata|evaluation` behavior remains compatible.

---

### Task 1: Integrate the strict environment doctor and add runtime-only readiness

**Files:**
- Modify: `src/physbench/environment_doctor.py`
- Modify: `src/physbench/dataset_hub.py`
- Modify: `src/physbench/cli.py`
- Test: `tests/test_environment_doctor.py`

**Interfaces:**
- Consumes: existing `Diagnostic` records and `diagnose_project(project_root, level=...)`.
- Produces: `doctor --level runtime`, which performs runtime checks without Dataset/checkpoint requirements.

- [ ] **Step 1: Cherry-pick the reviewed strict-doctor commit and write failing runtime-level tests.**
- [ ] **Step 2: Run `python -m unittest tests.test_environment_doctor -v` and verify rejection of the unknown runtime level.**
- [ ] **Step 3: Implement runtime-level orchestration while preserving full readiness.**
- [ ] **Step 4: Run the focused tests and verify all pass.**

### Task 2: Add deterministic bootstrap planning and execution

**Files:**
- Create: `src/physbench/bootstrap.py`
- Create: `scripts/bootstrap_env.sh`
- Create: `constraints/metadata.txt`
- Create: `constraints/evaluation-cu128.txt`
- Modify: `.gitignore`
- Test: `tests/test_bootstrap_env.py`

**Interfaces:**
- Produces: `python -m physbench.bootstrap --project-root PATH --profile metadata|evaluation [--venv PATH] [--dry-run]`.
- Produces: `bash scripts/bootstrap_env.sh` with root and interpreter auto-detection.

- [ ] **Step 1: Write failing tests for literal command plans, incompatible Python, idempotent venv reuse, dry-run non-mutation, and a checkout path containing spaces.**
- [ ] **Step 2: Run `python -m unittest tests.test_bootstrap_env -v` and verify the bootstrap module is missing.**
- [ ] **Step 3: Implement the minimal planner/executor and shell launcher.**
- [ ] **Step 4: Run focused tests and a real metadata dry-run from outside the checkout.**

### Task 3: Strengthen relocation and publication gates

**Files:**
- Modify: `scripts/verify_release_archive.py`
- Modify: `tests/test_release_archive.py`
- Modify: `tests/test_repository_portability.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: tracked Git archive and metadata-only runtime.
- Produces: relocation verification of help, doctor, Baseline discovery, and interface smoke.

- [ ] **Step 1: Write failing tests using an archive extracted beneath a path with spaces and active files containing POSIX/Windows machine paths.**
- [ ] **Step 2: Run the focused tests and verify they fail against the old gates.**
- [ ] **Step 3: Extend archive execution and path detection without scanning ignored local state.**
- [ ] **Step 4: Run portability, archive, bootstrap, and doctor tests.**

### Task 4: Document and release the `2026-08-31` benchmark

**Files:**
- Modify: `README.md`
- Modify: `docs/GETTING_STARTED.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `RELEASE_MANIFEST.json`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: copy-paste metadata/evaluation bootstrap commands and an explicit portability contract.

- [ ] **Step 1: Update onboarding, operations, release identity, and dependency-light CI coverage.**
- [ ] **Step 2: Run `make test`, `make smoke-interface`, `make release-check`, and `make release-archive-check`.**
- [ ] **Step 3: Run `physbench doctor --json` against the real local environment and parse the output.**
- [ ] **Step 4: Confirm `git ls-files` contains no Baseline, weight, local override, cache, Dataset payload, or Run artifact; commit and push `2026-08-31`.**
