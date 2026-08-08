# Clean Near-Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a date-only Git branch whose checkout contains the complete VPhysBench evaluator and generic baseline extension contracts, but no integrated model baseline, weight, model-specific configuration, or historical run.

**Architecture:** Preserve the current top-level repository layout. `baselines/` remains the user extension root and `run/<run_id>/` remains the only runtime-output root; both are empty in a fresh checkout except for documentation. Dataset media is resolved from the immutable Hugging Face Dataset binding, while generic managed-I2V/V2V and submission contracts remain in `physbench`.

**Tech Stack:** Python 3.11, setuptools, unittest, Hugging Face Hub CLI, JSON/JSONL schemas, GitHub Actions.

## Global Constraints

- The release branch name is exactly `2026-08-08`; no baseline name appears in the branch name.
- Preserve the current directory structure unless a path exists only for a concrete baseline or historical experiment.
- A fresh checkout has zero discovered baseline bundles.
- User-created baselines live under `baselines/<baseline_id>/`.
- All runtime predictions, evaluation outputs, logs, and artifacts live under `run/<run_id>/`.
- Keep five officially scored scenes and two preview/data-only scenes.
- Keep managed I2V, managed V2V, and submission integration contracts.
- The mock I2V command is a protocol fixture, is not registered as a baseline, and is never described as a reference score.
- Dataset downloads use the exact commit in `datasets/huggingface.json`; never silently use `main`.
- Do not commit tokens, private keys, local absolute paths, model weights, checkpoints, caches, or historical run outputs.
- Preserve concrete baseline work on the existing development branch; perform release cleanup in an isolated worktree.

---

### Task 1: Preserve the Dataset binding and create the isolated date branch

**Files:**
- Add on development branch: `datasets/HF_DATASET_CARD.md`
- Add on development branch: `datasets/huggingface.json`
- Add on development branch: `src/physbench/huggingface_binding.py`
- Add on development branch: `tests/test_huggingface_dataset_binding.py`
- Modify on development branch: `datasets/README.md`
- Add on development branch: `docs/superpowers/plans/2026-08-08-clean-near-release.md`

**Interfaces:**
- Consumes: current dirty worktree and current `feat/wan22-symbol-value-cross-attention` HEAD.
- Produces: a committed safety checkpoint and an isolated worktree on branch `2026-08-08`.

- [ ] **Step 1: Run the focused binding test before committing**

Run:

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_huggingface_dataset_binding -v
```

Expected: two tests pass and no credential is printed.

- [ ] **Step 2: Verify the staged scope and secret hygiene**

Run:

```bash
git diff --check
git status --short
git grep -nE 'hf_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' -- . ':!docs/superpowers/plans/2026-08-08-clean-near-release.md'
```

Expected: only the Dataset binding, its documentation/test, and this plan are pending; the secret scan has no matches.

- [ ] **Step 3: Commit the safety checkpoint on the development branch**

```bash
git add datasets/README.md datasets/HF_DATASET_CARD.md datasets/huggingface.json src/physbench/huggingface_binding.py tests/test_huggingface_dataset_binding.py docs/superpowers/plans/2026-08-08-clean-near-release.md
git commit -m "chore: bind immutable VPhysBench dataset release"
```

- [ ] **Step 4: Create the isolated release worktree**

```bash
git worktree add -b 2026-08-08 /root/Steven/.worktrees/VPhysBench-2026-08-08 HEAD
```

Expected: the original working tree stays on the development branch and the new worktree reports branch `2026-08-08`.

---

### Task 2: Define executable release-hygiene rules

**Files:**
- Create: `scripts/release_audit.py`
- Create: `tests/test_release_audit.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: repository root as `Path` and Git tracked-file listing.
- Produces: `audit_release(root: Path) -> list[str]` and `make release-check`.

- [ ] **Step 1: Write failing audit tests**

Create tests that construct temporary repositories and assert that the audit reports:

```python
def test_rejects_registered_baseline_bundle(self):
    self.write("baselines/model/baseline.json", "{}")
    self.assertIssue("tracked baseline bundle")

def test_rejects_model_weight(self):
    self.write("models/model.safetensors", "fixture")
    self.assertIssue("model weight or checkpoint")

def test_accepts_release_skeleton(self):
    self.write("baselines/README.md", "# Baselines")
    self.write("run/README.md", "# Runs")
    self.assertEqual([], audit_release(self.root, tracked_files=self.files()))
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_release_audit -v
```

Expected: import failure because `scripts.release_audit` does not exist.

- [ ] **Step 3: Implement the audit**

Implement exact checks for:

```python
WEIGHT_SUFFIXES = {
    ".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"
}
FORBIDDEN_ROOTS = {"cache", "results"}
FORBIDDEN_MODEL_MARKERS = {
    "wan22", "cosmos3", "causal_forcing", "quantity_embedding",
    "symbol_value_cross_attention",
}
LOCAL_ONLY_NAMES = {"baseline.local.json"}
```

The CLI exits 0 with `release_audit=ok` when no issue exists and exits 1 after printing every issue otherwise. Scan tracked paths and tracked UTF-8 text; reject `/root/`, `/mnt/`, credential patterns, tracked `run/` children other than `run/README.md`, and tracked baseline bundles under `baselines/`.

- [ ] **Step 4: Add the Make target and verify pass/fail behavior**

```make
.PHONY: release-check
release-check:
	PYTHONPATH=src:tests:. python3 scripts/release_audit.py .
```

Run the unit test again. It must pass; running `make release-check` against the current pre-cleanup tree must fail for concrete baseline paths.

- [ ] **Step 5: Commit the audit contract**

```bash
git add scripts/release_audit.py tests/test_release_audit.py Makefile
git commit -m "test: define clean release contract"
```

---

### Task 3: Remove concrete baseline integrations while preserving generic extension points

**Files:**
- Delete: concrete directories under `baselines/`
- Create: `baselines/README.md`
- Delete: `src/physbench/baselines/`
- Delete: concrete modules under `src/physbench/baseline_plugins/`
- Delete: WAN-specific drivers under `src/physbench/baseline_runtime/drivers/`
- Modify: `src/physbench/baseline_runtime/drivers/__init__.py`
- Modify: generic baseline tests that currently import WAN fixtures
- Delete: model-specific test modules

**Interfaces:**
- Consumes: `baseline_api`, generic managed runtime, `StandardI2VCLIDriver`, `StandardV2VCLIDriver`, and `SubmissionBaselinePlugin`.
- Produces: zero default bundles while keeping `physbench baseline init/list/inspect/validate` functional.

- [ ] **Step 1: Add a failing zero-default-baselines test**

Add to the generic managed-baseline test suite:

```python
def test_release_tree_has_no_integrated_baselines(self):
    self.assertEqual({}, discover_baseline_bundles(ROOT / "baselines"))
```

Run it and expect failure listing current concrete bundles.

- [ ] **Step 2: Remove every concrete bundle and model-family module**

Delete the six concrete bundle directories and all WAN/Cosmos/Causal-Forcing-only modules. Replace `baselines/` contents with a README documenting `physbench baseline init <name> --backend managed-i2v` and the rule that `baseline.local.json` is local-only.

- [ ] **Step 3: Keep only generic driver exports**

Set `src/physbench/baseline_runtime/drivers/__init__.py` to export exactly:

```python
from .subprocess_i2v import StandardI2VCLIDriver
from .subprocess_v2v import StandardV2VCLIDriver

__all__ = ["StandardI2VCLIDriver", "StandardV2VCLIDriver"]
```

- [ ] **Step 4: Convert mixed tests to temporary generic bundles**

Use `create_baseline_scaffold(..., root=Path(tempdir))` in architecture/task-runtime tests. Delete only assertions whose subject is a concrete model; retain registry, immutable bundle, input-policy, task-compilation, submission, and generic I2V/V2V tests.

- [ ] **Step 5: Run focused generic contract tests**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_managed_baselines tests.test_task_runtime_contracts tests.test_evaluation_variants -v
```

Expected: all generic tests pass, and `physbench baseline list` prints `[]`.

- [ ] **Step 6: Commit the generic-only baseline surface**

```bash
git add -A baselines src/physbench/baselines src/physbench/baseline_plugins src/physbench/baseline_runtime/drivers tests
git commit -m "refactor: remove integrated model baselines"
```

---

### Task 4: Remove experiments and keep runtime outputs under `run/<run_id>`

**Files:**
- Delete: model-specific scripts
- Delete: `tasks/experiments/`
- Delete: `docs/experiments/`
- Delete: `docs/superpowers/`
- Delete: model-specific top-level documentation
- Modify: `.gitignore`
- Modify: `run/README.md`

**Interfaces:**
- Consumes: existing AtomicRun layout.
- Produces: a release checkout with only `run/README.md` tracked under `run/` and no tracked `results/` or `cache/` files.

- [ ] **Step 1: Remove model and experiment material**

Delete WAN training/generation/plot/summarization scripts, all experimental task definitions, experiment logs, internal design/plan records, and model-specific docs. Retain evaluator audit scripts only if their imports and text contain no model-specific dependency or local path.

- [ ] **Step 2: Harden ignore rules without hiding shareable baseline code**

Keep `baselines/<name>` trackable. Add only local-only rules:

```gitignore
baselines/**/baseline.local.json
baselines/**/__pycache__/
baselines/**/*.{bin,ckpt,gguf,onnx,pt,pth,safetensors}
```

Keep `run/*` ignored except `run/README.md`; keep `results/*` and `cache/` ignored.

- [ ] **Step 3: Update the run-layout documentation**

Document that predictions are under `run/<run_id>/predictions/`, evaluation results under `evaluation/`, and logs/artifacts remain run-owned. Remove any suggestion of a separate user `local/` output tree.

- [ ] **Step 4: Run the release audit**

```bash
PATH=/root/miniconda3/bin:$PATH make release-check
```

Expected: no path, model marker, weight, secret, or absolute-path issue.

- [ ] **Step 5: Commit the release tree cleanup**

```bash
git add -A scripts tasks docs .gitignore run
git commit -m "chore: remove experiment and runtime residue"
```

---

### Task 5: Add immutable Dataset download and environment diagnostics

**Files:**
- Create: `src/physbench/dataset_hub.py`
- Modify: `src/physbench/cli.py`
- Create: `tests/test_dataset_hub_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `load_huggingface_dataset_binding(path, dataset_path=...)`.
- Produces: `pull_dataset(binding_path: Path, local_dir: Path) -> Path`, `physbench dataset pull`, and `physbench doctor`.

- [ ] **Step 1: Write failing CLI tests**

Test parser behavior and patch subprocess execution so that the expected download command is exactly equivalent to:

```bash
hf download StevenHedging/VPhysBench \
  --repo-type dataset \
  --revision 04ef16102216a3231f045152ab3a996f5f353f63 \
  --local-dir datasets
```

Also test a missing `hf` executable, a non-commit revision, and an inaccessible dataset error without secret leakage.

- [ ] **Step 2: Run tests and verify failure**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_dataset_hub_cli -v
```

- [ ] **Step 3: Implement `dataset pull` and `doctor`**

`dataset pull` reads `datasets/huggingface.json`, runs the HF CLI with the immutable revision, then calls `load_dataset(..., check_assets=True)`. `doctor` reports Python compatibility, HF CLI availability, binding validity, dataset metadata availability, and whether full assets are present. It must return nonzero only for requirements of the selected `--level metadata|evaluation`.

- [ ] **Step 4: Declare the Hub CLI onboarding dependency**

Add an optional `hub` dependency using the maintained `huggingface_hub` package with CLI support, without adding model inference libraries to the default install.

- [ ] **Step 5: Run focused and existing Dataset tests**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_dataset_hub_cli tests.test_huggingface_dataset_binding tests.test_current_dataset tests.test_single_current_physics_v13 -v
```

- [ ] **Step 6: Commit Dataset onboarding commands**

```bash
git add src/physbench/dataset_hub.py src/physbench/cli.py tests/test_dataset_hub_cli.py pyproject.toml
git commit -m "feat: add immutable dataset setup commands"
```

---

### Task 6: Add an unregistered mock I2V protocol fixture and one-case smoke path

**Files:**
- Create: `examples/dummy_i2v_command.py`
- Create: `scripts/smoke_custom_baseline.py`
- Create: `tests/test_clean_baseline_smoke.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: standard I2V CLI arguments `--prompt`, `--image`, `--output`, `--seed`, and `--job-spec`.
- Produces: a deterministic decodable MP4 for contract testing and a temporary generated bundle; no `baseline.json` is committed under `baselines/`.

- [ ] **Step 1: Write a failing fixture contract test**

Invoke the mock command with a 16×16 fixture image and job spec, then assert the MP4 exists, is decodable, and matches the requested frame count and dimensions. Assert again that `discover_baseline_bundles(ROOT / "baselines") == {}`.

- [ ] **Step 2: Implement the mock command**

Use OpenCV to repeat the provided first frame for the requested number of frames. The script must print a warning that output scores are meaningless and must not contain a baseline descriptor or model dependency.

- [ ] **Step 3: Implement the smoke wrapper**

The wrapper creates a temporary managed-I2V scaffold, rewrites only its generic command to point to the mock script, selects one official case, executes AtomicRun, and verifies prediction/evaluation paths are under `run/<run_id>/` in its temporary root.

- [ ] **Step 4: Add smoke targets**

```make
smoke-interface:
	PYTHONPATH=src:tests:. python3 scripts/smoke_custom_baseline.py --metadata-only
```

Keep the data-backed smoke as a separate target so a fresh clone can run contract tests before downloading 23 GB.

- [ ] **Step 5: Run and commit**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_clean_baseline_smoke -v
git add examples/dummy_i2v_command.py scripts/smoke_custom_baseline.py tests/test_clean_baseline_smoke.py Makefile
git commit -m "test: add unregistered I2V integration smoke"
```

---

### Task 7: Replace internal documentation with the near-release manual set

**Files:**
- Modify: `README.md`
- Create: `docs/GETTING_STARTED.md`
- Create: `docs/CUSTOM_BASELINE_QUICKSTART.md`
- Create: `docs/SUBMISSION_QUICKSTART.md`
- Create: `docs/BENCHMARK_PROTOCOL.md`
- Create: `docs/RUN_LAYOUT.md`
- Create: `docs/REPRODUCIBILITY.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/BASELINE_INTEGRATION.md`
- Modify: `docs/EVALUATION.md`
- Modify or replace: `docs/OPERATIONS.md`
- Modify or replace: `docs/TASKS.md`
- Create: `RELEASE_MANIFEST.json`
- Create: `tests/test_release_documentation.py`

**Interfaces:**
- Consumes: finalized CLI names, directory layout, Dataset binding, official tasks, and evaluator protocol.
- Produces: Chinese overview plus English authoritative onboarding and protocol documentation.

- [ ] **Step 1: Write a documentation consistency test**

Assert that required documents exist, contain no model-family names/local absolute paths, identify five scored and two preview scenes, and reference `baselines/<baseline_id>` plus `run/<run_id>/predictions`.

- [ ] **Step 2: Rewrite the README golden path**

Lead with clone, install, HF login, `dataset pull`, `doctor`, `baseline init`, one-case smoke, and official evaluation. Clearly label Dataset access as private and the mock output as non-scoring.

- [ ] **Step 3: Write focused manuals**

Keep each quickstart independently executable. Put complete schema/driver details in `BASELINE_INTEGRATION.md`; keep the quickstart limited to the first successful run. Remove stale five/six/seven-scene and v4/v6/v13 contradictions.

- [ ] **Step 4: Add the release manifest**

Record branch, the exact source development commit obtained from `git rev-parse feat/wan22-symbol-value-cross-attention`, Dataset repo and exact revision, Dataset ID/release, scored and preview scenes, protocol ID, official task paths, and verification commands. The final release commit remains represented by the Git ref itself because embedding a commit's own hash in that commit is self-referential.

- [ ] **Step 5: Run documentation and portability tests**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest tests.test_release_documentation tests.test_repository_portability -v
```

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs RELEASE_MANIFEST.json tests/test_release_documentation.py
git commit -m "docs: add clean benchmark onboarding manuals"
```

---

### Task 8: Add CI and seal the clean archive

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `RELEASE_MANIFEST.json`

**Interfaces:**
- Consumes: all release checks and focused test suites.
- Produces: a locally verified commit ready to push as `2026-08-08`.

- [ ] **Step 1: Add CPU-only CI**

Run on Python 3.11 and execute installation, Dataset metadata tests, generic baseline contracts, documentation consistency, `make smoke-interface`, and `make release-check`. Do not download full Dataset assets or model weights in CI.

- [ ] **Step 2: Run the full feasible local suite**

```bash
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=src:tests:. python3 -m unittest discover -s tests -v
PATH=/root/miniconda3/bin:$PATH make release-check
git diff --check
```

Classify any evaluator test requiring unavailable optional dependencies separately; no generic release-contract failure may be skipped.

- [ ] **Step 3: Test the tracked archive**

Create a temporary archive with `git archive`, extract it, assert `baselines/` and `run/` contain only their README files, install/import the package, run `physbench baseline list`, metadata tests, and `release_audit.py` from the archive.

- [ ] **Step 4: Seal the manifest and commit**

Verify that the manifest's `source_commit` exactly equals `git rev-parse feat/wan22-symbol-value-cross-attention`. Document `git rev-parse 2026-08-08` as the authoritative way to resolve the final release commit, then commit CI and the sealed manifest:

```bash
git add .github/workflows/ci.yml RELEASE_MANIFEST.json
git commit -m "ci: verify clean near-release archive"
```

- [ ] **Step 5: Verify remote connectivity without publishing**

```bash
git ls-remote origin
git push --dry-run origin HEAD:refs/heads/2026-08-08
```

Expected: SSH authentication succeeds and the dry run reports creation/update of only `2026-08-08`. Actual push is the final publication action after all verification evidence is recorded.
