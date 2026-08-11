# WAN2.2 Symbol-Value Main-Benchmark Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the WAN2.2 symbol-value cross-attention baseline on current main and complete a comparable 2184-step training, inference, and v14 evaluation AtomicRun.

**Architecture:** Restore the already-developed dedicated bundle and its symbol-value adapter/model/runtime modules from repository history, then make the smallest compatibility changes required by current main. Run it only through the current Registry, TaskBuilder, AtomicRun, and official five-scene v14 Task so all identity, artifact, and evaluation contracts remain benchmark-owned.

**Tech Stack:** Python 3, PyTorch, DiffSynth Studio, VPhysBench schema 5 baseline bundles, unittest, eight-GPU Accelerate, scene evaluation protocol v14.

## Global Constraints

- Preserve `baselines/wan22_symbol_value_cross_attention/baseline.local.json` exactly; it is an ignored machine-local deployment override.
- Do not modify the historical `run/wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1` directory.
- Use `datasets/releases/13.0.0/dataset.json` and `tasks/official/five_scene_finetune_eval_csti_identity_v3.json`.
- Use seed 42, rank 32, learning rate `1e-4`, one dataset repeat, eight epochs, 273 steps per epoch, and 2184 total optimizer steps.
- Any code-fingerprint change after a dry-run or failed execution requires a new immutable run ID.
- Do not omit failed or unavailable evaluation cases from coverage reporting.

---

### Task 1: Restore the Portable Bundle and Symbol-Value Runtime

**Files:**
- Create: `baselines/wan22_symbol_value_cross_attention/.gitignore`
- Create: `baselines/wan22_symbol_value_cross_attention/__init__.py`
- Create: `baselines/wan22_symbol_value_cross_attention/adapter.py`
- Create: `baselines/wan22_symbol_value_cross_attention/baseline.json`
- Create: `baselines/wan22_symbol_value_cross_attention/baseline.local.example.json`
- Create: `baselines/wan22_symbol_value_cross_attention/driver.py`
- Create: `baselines/wan22_symbol_value_cross_attention/quantity_registry.json`
- Create: `scripts/train_wan22_symbol_value.sh`
- Create: `scripts/wan22_symbol_value_train.py`
- Create: `scripts/wan22_symbol_value_generate.py`
- Create: `scripts/wan22_symbol_value_generate_batch.py`
- Create: `src/physbench/baseline_plugins/wan22_symbol_value.py`
- Create: `src/physbench/baseline_runtime/drivers/wan22_symbol_value.py`
- Create: `src/physbench/baselines/wan22_symbol_value.py`
- Create: `src/physbench/baselines/wan22_symbol_value_model.py`
- Test: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Consumes: schema-5 `BaselineBundle`, current `DataAdapter`, WAN dataset staging helpers, and DiffSynth WAN pipeline hooks.
- Produces: registry ID `wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1`, Python adapter `create_adapter(bundle)`, managed driver `Driver`, train CLI, and batch generation CLI.

- [ ] **Step 1: Demonstrate the missing registration**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m physbench baseline inspect \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1
```

Expected: FAIL because no portable manifest is present on current main.

- [ ] **Step 2: Restore the reviewed historical implementation**

Restore the exact files listed above from commit `51f8a1f741ec8aac41f48ac7a5acb08b24384d1f`, excluding the ignored `baseline.local.json`. Keep current-main versions of all unrelated files.

- [ ] **Step 3: Align the manifest training contract**

Set the trainer portion of `baseline.json` to:

```json
{
  "rank": 32,
  "learning_rate": 0.0001,
  "dataset_repeat": 1,
  "num_epochs": 8,
  "save_steps": 273,
  "scene_balancing": "oversample_each_scene_to_largest_world_aligned",
  "seed": 42,
  "precision": "bf16",
  "conditioner_precision": "fp32"
}
```

Preserve the remaining optimizer, target-module, checkpoint, and runner fields from the reviewed historical manifest.

- [ ] **Step 4: Run the restored focused regression suite**

Run:

```bash
PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest \
  tests.test_wan22_symbol_value_cross_attention -v
```

Expected: tests either pass or expose specific current-main API incompatibilities for Task 2.

- [ ] **Step 5: Commit the restoration**

```bash
git add baselines/wan22_symbol_value_cross_attention \
  scripts/train_wan22_symbol_value.sh \
  scripts/wan22_symbol_value_train.py \
  scripts/wan22_symbol_value_generate.py \
  scripts/wan22_symbol_value_generate_batch.py \
  src/physbench/baseline_plugins/wan22_symbol_value.py \
  src/physbench/baseline_runtime/drivers/wan22_symbol_value.py \
  src/physbench/baselines/wan22_symbol_value.py \
  src/physbench/baselines/wan22_symbol_value_model.py \
  tests/test_wan22_symbol_value_cross_attention.py
git commit -m "feat: register WAN symbol-value baseline"
```

### Task 2: Adapt and Verify Against Current Main

**Files:**
- Modify: only restored symbol-value files whose tests identify current-main incompatibilities
- Modify: `tests/test_wan22_symbol_value_cross_attention.py`
- Modify: `README.md` if the registry table does not discoverably document the new baseline

**Interfaces:**
- Consumes: the restored bundle and current main's Registry, TaskBuilder, artifact validator, WAN staging, and evaluator contracts.
- Produces: a deployable, fingerprint-complete baseline whose dry-run compiles the official v14 Task without changing shared benchmark semantics.

- [ ] **Step 1: Add current-main registration and contract assertions**

Ensure the focused tests assert:

```python
bundle = registry.get("wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1")
assert bundle.manifest["trainer"]["config"]["num_epochs"] == 8
assert bundle.manifest["trainer"]["config"]["save_steps"] == 273
assert bundle.manifest["input_policy"]["physics"]["representations"] == [
    "symbol_value_cross_attention_v1"
]
```

Also assert the adapter emits auditable symbol/value records for each of the five official scenes and never reads evaluator-only assets.

- [ ] **Step 2: Run tests to identify compatibility failures**

Run:

```bash
PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest \
  tests.test_wan22_symbol_value_cross_attention \
  tests.test_integrated_baselines -v
```

Expected: any failure names an exact interface or fingerprint mismatch.

- [ ] **Step 3: Apply the minimum compatibility fixes**

Update restored imports, adapter output fields, dependency fingerprints, checkpoint sealing, or training/generation arguments only where the failing current-main contract requires it. Do not change Dataset, Task, evaluator, or other baseline behavior.

- [ ] **Step 4: Validate registration and deployment**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m physbench baseline list
PYTHONPATH=src ./.venv/bin/python -m physbench baseline inspect \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1
PYTHONPATH=src ./.venv/bin/python -m physbench baseline validate \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1
PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest \
  tests.test_wan22_symbol_value_cross_attention \
  tests.test_integrated_baselines -v
PYTHONPATH=src ./.venv/bin/python -m compileall -q \
  src tests baselines scripts
git diff --check
```

Expected: all commands exit 0 and registry output includes the baseline ID.

- [ ] **Step 5: Commit compatibility fixes**

```bash
git add README.md baselines/wan22_symbol_value_cross_attention \
  scripts src/physbench tests/test_wan22_symbol_value_cross_attention.py
git commit -m "fix: align symbol-value baseline with current benchmark"
```

### Task 3: Compile and Audit the Official v14 Dry-Run

**Files:**
- Create: `run/wan22_symbol_value_cross_attention_2184_v1_v14_dryrun/`

**Interfaces:**
- Consumes: validated baseline ID, Dataset 13.0.0, and official v14 fine-tune/eval Task.
- Produces: sealed dry-run plan, TaskInstance, training recipe, inference jobs, adaptations, and fingerprints.

- [ ] **Step 1: Build the dry-run**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_csti_identity_v3.json \
  --baseline wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1 \
  --run-id wan22_symbol_value_cross_attention_2184_v1_v14_dryrun \
  --output-root run
```

Expected: run state `frozen` or `planned`, with no model process started.

- [ ] **Step 2: Audit immutable identities and counts**

Read the dry-run JSON and assert:

```text
dataset release = 13.0.0
task_id = five_scene_finetune_eval_v14_csti_identity_v3
evaluation protocol = scene_default_v14
training seed = 42
num_epochs = 8
save_steps = 273
selected scenes = pendulum, collision_1d, inclined_plane_slide,
                  uniform_circular_motion, parabolic_motion
```

Compare canonical training and inference case IDs against the same official Task compiled for a recent registered WAN baseline; differences may only be baseline-owned adaptations and model inputs.

- [ ] **Step 3: Verify artifact paths and dependency fingerprints**

Confirm all writable model outputs resolve inside the dry-run directory and all output-affecting restored source paths appear in component fingerprints.

### Task 4: Execute and Monitor the Full Experiment

**Files:**
- Create: `run/wan22_symbol_value_cross_attention_2184_v1_v14/`

**Interfaces:**
- Consumes: the audited portable baseline, local deployment override, Dataset 13.0.0, and official v14 Task.
- Produces: 2184-step checkpoints, all inference videos, artifact records, case results, Task result, and terminal AtomicRun state.

- [ ] **Step 1: Start the immutable execution**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval_csti_identity_v3.json \
  --baseline wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1 \
  --run-id wan22_symbol_value_cross_attention_2184_v1_v14 \
  --output-root run \
  --execute
```

- [ ] **Step 2: Monitor training to the sealed checkpoint**

Track `state.json`, run-local training logs, GPU processes, and checkpoint audits. Success requires optimizer step 2184 plus sealed LoRA, symbol-value conditioner, and optimizer-state artifacts.

- [ ] **Step 3: Monitor inference coverage**

Track run-local jobs and prediction records until every official inference job is terminal. Retry only through the driver's recoverable mechanisms; if code or frozen identity changes, stop using this run ID and create a versioned successor.

- [ ] **Step 4: Monitor canonical v14 evaluation**

Confirm evaluation uses the run-frozen `scene_default_v14` protocol and produces `evaluation/case_results.jsonl` plus `evaluation/task_result.json`. Preserve unavailable cases and strict coverage semantics.

### Task 5: Final Verification and Handoff

**Files:**
- Read: `run/wan22_symbol_value_cross_attention_2184_v1_v14/state.json`
- Read: `run/wan22_symbol_value_cross_attention_2184_v1_v14/artifacts/`
- Read: `run/wan22_symbol_value_cross_attention_2184_v1_v14/predictions.jsonl`
- Read: `run/wan22_symbol_value_cross_attention_2184_v1_v14/evaluation/task_result.json`

**Interfaces:**
- Consumes: all registration and experiment artifacts.
- Produces: evidence-backed completion report with exact run ID, commit IDs, training step, prediction coverage, evaluation coverage, scores, and any unavailable-case reasons.

- [ ] **Step 1: Run final code verification**

```bash
PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest \
  tests.test_wan22_symbol_value_cross_attention \
  tests.test_integrated_baselines -v
PYTHONPATH=src ./.venv/bin/python -m physbench baseline validate \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1
git diff --check
git status --short --branch
```

- [ ] **Step 2: Verify run completeness from fresh evidence**

Require terminal `state.stage == "complete"`, checkpoint step 2184, expected prediction count equal to recorded prediction count, v14 evaluation manifest, and explicit Task coverage. Do not claim a complete score when strict coverage leaves the Task score null.

- [ ] **Step 3: Report the outcome**

Provide clickable paths to the registered manifest, run state, checkpoint audit, predictions, and Task result. State separately whether registration, training, inference, and evaluation completed.
