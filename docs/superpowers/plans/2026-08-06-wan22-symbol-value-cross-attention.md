# WAN2.2 Symbol–Value Cross-Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute a WAN2.2 baseline that turns structured symbol/value annotations into word-space tokens and cross-attends the original UMT5 text context to them.

**Architecture:** A strict bundle-local adapter emits original text plus SI-normalized symbol/value records. A trainable FP32 `SymbolValueConditioner` combines frozen UMT5 token embeddings with quantity features and injects them through 512-dimensional bottleneck cross-attention into frozen UMT5 context; WAN DiT LoRA and the conditioner train jointly and serialize into one authenticated checkpoint.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, DiffSynth-Studio WAN2.2-TI2V-5B, safetensors, VPhysBench Baseline Bundle v5 and AtomicRun.

## Global Constraints

- Dataset is `physics_video_seven_scene_v13`, release `13.0.0`, View A.
- Baseline does not mutate existing baseline identities or Dataset assets.
- Model prompt remains the normalized original caption and contains no appended numeric literals.
- Symbol vectors come from frozen UMT5 token embeddings; UMT5 remains frozen.
- Value vectors contain SI magnitude, seven SI dimensions, and exact unit identity.
- Fusion is additive in 4096 dimensions followed by explicit text-to-physics cross-attention.
- Training seed and inference seed are both 42; LoRA rank is exactly 32.
- Use all eight GPUs, bf16 WAN computation, FP32 conditioner, one dataset repeat, and two epochs.
- Generate every selected test case; record unsupported evaluator status without inventing scores.

---

### Task 1: Structured symbol/value adapter

**Files:**
- Create: `baselines/wan22_symbol_value_cross_attention/__init__.py`
- Create: `baselines/wan22_symbol_value_cross_attention/adapter.py`
- Create: `baselines/wan22_symbol_value_cross_attention/quantity_registry.json`
- Test: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Consumes: Bundle v5, `StandardDataAdapter`, Dataset 13 conditionable cases.
- Produces: `SymbolValueRegistry.render(case, prompt) -> (records, used)` and `SymbolValueCrossAttentionDataAdapter.adapt_case(case, role) -> adaptation` with representation `symbol_value_cross_attention_v1`.

- [ ] **Step 1: Write failing registry and adapter tests**

  Add literal fixtures asserting that a scalar `{symbol: "x_0", value: 0.1,
  unit: "m"}` becomes a record with `si_value=0.1`, dimension
  `[1,0,0,0,0,0,0]`, a stable integer `unit_id`, and an unchanged prompt.
  Add separate failures for an absent symbol, unexpected unit, temporal
  samples, non-finite value, and missing required field.

- [ ] **Step 2: Verify the tests fail for the missing bundle**

  Run: `PYTHONPATH=src:tests:. python -m unittest tests.test_wan22_symbol_value_cross_attention.SymbolValueRegistryTests -v`

  Expected: import failure because the new adapter does not exist.

- [ ] **Step 3: Implement the registry and adapter**

  Copy only the SI unit/scene selection data from the v2 quantity registry,
  assign unit IDs by a committed explicit `unit_ids` mapping, validate symbols
  against the caption, and emit inline JSON records without sentinels.

- [ ] **Step 4: Verify adapter behavior passes**

  Run the command from Step 2 and require zero failures.

- [ ] **Step 5: Commit the adapter slice**

  ```bash
  git add baselines/wan22_symbol_value_cross_attention tests/test_wan22_symbol_value_cross_attention.py
  git commit -m "feat: add symbol-value conditioning adapter"
  ```

### Task 2: Word-space conditioner and prompt injection

**Files:**
- Create: `src/physbench/baselines/wan22_symbol_value_model.py`
- Modify: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Consumes: `records: list[dict]`, UMT5 tokenizer, frozen UMT5 token embedding, UMT5 context and mask.
- Produces: `numeric_features(float)`, `SymbolValueConditioner(config)`, `encode_symbol_value_prompt(pipe, prompt, records)`, `install_symbol_value_prompt_unit(pipe, conditioner)`, and `symbol_value_inference_conditioning(pipe, records)`.

- [ ] **Step 1: Write failing model behavior tests**

  Assert the eight hand-derived numeric features for `0`, `10`, and `-0.1`;
  assert subword means from a literal fake embedding table; assert
  `LayerNorm(symbol + value)` returns one 4096-D token per record; assert the
  cross-attention changes valid text positions but leaves padded positions
  exactly zero; assert the negative CFG branch never receives records.

- [ ] **Step 2: Verify RED**

  Run: `PYTHONPATH=src:tests:. python -m unittest tests.test_wan22_symbol_value_cross_attention.SymbolValueModelTests -v`

  Expected: import failure for `wan22_symbol_value_model`.

- [ ] **Step 3: Implement the minimal conditioner**

  Implement magnitude, dimension, and unit branches; frozen symbol pooling;
  additive LayerNorm fusion; 512-D eight-head attention; residual gate; prompt
  unit replacement; explicit positive/negative CFG bindings; finite/shape
  validation; and per-record attention/token audits.

- [ ] **Step 4: Verify GREEN and refactor**

  Run the Step 2 test plus `tests.test_wan22_quantity_embedding` to ensure the
  old baseline remains unchanged.

- [ ] **Step 5: Commit the model slice**

  ```bash
  git add src/physbench/baselines/wan22_symbol_value_model.py tests/test_wan22_symbol_value_cross_attention.py
  git commit -m "feat: add symbol-value cross-attention conditioner"
  ```

### Task 3: Strict combined checkpoints

**Files:**
- Modify: `src/physbench/baselines/wan22_symbol_value_model.py`
- Modify: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Produces: `load_symbol_value_conditioner_checkpoint`, `load_combined_symbol_value_checkpoint`, `load_verified_combined_symbol_value_checkpoint`, `verify_symbol_value_checkpoint_manifest`, and `symbol_value_pipeline_shared_fingerprint`.

- [ ] **Step 1: Write failing checkpoint tests**

  Build small real conditioner states and assert rejection of missing,
  unexpected, wrong-shape, non-finite, mixed-prefix, and hash-mismatched
  tensors. Assert validation occurs before any `load_state_dict` or LoRA fusion.

- [ ] **Step 2: Verify RED**

  Run the checkpoint test class and require failures caused by absent helpers.

- [ ] **Step 3: Implement strict checkpoint separation**

  Use prefix `pipe.symbol_value_conditioner.`, derive the exact expected state
  keys/shapes from the configured conditioner, reuse the frozen WAN LoRA target
  validator, and authenticate the same bytes that safetensors parses.

- [ ] **Step 4: Verify GREEN**

  Run all tests in `tests.test_wan22_symbol_value_cross_attention`.

- [ ] **Step 5: Commit checkpoint support**

  ```bash
  git add src/physbench/baselines/wan22_symbol_value_model.py tests/test_wan22_symbol_value_cross_attention.py
  git commit -m "feat: seal symbol-value combined checkpoints"
  ```

### Task 4: Managed baseline, training, and generation

**Files:**
- Create: `baselines/wan22_symbol_value_cross_attention/baseline.json`
- Create: `baselines/wan22_symbol_value_cross_attention/baseline.local.example.json`
- Create: `baselines/wan22_symbol_value_cross_attention/baseline.local.json`
- Create: `baselines/wan22_symbol_value_cross_attention/driver.py`
- Create: `src/physbench/baseline_runtime/drivers/wan22_symbol_value.py`
- Create: `scripts/wan22_symbol_value_train.py`
- Create: `scripts/train_wan22_symbol_value.sh`
- Create: `scripts/wan22_symbol_value_generate.py`
- Create: `scripts/wan22_symbol_value_generate_batch.py`
- Modify: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Consumes: AtomicRun training plan, adapted metadata JSONL, eight-GPU Accelerate config, combined checkpoint manifest, inference job JSON.
- Produces: managed driver lifecycle, train command/environment, checkpoint audits, persistent batch generation, prediction/token audit records.

- [ ] **Step 1: Write failing integration tests**

  Assert baseline discovery/validation, exact trainer hyperparameters, generated
  metadata fields, runtime environment sealing, command construction,
  checkpoint manifest validation, job payload conditioning, and a fake-pipeline
  generation smoke path.

- [ ] **Step 2: Verify RED**

  Run the new integration test classes; expect missing manifest/driver/scripts.

- [ ] **Step 3: Implement bundle and managed driver**

  Subclass the existing WAN quantity driver only where media orchestration is
  identical, override identity/config/checkpoint/generation boundaries, and
  fingerprint every new implementation file.

- [ ] **Step 4: Implement training and generation entrypoints**

  Adapt the proven quantity scripts to train `pipe.symbol_value_conditioner`,
  audit non-zero finite conditioner gradients, save recovery state, load one
  authenticated combined checkpoint, and bind physics records only to positive
  CFG.

- [ ] **Step 5: Verify GREEN**

  Run the complete new test module, managed-baseline tests, conditioning
  contract tests, integrated-baseline tests, and shell syntax checks.

- [ ] **Step 6: Commit runtime integration**

  ```bash
  git add baselines/wan22_symbol_value_cross_attention src/physbench/baseline_runtime/drivers/wan22_symbol_value.py scripts/wan22_symbol_value_* scripts/train_wan22_symbol_value.sh tests/test_wan22_symbol_value_cross_attention.py
  git commit -m "feat: integrate WAN symbol-value baseline"
  ```

### Task 5: Seven-scene task and honest unsupported evaluation

**Files:**
- Create: `tasks/experiments/seven_scene_symbol_value_finetune_eval.json`
- Create: `configs/evaluation/protocols/scene_default_v10_seven_scene_unscored.json`
- Create: `docs/WAN22_SYMBOL_VALUE_CROSS_ATTENTION.md`
- Modify: `README.md`
- Modify: `docs/WAN22.md`
- Modify: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Produces: one seven-scene finetune task selecting View A train/ID test and an evaluation protocol with explicit `unsupported` entries for unimplemented evaluators.

- [ ] **Step 1: Write failing task compilation tests**

  Compile the task against Dataset 13 and assert train/test counts
  `{pendulum:80/20, collision:310/20, incline:80/15, circular:30/6,
  parabolic:82/15, push:127/14, spring:97/20}`, disjoint splits, 110 jobs, and
  explicit unsupported evaluator identities where required.

- [ ] **Step 2: Verify RED**

  Run the task test and expect missing task/protocol files.

- [ ] **Step 3: Add task, protocol, and operator documentation**

  Copy the five supported scene configs byte-for-byte from v10, add unsupported
  entries for push/spring, describe exact commands and artifact semantics, and
  list the baseline without changing official ranking tasks.

- [ ] **Step 4: Verify GREEN and commit**

  Run task compilation, baseline validation, and dataset validation, then
  commit the task/config/docs/test slice.

### Task 6: Smoke, full training, generation, and evaluation

**Files:**
- Create at runtime: `run/wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1/`
- Create: `docs/experiments/WAN22_SYMBOL_VALUE_CROSS_ATTENTION_20260806.md`

**Interfaces:**
- Consumes: completed bundle, Dataset 13, local WAN model, eight GPUs.
- Produces: complete AtomicRun artifacts and evidence-backed experiment report.

- [ ] **Step 1: Establish clean software baseline**

  Run `make test`, the new module tests, baseline validation, and task-build.
  Record any pre-existing unrelated failure before execution.

- [ ] **Step 2: Run one-step eight-GPU training smoke**

  Compile a smoke run with one balanced batch per rank, execute it, and verify
  finite loss, positive conditioner gradients, exact 600 LoRA tensors plus the
  configured conditioner state, and checkpoint manifest hash.

- [ ] **Step 3: Run one-case generation smoke**

  Load the smoke checkpoint through the authenticated boundary, generate one
  121-frame video, and verify media contract plus symbol/value attention audit.

- [ ] **Step 4: Execute full AtomicRun**

  Run the seven-scene task with `--execute`, keep all logs and state transitions,
  and monitor GPU utilization, loss, checkpoint creation, and prediction count.

- [ ] **Step 5: Evaluate and audit artifacts**

  Score supported scenes, retain explicit unsupported case results for spring
  and any other unimplemented evaluator, reconcile all 110 predictions with
  jobs, and verify every checkpoint/prediction artifact digest.

- [ ] **Step 6: Run final verification**

  Run `make full-test`, new targeted tests, `baseline validate`, task-build,
  AtomicRun artifact audits, `git diff --check`, and `git status --short`.

- [ ] **Step 7: Write and commit the experiment report**

  Report exact run identity, commands, hyperparameters, wall time, hardware,
  training curves, supported-scene metrics, unsupported statuses, limitations,
  and paths to primary artifacts without extrapolating unmeasured results.
