# WAN2.2 Subject-Motion Flow Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register, train, and canonically evaluate a warm-start WAN2.2 baseline that combines whole-tube IoU with area-normalized subject Flow Matching and subject-supported latent temporal-difference losses.

**Architecture:** A new managed Baseline subclasses the existing Tube-IoU data/checkpoint adapter but owns its registration, cache seeding, pure loss functions, trainer, and launcher. One normal Wan Flow Matching forward produces the stock loss, the inherited latent occupancy loss, a directly masked velocity loss, and a low-noise latent temporal-difference loss; inference exports and consumes only a standard LoRA checkpoint.

**Tech Stack:** Python 3.10, PyTorch, DiffSynth-Studio, Accelerate, PEFT LoRA, SAM2.1, NumPy, safetensors, VPhysBench managed runtime, unittest, four NVIDIA GPUs.

## Global Constraints

- Use only GPUs `0,1,2,3` for every training, preflight, and inference command.
- Do not change existing Baseline descriptors or their runtime behavior.
- Keep all masks and occupancy heads training-only; inference jobs may contain only prompt, first frame, seed, generation config, and LoRA checkpoint.
- Preserve the audited Wan formulas `x_sigma=(1-sigma)*x0+sigma*epsilon`, `v*=epsilon-x0`, and `x0_hat=x_sigma-sigma*v_hat`.
- Preserve TI2V frame-zero conditioning and exclude that slice from all velocity losses.
- Warm start requires the exact paired final Tube-IoU LoRA and occupancy head.
- Use a new run directory for every preflight and canonical experiment.
- Canonical evaluation must use Dataset 13.0.0 and the synchronized split-scene Task `six_scene_train_five_scene_eval_v14` without evaluator modifications.

---

### Task 1: Pure Subject and Motion Losses

**Files:**
- Create: `src/physbench/baselines/wan22_subject_motion_model.py`
- Create: `tests/test_wan22_subject_motion.py`

**Interfaces:**
- Consumes: `torch.Tensor` velocity/latent tensors in `BCTHW`, binary aligned masks in `BTHW`, scheduler/sample weights in `B`.
- Produces: `masked_subject_flow_loss(prediction, target, subject_mask, sample_weight, eps) -> MaskedLossResult`, `subject_temporal_difference_loss(clean_estimate, clean_target, subject_mask, sample_weight, eps) -> MaskedLossResult`, and `MaskedLossResult(loss, per_sample_loss, support_fraction)`.

- [ ] **Step 1: Write failing tests for foreground normalization**

  Add tests showing background error contributes zero, equal foreground errors produce equal per-sample losses despite different mask areas, scheduler weights apply before the batch mean, and an empty aligned mask raises `ValueError`.

- [ ] **Step 2: Run the focused tests and confirm RED**

  Run:

  ```bash
  PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion.SubjectFlowLossTests -v
  ```

  Expected: import failure because `wan22_subject_motion_model` does not exist.

- [ ] **Step 3: Implement `masked_subject_flow_loss`**

  Compute channel-mean squared velocity error, multiply by the tail mask, divide each sample by its mask volume plus `1e-6`, multiply per-sample scheduler weights, and batch-mean. Validate shapes, ranges, finiteness, positive epsilon, and non-empty support.

- [ ] **Step 4: Run the subject-loss tests and confirm GREEN**

  Run the command from Step 2; expected all tests pass.

- [ ] **Step 5: Write failing temporal-difference tests**

  Add tests showing identical temporal differences give the Charbonnier floor, an error outside endpoint-union support is ignored, a shifted foreground change is penalized, low-noise sample weights apply before the batch mean, gradients reach `clean_estimate`, and fewer than two latent frames fails.

- [ ] **Step 6: Implement `subject_temporal_difference_loss`**

  Form consecutive latent differences, use `maximum(mask[:,1:], mask[:,:-1])` as support, compute channel-mean `sqrt(delta_error**2 + eps)`, normalize per sample by support volume, and weight before the batch mean.

- [ ] **Step 7: Run all pure-loss tests and commit**

  ```bash
  PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion -v
  git add src/physbench/baselines/wan22_subject_motion_model.py \
    tests/test_wan22_subject_motion.py
  git commit -m "feat: add WAN subject-motion losses"
  ```

---

### Task 2: Combined Wan Training Objective

**Files:**
- Create: `scripts/wan22_subject_motion_train.py`
- Modify: `tests/test_wan22_subject_motion.py`

**Interfaces:**
- Consumes: DiffSynth Wan pipeline inputs, a pixel-space subject tube, the inherited `LatentOccupancyHead`, and config keys `lambda_st`, `lambda_subject_flow`, `lambda_motion_delta`, and `aux_warmup_steps`.
- Produces: `compute_subject_motion_objective(...) -> dict[str, torch.Tensor]` with base, Tube-IoU, subject Flow, motion-delta, coefficient, contribution, foreground, sigma, and total metrics; `SubjectMotionWanTrainingModule`; `launch_subject_motion_training`.

- [ ] **Step 1: Write a failing exact-objective test**

  Use the existing fake Wan scheduler/pipe test fixtures to verify one independently sampled timestep per sample, exact stock base loss when all new lambdas are zero, exact weighted total when fixed tensors are supplied, known TI2V first-latent insertion, and gradients reaching both LoRA-surrogate prediction and occupancy head.

- [ ] **Step 2: Run the objective test and confirm RED**

  ```bash
  PYTHONPATH=src:tests:.:/root/Steven/wan22_pair_text_runtime/vendor/DiffSynth-Studio \
    /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion.SubjectMotionObjectiveTests -v
  ```

- [ ] **Step 3: Implement the combined objective**

  Copy no scheduler assumptions: call the same `_scheduler_batch_values`, `flowmatch_clean_estimate`, `align_mask_tube`, `SpatioTemporalTubeIoULoss`, and `noise_weight` helpers used by the audited Tube-IoU trainer. Use the aligned tail mask for subject Flow and the full aligned mask for temporal differences. Apply the Wan scheduler weight to subject Flow, `1-sigma` to motion delta, and linear optimizer-step warmup to all auxiliary coefficients.

- [ ] **Step 4: Implement the module and metrics-aware training loop**

  Subclass `STTubeIoUWanTrainingModule`, override `forward`, retain the separate-head logger/checkpoint code, and extend synchronized JSONL logging with `subject_flow_loss`, `motion_delta_loss`, `lambda_subject_effective`, `lambda_motion_effective`, `subject_flow_contribution`, `motion_delta_contribution`, and support fractions. Preserve fatal finite-loss and finite-gradient audits.

- [ ] **Step 5: Run objective and existing Tube-IoU tests**

  ```bash
  PYTHONPATH=src:tests:.:/root/Steven/wan22_pair_text_runtime/vendor/DiffSynth-Studio \
    /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion tests.test_wan22_st_tube_iou -v
  ```

  Expected: every test passes and the original baseline's disabled/enabled contracts remain unchanged.

- [ ] **Step 6: Commit the trainer**

  ```bash
  git add scripts/wan22_subject_motion_train.py tests/test_wan22_subject_motion.py
  git commit -m "feat: train WAN with subject-motion objectives"
  ```

---

### Task 3: Adapter, Warm Start, and Cache Seeding

**Files:**
- Create: `src/physbench/baselines/wan22_subject_motion.py`
- Create: `scripts/train_wan22_subject_motion.sh`
- Modify: `tests/test_wan22_subject_motion.py`

**Interfaces:**
- Consumes: `runtime.warm_start_checkpoint`, `runtime.subject_tube_cache_dir`, the inherited Tube-IoU adapter, and `ST_TUBE_IOU_CONFIG_JSON` containing all combined-loss settings.
- Produces: `Wan22SubjectMotionAdapter`, validated run-local hard links/copies for cached `.npz`/`.audit.json` pairs, a subject-motion launcher command, and a standard LoRA plus separate occupancy-head checkpoint manifest.

- [ ] **Step 1: Write failing adapter contract tests**

  Verify a valid cache pair is seeded, wrong `case_id`/hash/binary dtype is rejected, missing cache falls back instead of fabricating data, warm start requires `.safetensors` plus `.st-head.safetensors`, launcher environment is exactly `CUDA_VISIBLE_DEVICES=0,1,2,3`, and inference checkpoint selection excludes `.st-head.safetensors`.

- [ ] **Step 2: Implement fail-closed cache validation and seeding**

  Validate each source NPZ against its audit and use `os.link` with `shutil.copy2` fallback. Seed only requested training case IDs into the new run's `artifacts/wan22/dataset/masks`; never mutate the source cache.

- [ ] **Step 3: Implement warm-start and launcher binding**

  Resolve the local warm-start path, validate the paired head, expose it as `initial_lora_checkpoint`, and launch `scripts/train_wan22_subject_motion.sh`. The shell script must validate the four-GPU string, model assets, paired initialization, sealed rank/target modules, and call the new Python trainer through frozen Accelerate configuration.

- [ ] **Step 4: Run adapter tests and commit**

  ```bash
  PYTHONPATH=src:tests:.:/root/Steven/wan22_pair_text_runtime/vendor/DiffSynth-Studio \
    /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion -v
  git add src/physbench/baselines/wan22_subject_motion.py \
    scripts/train_wan22_subject_motion.sh tests/test_wan22_subject_motion.py
  git commit -m "feat: orchestrate WAN subject-motion training"
  ```

---

### Task 4: Managed Baseline Registration

**Files:**
- Create: `baselines/wan22_subject_motion/__init__.py`
- Create: `baselines/wan22_subject_motion/.gitignore`
- Create: `baselines/wan22_subject_motion/baseline.json`
- Create: `baselines/wan22_subject_motion/baseline.local.example.json`
- Create locally/ignored: `baselines/wan22_subject_motion/baseline.local.json`
- Create locally/ignored: `baselines/wan22_subject_motion/accelerate.local.yaml`
- Create: `baselines/wan22_subject_motion/driver.py`
- Create: `src/physbench/baseline_plugins/wan22_subject_motion.py`
- Create: `src/physbench/baseline_runtime/drivers/wan22_subject_motion.py`
- Modify: `tests/test_wan22_subject_motion.py`

**Interfaces:**
- Consumes: managed Baseline schema 5, `Wan22ExecutionEngine`, and `Wan22SubjectMotionAdapter`.
- Produces: discoverable Baseline `wan22_ti2v_5b_lora_r32_subject_motion_v1` with two-epoch warm-start trainer config and stock Wan inference config.

- [ ] **Step 1: Write failing discovery and isolation tests**

  Assert identity/version, seven scenes, finetune/I2V capabilities, 48-channel head, exact loss weights, two epochs, four-GPU global batch contract, dependency fingerprint coverage, local deployment exclusion from the portable digest, mask-free jobs, and stock-LoRA-only inference.

- [ ] **Step 2: Add descriptor, driver, plugin, and managed driver**

  Bind `lambda_st=0.05`, `lambda_subject_flow=0.10`, `lambda_motion_delta=0.05`, `aux_warmup_steps=100`, two epochs, save every 234 steps, rank 32, and the existing scene/media profiles. The plugin maps every new config key into the legacy adapter config and injects the validated warm-start deployment path.

- [ ] **Step 3: Add the local four-GPU deployment**

  Point to `/root/Steven/wan22_pair_text_runtime`, the existing Python/model paths, GPUs `0,1,2,3`, a copied/frozen-compatible Accelerate YAML, the parent final LoRA, and the parent run's validated mask directory. Keep local files ignored.

- [ ] **Step 4: Validate registration and run the complete focused suite**

  ```bash
  PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench \
    baseline validate wan22_ti2v_5b_lora_r32_subject_motion_v1
  PYTHONPATH=src:tests:.:/root/Steven/wan22_pair_text_runtime/vendor/DiffSynth-Studio \
    /root/Steven/.venvs/wan22-pair-text/bin/python -m unittest \
    tests.test_wan22_subject_motion tests.test_wan22_st_tube_iou \
    tests.test_integrated_baselines.IntegratedBaselineTests.test_all_integrated_bundles_use_managed_runtime -v
  ```

- [ ] **Step 5: Commit registration**

  ```bash
  git add baselines/wan22_subject_motion \
    src/physbench/baseline_plugins/wan22_subject_motion.py \
    src/physbench/baseline_runtime/drivers/wan22_subject_motion.py \
    tests/test_wan22_subject_motion.py
  git commit -m "feat: register WAN subject-motion baseline"
  ```

---

### Task 5: Real Four-GPU Preflight and Parameter Gate

**Files:**
- Runtime output: `run/wan22_subject_motion_v1_preflight/`
- Runtime output: `run/wan22_subject_motion_v1_preflight_disabled/`

**Interfaces:**
- Consumes: registered Baseline, cached tubes, parent paired checkpoint, GPUs 0--3.
- Produces: three synchronized optimizer steps per arm, real loss/gradient/memory metrics, and an auditable coefficient decision.

- [ ] **Step 1: Stage minimal preflight runs without touching canonical artifacts**

  Freeze a small world-aligned training subset covering all six selected training scenes. Run one disabled arm with all added coefficients zero and one enabled arm with proposed coefficients, identical initialization/data/order/seed, on GPUs 0--3.

- [ ] **Step 2: Verify numerical and systems contracts**

  Require three metric rows, finite total/base/auxiliary losses, non-zero LoRA/head gradients when required, identical disabled total/base loss, paired checkpoint export, no worker on GPUs 4--7, and less than 35 GiB peak allocation per GPU.

- [ ] **Step 3: Apply the coefficient gate**

  Compare each enabled weighted contribution against `0.5 * base_loss`. If either exceeds the bound, reduce only that coefficient in the new Baseline descriptor and rerun enabled preflight. Record the final coefficient decision in `artifacts/wan22/preflight_decision.json`.

---

### Task 6: Two-Epoch Training and Stock Inference

**Files:**
- Runtime output: `run/wan22_subject_motion_v1_v14/`

**Interfaces:**
- Consumes: frozen Dataset 13.0.0, v14 task, validated Baseline, GPUs 0--3, parent paired checkpoint.
- Produces: 468-step LoRA/head checkpoints, 76 staged jobs, and 76 run-local videos.

- [ ] **Step 1: Launch the canonical atomic run**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
    /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench atomic-run \
    --dataset datasets/releases/13.0.0/dataset.json \
    --task tasks/experiments/six_scene_train_five_scene_eval_v14.json \
    --baseline wan22_ti2v_5b_lora_r32_subject_motion_v1 \
    --run-id wan22_subject_motion_v1_v14 --output-root run --execute
  ```

- [ ] **Step 2: Monitor training without mutating the run**

  Check synchronized metric count, expected 234/468 checkpoints, finite
  gradients, GPU ownership, and progress at least every 60 seconds. Diagnose
  failures with `superpowers:systematic-debugging` before any fix.

- [ ] **Step 3: Validate final training artifacts**

  Require exactly 468 ordered metric rows, final LoRA and paired head, matching SHA-256 manifest, complete optimizer/scheduler/RNG state, final coefficient metrics, and no inference dependency on the head.

- [ ] **Step 4: Complete four-worker stock inference**

  Require 76 unique jobs, four successful persistent workers bound only to GPUs 0--3, 76 complete videos, valid media contracts, and run-local artifact hashes.

---

### Task 7: Canonical Evaluation, Comparison, and Next Iteration

**Files:**
- Runtime output: `run/wan22_subject_motion_v1_v14/evaluation/`
- Create: `docs/experiments/wan22-subject-motion-v1-results.md`

**Interfaces:**
- Consumes: 76 predictions, v14 protocol, parent Tube-IoU summary/metrics, new training metrics.
- Produces: canonical case results/summary, direct comparison, failure analysis, and the selected next Baseline hypothesis.

- [ ] **Step 1: Complete canonical evaluation**

  Require 76 case result documents, zero evaluator/protocol integrity errors, explicit unsupported/unavailable counts, and an unmodified evaluation protocol fingerprint.

- [ ] **Step 2: Compare against the Tube-IoU parent**

  Report CSTI observed macro and per-scene deltas, physical-score deltas, coverage/status changes, generation failures, last-100 training proxies, step time, and peak memory. Label partial observed scores separately from official strict-coverage scores.

- [ ] **Step 3: Write the reflection and choose the next loss**

  Use the decision rules in the design spec. Document which evidence supports or contradicts foreground dilution, occupancy co-adaptation, and temporal-motion hypotheses; then register the next independent WAN2.2 Baseline if at least 90 minutes remain in the experiment window.

- [ ] **Step 4: Run fresh verification and commit the report**

  Use `superpowers:verification-before-completion`, rerun focused unit/integration tests, Baseline validation, artifact/hash checks, `git diff --check`, and `git status --short`. Commit only source/docs, never run artifacts.
