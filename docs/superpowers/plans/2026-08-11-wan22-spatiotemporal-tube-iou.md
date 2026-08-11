# WAN2.2 Spatio-Temporal Tube IoU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, register, train, infer, and evaluate a WAN2.2-TI2V-5B LoRA baseline with a differentiable whole-video latent occupancy tube-IoU auxiliary loss.

**Architecture:** Keep shared framework and existing baselines unchanged. A new WAN adapter derives training-only full pseudo-GT tubes from canonical reference videos and sibling first-frame masks, while a custom DiffSynth trainer predicts occupancy from the one-step Flow Matching clean-latent estimate and exports a stock-compatible LoRA separately from its auxiliary head.

**Tech Stack:** Python 3.10, PyTorch, DiffSynth-Studio, Accelerate, PEFT LoRA, SAM2.1, OpenCV, NumPy, safetensors, unittest, VPhysBench managed baseline runtime.

## Global Constraints

- Work directly in `/root/Steven/VPhysBench` on the main worktree.
- Do not modify shared compiler/input contracts or existing baseline behavior.
- Preserve existing user files and historical run directories.
- Use CUDA devices `0,1,2,3` for training and inference.
- Use one random Flow Matching timestep per sample and no training sampler loop.
- Compute one soft IoU over the complete `T*H*W` tube per sample, then average the batch.
- Keep WAN TI2V conditioning, VAE temporal compression, and DiT routing unchanged.
- Keep the inference checkpoint compatible with the existing stock WAN generator.
- Never place mask paths in adaptations, native model inputs, or inference jobs.

---

### Task 1: Independent Tube-IoU and Latent Occupancy Modules

**Files:**
- Create: `src/physbench/baselines/wan22_st_tube_iou_model.py`
- Create: `tests/test_wan22_st_tube_iou.py`

**Interfaces:**
- Produces: `SpatioTemporalTubeIoULoss`, `TubeIoUResult`, `flowmatch_clean_estimate`, `align_mask_tube`, `noise_weight`, and `LatentOccupancyHead`.

- [ ] **Step 1: Write failing mathematical behavior tests**

Cover identical, disjoint, partial-overlap, oversized prediction,
disappearance, delayed appearance, one-frame temporal shift, both-empty,
target-only-empty, and prediction-only-empty tubes. Assert that all temporal
and spatial voxels are flattened before one per-sample IoU is computed.

- [ ] **Step 2: Write failing reconstruction, alignment, and gradient tests**

Assert `x_s - sigma*v == x0` for exact target velocity; first-frame replacement
is exact; pixel lengths 5, 9 and 121 map to latent lengths 2, 3 and 31; an
off-by-one expected length raises; nearest-neighbor targets stay binary; and
backward produces nonzero gradients in latent input and head weights.

- [ ] **Step 3: Run tests and observe missing-module failure**

Run: `PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest tests.test_wan22_st_tube_iou -v`

- [ ] **Step 4: Implement minimal modules**

Use float32 reductions, define both-empty score as one, implement exact
batch-broadcasting for sigma, and use
`Conv3d(48,32,3,padding=1) -> SiLU -> Conv3d(32,1,1)` for occupancy logits;
48 is derived from the actual TI2V-5B VAE checkpoint's 96-channel posterior
statistics and 48-channel decoder input.

- [ ] **Step 5: Run tests and commit**

Run the focused unittest command, then commit the model module and tests with
message `feat: add spatiotemporal tube IoU loss`.

### Task 2: Auditable Full-Tube Materialization

**Files:**
- Create: `src/physbench/baselines/wan22_st_tube_iou_masks.py`
- Modify: `tests/test_wan22_st_tube_iou.py`

**Interfaces:**
- Consumes: an authorized training reference-video path, its canonical sibling mask manifest, and `Wan22MediaAdapter` time/spatial policy.
- Produces: `materialize_subject_mask_tube(...) -> MaskTubeAudit` and a compressed binary `THW` NPZ.

- [ ] **Step 1: Write failing fixture tests**

Build a synthetic moving-square video and two seed masks. With a stub segmenter,
assert instance union, timestamp resampling, aspect-preserving nearest-neighbor
geometry, exact output frame count, binary dtype, case-ID validation,
deterministic cache reuse, and path containment.

- [ ] **Step 2: Run tests and observe missing implementation**

Run the `MaskTubeMaterializerTests` class from the focused test module.

- [ ] **Step 3: Implement direct-tube and SAM2 sources**

For current `1HW` assets, decode the canonical video, build prompts from each
mask bbox and interior centroid, invoke the repository's frozen SAM2 adapter,
union propagated instances, and transform with the exact WAN time/contain-pad
mapping. Accept a future validated `THW` mask directly. Save source hashes,
SAM2 metadata, dimensions, foreground fractions and mapping in the audit.

- [ ] **Step 4: Run tests and commit**

Run all focused tests and commit with message
`feat: materialize WAN subject mask tubes`.

### Task 3: New Managed Baseline Registration and Adapter

**Files:**
- Create: `src/physbench/baselines/wan22_st_tube_iou.py`
- Create: `src/physbench/baseline_plugins/wan22_st_tube_iou.py`
- Create: `src/physbench/baseline_runtime/drivers/wan22_st_tube_iou.py`
- Create: `baselines/wan22_st_tube_iou/__init__.py`
- Create: `baselines/wan22_st_tube_iou/adapter.py`
- Create: `baselines/wan22_st_tube_iou/driver.py`
- Create: `baselines/wan22_st_tube_iou/baseline.json`
- Create: `baselines/wan22_st_tube_iou/baseline.local.example.json`
- Create ignored local file: `baselines/wan22_st_tube_iou/baseline.local.json`
- Create: `baselines/wan22_st_tube_iou/.gitignore`
- Modify: `tests/test_wan22_st_tube_iou.py`

**Interfaces:**
- Produces: baseline ID `wan22_ti2v_5b_lora_r32_st_tube_iou_v1`, JSONL training rows containing `video`, `subject_mask`, prompt and audit identity, and a driver fingerprint covering all new runtime files.

- [ ] **Step 1: Write failing registration and adapter tests**

Assert bundle validation, seven supported scenes, ignored structured physics,
complete loss config, stock inference script, world-aligned four-GPU batch
semantics, local devices `0,1,2,3`, train-only sibling manifest discovery, and
absence of mask paths from prepared generation jobs.

- [ ] **Step 2: Run and observe failure**

Run all focused tests.

- [ ] **Step 3: Implement adapter and driver**

Subclass the existing audited WAN components only from new files. Override the
training metadata/command/environment, freeze the accelerate config, preserve
scene balancing, materialize tubes only inside executable training, and expose
all auxiliary settings as explicit environment variables. Use defaults
`enable_st_iou_loss=true`, `lambda_st=0.1`, `st_iou_eps=1e-6`,
`st_loss_weighting=linear_clean`, `st_noise_threshold=0.5`,
`st_loss_warmup_steps=100`, micro-batch `1`, global batch `8`, and
optimizer-state saving. Preserve `none`, `linear_clean`, and `threshold` as
config-only ablations.

- [ ] **Step 4: Validate and commit**

Run focused and integrated-baseline tests plus `physbench baseline validate`,
then commit with message `feat: register WAN tube IoU baseline`.

### Task 4: Custom WAN Flow Matching Trainer

**Files:**
- Create: `scripts/wan22_st_tube_iou_train.py`
- Create: `scripts/train_wan22_st_tube_iou.sh`
- Modify: `tests/test_wan22_st_tube_iou.py`

**Interfaces:**
- Consumes: DiffSynth pipeline inputs and metadata `subject_mask`.
- Produces: base/ST/total losses, JSONL diagnostics, LoRA-only checkpoints, paired head checkpoints, optional training state, and gradient/performance audits.

- [ ] **Step 1: Write failing trainer-contract tests**

With fake scheduler, pipeline and DiT modules, assert per-sample timesteps,
correct TI2V first-frame reconstruction, exact base parity when disabled,
warmup/weighting, LoRA-only export filtering, paired head save/load, all metric
keys, and nonzero LoRA gradient attributable to ST loss.

- [ ] **Step 2: Run and observe failure**

Run the `TrainerContractTests` class.

- [ ] **Step 3: Implement the module and distributed loop**

Mirror the existing four-GPU symbol-value training loop. Run existing pipeline
units once, sample batch-sized timesteps, call the unchanged WAN model function,
reconstruct full `x0_hat`, align GT masks to its actual shape, compute the
independent tube loss, gather logs through Accelerate, and profile CUDA time and
peak allocated memory. The main-rank JSONL record contains `base_loss`,
`st_loss`, `st_loss_unweighted`, `st_iou`, `noise_sigma`, `noise_weight`,
`lambda_effective`, `total_loss`, `gt_foreground_fraction`, and
`pred_foreground_fraction`.

- [ ] **Step 4: Implement checkpoint separation and resume**

Export only `pipe.dit` LoRA tensors to `step-N.safetensors`; save the head and
loss config to `step-N.st-head.safetensors`; save optimizer, scheduler, RNG,
sampler epoch and counters to training state; validate the complete set before
resume. The generation path must only receive the LoRA file.

- [ ] **Step 5: Run tests and commit**

Run all focused tests and `bash -n scripts/train_wan22_st_tube_iou.sh`, then
commit with message `feat: train WAN LoRA with tube IoU`.

### Task 5: Documentation, Integration, and Dry Run

**Files:**
- Modify: `README.md`
- Create at runtime: `run/wan22_st_tube_iou_v1_v14_dryrun/`

**Interfaces:**
- Produces: validated codebase and a sealed dry-run with 806 train cases and 110 inference jobs.

- [ ] **Step 1: Run relevant test suites**

Run: `PYTHONPATH=src:tests:. ./.venv/bin/python -m unittest tests.test_wan22_st_tube_iou tests.test_integrated_baselines -v`

- [ ] **Step 2: Build the dry run**

Run: `PYTHONPATH=src ./.venv/bin/python -m physbench atomic-run --dataset datasets/releases/13.0.0/dataset.json --task tasks/experiments/seven_scene_entity_vector_finetune_eval.json --baseline wan22_ti2v_5b_lora_r32_st_tube_iou_v1 --run-id wan22_st_tube_iou_v1_v14_dryrun --output-root run`

- [ ] **Step 3: Audit immutable identities and leakage**

Verify Dataset/Task/baseline digests, exact counts, scene balancing, artifact
paths, trainer config, and that neither mask paths nor tube artifacts appear in
any evaluation adaptation or inference job.

- [ ] **Step 4: Update README and commit**

Document the baseline, loss and pseudo-tube source, then commit with message
`docs: document WAN tube IoU baseline`.

### Task 6: Real WAN Gradient and Performance Preflight

**Files:**
- Create at runtime: `run/wan22_st_tube_iou_v1_preflight/`

**Interfaces:**
- Produces: a real WAN gradient audit, 1--3 optimizer steps, and baseline/enabled time and peak-memory measurements.

- [ ] **Step 1: Materialize a seven-case stratified subset**

Use one training case per supported scene; require each tube to match its
normalized video's pixel frame count and expected VAE latent length.

- [ ] **Step 2: Run a real ST-only backward audit**

Set base-loss contribution to zero for the audit and require finite nonzero
gradient norms in at least one LoRA A/B pair and both occupancy-head layers.

- [ ] **Step 3: Run three optimizer steps and A/B profile**

Run disabled and enabled modes under the same seed, batch and inputs. Discard
the first warm-up step, then record mean step time and CUDA peak memory.

- [ ] **Step 4: Debug and regress any runtime defect**

Preserve the failing log, isolate the root cause, add a regression test, apply
the minimum correction, and rerun the focused test and preflight.

### Task 7: Four-GPU Training, Inference, and Evaluation

**Files:**
- Create at runtime: `run/wan22_st_tube_iou_v1_v14/`

**Interfaces:**
- Produces: terminal full-run state, LoRA/head/training-state artifacts, 110 predictions, and canonical v14 results.

- [ ] **Step 1: Execute the immutable run**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src ./.venv/bin/python -m physbench atomic-run --dataset datasets/releases/13.0.0/dataset.json --task tasks/experiments/seven_scene_entity_vector_finetune_eval.json --baseline wan22_ti2v_5b_lora_r32_st_tube_iou_v1 --run-id wan22_st_tube_iou_v1_v14 --output-root run --execute`

- [ ] **Step 2: Monitor training**

Track run state, tube audits, JSONL loss metrics, GPU processes, scheduled
optimizer steps and checkpoint pairs until training is terminal.

- [ ] **Step 3: Monitor stock inference**

Require one terminal prediction per job and verify generation workers never
load auxiliary-head files.

- [ ] **Step 4: Monitor canonical v14 evaluation**

Require case and task result artifacts, then record strict coverage, overall
score and scene breakdowns.

### Task 8: Final Verification and Handoff

**Files:**
- Read: `run/wan22_st_tube_iou_v1_v14/state.json`
- Read: `run/wan22_st_tube_iou_v1_v14/artifacts/wan22/`
- Read: `run/wan22_st_tube_iou_v1_v14/predictions.jsonl`
- Read: `run/wan22_st_tube_iou_v1_v14/evaluation/task_result.json`

**Interfaces:**
- Produces: evidence-backed report with formulas, files, data flow, occupancy source, actual losses, gradients, tests, performance, scores, and limitations.

- [ ] **Step 1: Run clean verification**

Run `git diff --check`, the focused and integrated test suites, and baseline
validation from fresh processes.

- [ ] **Step 2: Audit repository and run artifacts**

Confirm no shared production file or existing baseline changed, no Dataset
source was mutated, no mask leaked to inference, checkpoint hashes are present,
coverage is exact, and the run state is terminal.

- [ ] **Step 3: Report results and limitations**

State the formulas, config, commands and outcomes, measured overhead, gradient
norms, scores, SAM2 pseudo-label limitation, and any unavailable cases.
