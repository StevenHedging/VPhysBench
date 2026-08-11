# WAN2.2 Symbol-Value Cross-Attention Main-Benchmark Rerun Design

## Goal

Register the previously experimental
`wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1` implementation as a
first-class VPhysBench baseline, then train and evaluate it with the current
main-tree Dataset and latest official fine-tune/evaluation Task.

## Evidence and Scope

The historical AtomicRun
`run/wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1` froze Dataset
13.0.0 against a temporary seven-scene Task whose protocol was explicitly
`scene_default_v10_seven_scene_unscored`. Its run state never advanced beyond
`frozen`; a model process was started, but no reusable checkpoint or canonical
evaluation was published. The historical portable implementation remains
available in Git history, while the current worktree contains only the ignored
local deployment override.

This work will not modify or resume the historical run. It will not change the
Dataset, official Task, evaluator, or the semantics of other baselines.

## Baseline Registration

Restore the dedicated symbol-value bundle and runtime implementation from the
existing feature history, adapting it only where current main-tree APIs require
it. The registered identity remains
`wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1` version `1.0.0`.

The baseline consumes the canonical Case caption and structured formal physics
quantities. Frozen UMT5 subword embeddings represent quantity symbols. An
independently trainable value encoder represents SI magnitude, dimension, and
unit; bottleneck cross-attention injects the fused symbol/value tokens into the
frozen UMT5 text context before WAN DiT cross-attention. Rank-32 DiT LoRA and
the symbol-value conditioner train jointly. Base WAN, UMT5, and VAE weights
remain frozen.

Portable files live under
`baselines/wan22_symbol_value_cross_attention/`. The existing ignored
`baseline.local.json` is preserved when valid. Deployment validation found its
historical Python and project paths no longer exist, so the local-only override
is updated to the already validated `wan22_pair_text_runtime` deployment used
by the recent comparable WAN experiments. Portable baseline identity and
semantics remain unchanged.

## Comparable Training and Evaluation Contract

Use the current main-tree Dataset
`datasets/releases/13.0.0/dataset.json`, View A, seed 42, and the official Task
`tasks/official/five_scene_finetune_eval_csti_identity_v3.json`. This selects
pendulum, collision, inclined-plane, circular-motion, and parabolic-motion ID
test cases and evaluates them with `scene_default_v14`.

Match the recent WAN controls' training budget and sampling contract:

- rank-32 LoRA targeting `q,k,v,o,ffn.0,ffn.2`;
- learning rate `1e-4`, AdamW, ConstantLR, weight decay `0.01`;
- one dataset repeat, scene balancing by
  `oversample_each_scene_to_largest_world_aligned`;
- eight GPUs, seed 42, bf16 model training and fp32 conditioner parameters;
- 195 optimizer steps per epoch for eight epochs, 1560 total steps; this is
  derived from the official Task's five scene-balanced 312-row buckets across
  eight workers;
- optimizer state and recoverable checkpoints retained in the AtomicRun.

Inference uses the frozen Task jobs, first-frame conditioning, 121 frames at
24 FPS, 50 diffusion steps, CFG 5.0, LoRA alpha 1.0, and seed 42.

## Execution and Artifacts

Before allocating GPUs, run focused unit tests, full baseline discovery and
validation, deployment validation, compile checks, and an AtomicRun dry-run.
Confirm the dry-run freezes Dataset release 13.0.0, the v14 Task, the expected
five scenes, and the 1560-step training contract.

The executed run uses a new immutable ID such as
`wan22_symbol_value_cross_attention_1560_v1_v14`. It owns all staged training
media, checkpoints, inference outputs, logs, provenance, and evaluation
artifacts. The old run remains unchanged.

Execution proceeds through training, inference, and canonical evaluation.
Operational failures are diagnosed from run-local state and logs; fixes are
tested and a fresh run ID is used whenever immutable frozen inputs or code
fingerprints change.

## Validation and Success Criteria

Registration succeeds when the registry lists, inspects, and validates the
baseline from portable source files and its implementation-specific regression
tests pass on current main.

Experiment success requires:

1. a completed 1560-step training stage with sealed LoRA and conditioner
   checkpoints;
2. one valid prediction record and artifact for every official inference job;
3. canonical `scene_default_v14` case and Task results under the new run;
4. run state `complete`, with task coverage and any unavailable cases reported
   explicitly rather than silently omitted.
