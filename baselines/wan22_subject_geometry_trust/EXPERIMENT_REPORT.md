# WAN2.2 Subject-Loss Iteration Report

Date: 2026-08-13

This report records the registered, trained, inferred, and evaluated loss
iterations that led to `wan22_ti2v_5b_lora_r32_subject_geometry_trust_v1`.
Only baseline implementation files were changed except for the explicitly
requested 2026-08-12 evaluation/task protocol sync and its contract tests.

## Benchmark protocol used for iteration

- Task: `six_scene_train_five_scene_eval_v14`.
- Training: 679 unique cases from collision, incline, parabolic, pendulum,
  circular, and vertical spring; the balanced WAN loader materializes 312 rows
  per scene, or 1872 rows per epoch.
- Evaluation: 76 jobs from collision, incline, parabolic, pendulum, and
  circular motion.
- Evaluation protocol: `scene_default_v14`, fingerprint
  `9db7ba1d6a7b9f54a14491f66d035821d353ba9570fd6ccad345305d29989e78`.
- CSTI tolerance: half the equivalent diameter measured from each reference
  subject tube; the conditioned initial frame is excluded.
- Strict aggregation is fail closed. Twelve of the 76 reference videos cannot
  be observed reliably, so all reported aggregate numbers below are observed
  diagnostics over 64 jobs. The official strict score remains unset.

## Verified WAN2.2 parameterization

The vendored scheduler and trainer implement rectified flow as

```text
x_sigma = (1 - sigma) * x0 + sigma * epsilon
v_target = epsilon - x0
v_hat = DiT(x_sigma, sigma * 1000, condition)
x0_hat = x_sigma - sigma * v_hat
```

Large `sigma` is noisy; therefore the auxiliary clean weighting is
`w(sigma) = 1 - sigma`. For TI2V, the known first latent replaces the noisy
first latent, is excluded from Flow-MSE, and is restored in `x0_hat`. The 5B
configuration loads one `dit`, not the optional dual-DiT WAN path, so no expert
routing was changed.

The auxiliary branch performs one normal random-flow-timestep forward over the
whole video latent. It does not run an inference sampler, decode RGB, apply a
hard threshold, or use evaluation-time EDT/CSTI inside training.

## Original Tube-IoU baseline

```text
GT video -> VAE x0 -> sample sigma -> x_sigma -> WAN DiT LoRA -> v_hat
        -> one-step x0_hat -> trainable FP32 latent occupancy head -> sigmoid p
GT subject tube -> synchronized SAM2 propagation -> WAN-grid max/nearest align m
(p, m) -> one global soft IoU over T * H * W -> total loss -> backward
```

For each sample independently,

```text
intersection = sum_T,H,W(p * m)
union        = sum_T,H,W(p + m - p * m)
L_tube       = 1 - (intersection + eps) / (union + eps)
```

The parent used `lambda_st=0.1`, 100-step warmup, linear-clean weighting,
rank-32 LoRA, learning rate `1e-4`, and eight epochs (2184 optimizer steps).
The saved LoRA and occupancy head are separate artifacts:

- LoRA SHA256: `9b21dc78b48a55d0de4da865018f9d22a87fecdb60b6e6d59c78f6916a4d1b34`.
- Head SHA256: `ba627d105ee6c8f257e4253312008b2935f2c48b5c498f9e91f5377b77f6bdc3`.

The first/last 100-step mean latent Tube-IoU changed from `0.05000` to
`0.49825`. Predicted foreground fraction changed from `0.21525` to `0.01152`,
close to the final GT fraction `0.01090`; this rules out all-zero/all-one probe
collapse in the training audit. Tube-only backward produced finite nonzero
gradients in both the DiT LoRA and occupancy head.

## Iterations and evaluation

| completed iteration | training | physical | CSTI |
| --- | --- | ---: | ---: |
| global Tube-IoU parent | 8 epochs / 2184 steps | 0.36630 | 0.26758 |
| subject-flow + clean temporal delta | 2 epochs / 468 steps | 0.23743 | 0.20172 |
| absolute centroid position/velocity | 1 epoch / 234 steps | 0.34969 | 0.26023 |
| first-frame-relative displacement + LoRA trust | 1 epoch / 234 steps | 0.27719 | 0.23295 |

The subject-flow objective was too aggressive: local velocity matching did not
preserve generation quality and reduced paired physical/CSTI scores. Absolute
centroids recovered incline but harmed collision and did not beat the parent.
Relative anchored displacement plus a trust region helped parabolic CSTI by
`+0.03098` and circular CSTI by `+0.06292`, but hurt incline CSTI by `-0.25107`.
Green-background incline cases were the dominant failure. Relative geometry
alone can describe motion while allowing a common spatial/appearance offset.

An archived FlowMatch-only checkpoint trained on the same 806 case IDs was
also re-evaluated on the 76 current jobs. It produced observed scene-macro
physical `0.29789` and CSTI `0.24974`. Across the common 64 evaluated jobs,
its physical difference from the Tube parent was `-0.07020` (bootstrap 95%
`[-0.13120, -0.01307]`) and CSTI difference was `-0.01486`
(`[-0.05659, 0.02642]`). This is a high-comparability control rather than a
strict single-variable ablation: it used 8 processes x accumulation 1 instead
of 4 x 2, and 18 old collision training videos have small re-encoding
differences. Evaluation conditioning images were pairwise byte-identical.

## Final registered geometry candidate

The parent occupancy head is frozen in FP32 but remains in the autograd path;
only the parent-initialized DiT LoRA is trainable. With a 50-step warmup,

```text
L = L_flowmatch
  + (1 - sigma) * (
        0.020 * L_global_tube_iou
      + 0.010 * L_mean_frame_iou
      + 0.020 * L_first_frame_relative_centroid
      + 0.005 * L_log_foreground_mass
      + 0.050 * L_normalized_spatial_covariance
    )
  + 1.000 * L_relative_lora_trust
```

Global Tube-IoU anchors absolute space-time support. Mean-frame IoU prevents
large or long-lived subjects from dominating. Relative centroids supervise
trajectory without forcing a fixed initial offset. Log mass strengthens small
or disappearing subjects. The normalized `yy`, `xx`, and `yx` covariance
distinguishes separated objects from a same-centroid collapse. The trust loss
is squared LoRA distance from the parent divided by the parent's squared LoRA
norm.

The planned run is one 234-step epoch at learning rate `2.5e-5`, global batch
8, on `CUDA_VISIBLE_DEVICES=0,1,2,3`. A sealed no-execute AtomicRun produced
task-instance digest
`a9a90e01788eca1a9b3a649eca68f2d82fe384d4b33f7b51c3b518f4c481d342`.
The final GPU training/inference was not started because an unrelated process
occupied GPUs 0-7 at roughly 68.5 GiB per GPU; that process was not modified.

## Runtime and verification evidence

Completed training performance after the first ten steps:

| iteration | mean step time | median step time | peak allocated memory/process |
| --- | ---: | ---: | ---: |
| Tube parent | 10.368 s | 10.888 s | 28.120 GiB |
| subject motion | 10.359 s | 10.924 s | 28.120 GiB |
| centroid trajectory | 10.366 s | 11.039 s | 28.120 GiB |
| anchor trust | 10.586 s | 11.230 s | 28.585 GiB |

Final focused verification passed 146 tests and 48 subtests, covering the
Tube formula and edge cases, gradient paths, LoRA/head checkpoint separation,
subject-loss objectives, fixed-head behavior, CSTI v14, the 2026-08-12 task
split, TaskBuilder, and runtime contracts. Both final candidates pass
`physbench baseline validate`; the geometry bundle digest is
`abf8a39cc02abbae990548769ddc067336621a469e604ee3ce6a64dfa4d59a58`.

## Known limitations

- `x0_hat` becomes unreliable at high noise; linear-clean weighting reduces
  but does not remove this source of noisy gradients.
- The learned latent probe is cheaper than VAE decode plus a differentiable
  RGB segmenter, but its occupancy error is an optimization proxy error.
- SAM2 propagation and conservative latent-grid pooling can over-cover thin
  trails or merged objects; covariance helps but does not provide identities.
- Very small foregrounds still have weaker gradients than the full-frame Flow
  objective. Frame balancing, log mass, and covariance address different parts
  of this problem but require a real run to validate their combined effect.
- The training loss is exact soft Jaccard without tolerance, prefix scoring, or
  EDT. Adaptive EDT is evaluation-only.
- The current 5B run has no active dual-DiT expert routing. A WAN variant with
  timestep-routed experts would require a fresh gradient audit without changing
  its native routing rule.
