# WAN2.2 Subject Geometry Trust

This baseline fine-tunes the WAN2.2 TI2V 5B DiT LoRA while keeping the
parent ST-Tube-IoU occupancy head fixed. The head remains in the autograd
path, so auxiliary gradients reach the DiT LoRA without updating or exporting
the probe.

## Objective

The vendored WAN scheduler and the training entrypoint use the same rectified
flow parameterization:

```text
x_sigma = (1 - sigma) * x0 + sigma * epsilon
v_target = epsilon - x0
v_hat = DiT(x_sigma, sigma * 1000, condition)
x0_hat = x_sigma - sigma * v_hat
```

Thus larger `sigma` is noisier and `1 - sigma` is the clean-estimate weight.
For TI2V, the conditioned first latent is inserted into `x_sigma`, excluded
from the Flow-MSE tail, and restored in `x0_hat`. The 5B checkpoint loaded by
this recipe has one `dit` model rather than the optional Wan dual-DiT path, so
there is no expert switch or routing policy to change.

For a noisy latent `x_sigma` and predicted flow `v_theta`, the trainer first
recovers the clean estimate

```text
x0_hat = x_sigma - sigma * v_theta
p       = sigmoid(H_parent(x0_hat))
```

where `H_parent` is the frozen parent occupancy probe. The first conditioned
latent is restored before probing, following the stock TI2V training path.
The SAM2-propagated subject mask is conservatively max-pooled onto the actual
WAN latent grid to form target tube `m`.

After a 50-step linear warmup, the optimized loss is

```text
L = L_flowmatch
  + w(sigma) * [
        0.020 * L_global_tube_iou(p, m)
      + 0.010 * L_mean_frame_iou(p, m)
      + 0.020 * L_first_frame_relative_centroid(p, m)
      + 0.005 * L_log_foreground_mass(p, m)
      + 0.050 * L_spatial_covariance(p, m)
    ]
  + 1.000 * L_relative_lora_trust
```

`w(sigma) = 1 - sigma` emphasizes cleaner estimates. Global Tube-IoU fixes
absolute space-time support; mean frame IoU prevents large frames from
dominating small or short-lived subjects; anchored displacement supervises
motion independently of a common offset; log mass penalizes missing objects;
and the normalized `yy`, `xx`, and `yx` covariance terms distinguish separated
objects from a same-centroid collapse. The trust term is squared LoRA distance
from the parent checkpoint, normalized by the parent's squared LoRA norm.

## Frozen recipe

- Parent: `wan22_ti2v_5b_lora_r32_st_tube_iou_v1`, step 2184.
- Trainable parameters: rank-32 DiT LoRA only.
- Probe: fixed FP32 `Conv3d-SiLU-Conv3d`; never needed at inference.
- Learning rate: `2.5e-5`; one 234-step epoch; AdamW; BF16.
- Hardware contract: four processes on `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Task: six training scenes, balanced to 312 rows per scene (1872 rows from
  679 unique cases); five evaluation scenes and 76 inference jobs.
- Generation: stock WAN2.2 TI2V LoRA inference. Masks and the occupancy probe
  are training-only and cannot enter inference jobs.

Local model paths and the four-process Accelerate configuration belong in the
ignored `baseline.local.json` and `accelerate.local.yaml` files. Run planning,
training, inference, and evaluation through the benchmark's `atomic-run` or
`matrix-run` entrypoint so the task, protocol, and component fingerprints are
sealed together.

## Empirical rationale and current status

All values below are observed scene-macro diagnostics under
`scene_default_v14` (64 evaluated jobs out of 76); strict official scores are
unset because 12 reference videos fail closed in the evaluator. The first
three iterations were trained and evaluated, while this geometry recipe is a
registered next iteration whose GPU run is still pending.

| iteration | physical | CSTI | result |
| --- | ---: | ---: | --- |
| learned global Tube-IoU parent | 0.36630 | 0.26758 | strongest completed run |
| subject-flow + temporal delta | 0.23743 | 0.20172 | over-weighted local velocity |
| absolute centroid trajectory | 0.34969 | 0.26023 | close overall, weak collision |
| anchored displacement + trust | 0.27719 | 0.23295 | helps curved motion, loses absolute support |

The anchor iteration improved paired parabolic CSTI by `+0.03098` and circular
CSTI by `+0.06292`, but reduced incline CSTI by `-0.25107`; green-background
incline cases were the dominant failure. This is why the next recipe combines
relative displacement with global/mean-frame support, mass, and covariance
instead of using a trajectory loss alone. A high-comparability archived
FlowMatch-only control scored physical `0.29789` and CSTI `0.24974` at the same
observed coverage. Its paired CSTI difference versus the Tube-IoU parent was
`-0.01486` with bootstrap 95% interval `[-0.05659, 0.02642]`, while its paired
physical difference was `-0.07020` with interval `[-0.13120, -0.01307]`.
That control used the same 806 case IDs, optimizer hyperparameters, global
batch, prompts, seeds, runner, and byte-identical evaluation conditioning
images, but differed in process/accumulation layout and had 18 lightly
re-encoded collision training videos, so it is not claimed as a strict
single-variable ablation.
