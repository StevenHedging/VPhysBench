# WAN2.2 Subject Geometry Trust

This baseline fine-tunes the WAN2.2 TI2V 5B DiT LoRA while keeping the
parent ST-Tube-IoU occupancy head fixed. The head remains in the autograd
path, so auxiliary gradients reach the DiT LoRA without updating or exporting
the probe.

## Objective

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
