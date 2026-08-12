# WAN2.2 Subject-Motion Flow Baseline Design

## Goal and Evidence

Register a second WAN2.2-TI2V-5B fine-tuning baseline that gives small moving
physical subjects direct, area-normalized gradient signal while retaining the
already implemented whole-tube occupancy objective. The first
`wan22_ti2v_5b_lora_r32_st_tube_iou_v1` run provides the starting evidence:

- 2,184 training steps completed with finite non-zero LoRA and occupancy-head
  gradients;
- the last-100-step latent Tube IoU reached `0.4983`, showing the proxy head can
  learn the pseudo-label tubes;
- canonical CSTI remained weak for collision (`0.0512`), projectile motion
  (`0.0000`), and pendulum (`0.0972`);
- training subjects occupy roughly `0.1%--1%` of the video volume, so ordinary
  full-video Flow Matching MSE strongly dilutes their contribution;
- a learned occupancy head can co-adapt with the LoRA and does not directly
  force the Wan velocity field to reduce its error on the physical subject.

The new baseline therefore adds direct masked velocity and temporal-change
objectives. It is an incremental warm-start experiment from the final audited
Tube-IoU LoRA/head, not a from-scratch comparison. Initialization provenance is
part of the Baseline descriptor and checkpoint manifest.

## Alternatives Considered

### 1. Increase `lambda_st`

This is cheap but keeps all supervision behind the learned occupancy proxy. It
may improve the proxy without improving generated subject motion, so it is not
the first experiment.

### 2. Subject-normalized Flow Matching only

This directly addresses foreground dilution and has no new prediction head.
However, a per-voxel velocity objective still does not explicitly emphasize
frame-to-frame motion. It is retained as one component rather than the whole
method.

### 3. Tube IoU plus subject Flow and latent temporal-difference losses

This is the selected approach. It preserves the previous whole-trajectory
occupancy signal, directly upweights the physical subject in the native Wan
objective, and adds a low-noise temporal-change constraint aimed at collision,
projectile, and oscillatory dynamics. TIV-Diffusion's object-centric motion
alignment and recent work reading trajectories from video-diffusion latents
support the general direction, but the implementation here is deliberately
minimal and uses only already authorized VPhysBench training supervision.

Primary-source context:

- TIV-Diffusion: <https://arxiv.org/abs/2412.10275>
- Motion-Zero: <https://arxiv.org/abs/2401.10150>

## Baseline Identity and Isolation

The registered identity is
`wan22_ti2v_5b_lora_r32_subject_motion_v1`, version `1.0.0`. It is an
independent managed Baseline bundle and owns its plugin, managed driver,
adapter, loss module, trainer, launch script, tests, and documentation. Existing
Baseline descriptors and execution semantics remain unchanged.

The new adapter subclasses the Tube-IoU adapter solely to reuse its audited
training-only subject-tube preparation and separate LoRA/head checkpoint
contract. Its runtime can seed a new run-local mask directory from a validated
previous tube cache. Every reused NPZ must have a paired audit whose `case_id`,
recorded SHA-256, binary dtype, and source fingerprint validate. Missing or
invalid cache entries fall back to normal SAM2 materialization; they never
silently bypass supervision.

Only GPUs `0,1,2,3` may be exposed by the local deployment. Training and
generation commands fail if their resolved device set differs.

## Data Flow

The input contract is unchanged:

```text
training:  prompt + canonical reference video + training-only first-frame masks
inference: prompt + first-frame image
```

Training videos are normalized to 24 FPS, 121 frames, and the scene-specific
`480x832` or `832x480` Wan canvas. First-frame instance masks are propagated by
SAM2 on those exact normalized frames and unioned into a binary `THW` subject
tube. No mask, reference video, or evaluator artifact enters an inference job.

The synchronized 2026-08-12 Task separates training and evaluation scenes.
It excludes `push_bottle`, trains on 679 unique cases across the other six
scenes, and evaluates the five scenes implemented by current v14. Training is
balanced deterministically to 312 rows per selected scene, or 1,872 rows per
epoch. The first experiment warm-starts from the final Tube-IoU checkpoint and
runs two epochs: four ranks, micro-batch one per rank, two accumulation
micro-steps, global batch eight, 234 optimizer steps per epoch, and 468
optimizer steps total.

## Audited Wan Flow Parameterization

For a per-sample Wan noise level `sigma`, the deployed DiffSynth trainer uses

\[
x_\sigma=(1-\sigma)x_0+\sigma\epsilon,
\qquad v^*=\epsilon-x_0.
\]

The DiT predicts `v_hat`, so one forward gives

\[
\hat{x}_0=x_\sigma-\sigma\hat{v}.
\]

TI2V latent frame zero is the known conditioning latent. It replaces the noisy
slice, is excluded from velocity losses, and is inserted unchanged into
`x0_hat`. No iterative sampler or RGB decoder appears in the training graph.

WAN2.2-TI2V-5B uses 48-channel latents and the VAE maps 121 pixel frames to 31
latent frames with 16x spatial compression. Masks are conservatively aligned
to the actual `BTHW` latent shape with adaptive max pooling so sub-latent-cell
subjects cannot disappear during alignment.

## Losses

### Stock full-video Flow Matching loss

Let

\[
e_{b,t,h,w}=\operatorname{mean}_c
(\hat v_{b,c,t,h,w}-v^*_{b,c,t,h,w})^2.
\]

The unchanged Wan scheduler-weighted loss is `L_base`.

### Whole-volume Tube IoU

The inherited `Conv3d(48,32,3) -> SiLU -> Conv3d(32,1,1)` occupancy head maps
`x0_hat` to a soft latent tube. One soft IoU is computed over the complete
`T*H*W` volume per sample. This term remains training-only.

### Area-normalized subject Flow loss

With aligned subject occupancy `m`, the TI2V tail support is the union of each
predicted frame and its preceding endpoint,

\[
q_t=\max(m_t,m_{t-1}).
\]

This includes both the origin and destination footprint of moving subjects and
remains meaningful when a subject leaves the canvas immediately after the
conditioned first frame. The area-normalized loss is

\[
L_{subject,b}=
\frac{\sum_{t,h,w}q_{b,t,h,w}e_{b,t,h,w}}
{\sum_{t,h,w}q_{b,t,h,w}+\varepsilon}.
\]

Each sample is then multiplied by the same Wan scheduler training weight used
by `L_base` before the batch mean. Normalizing by foreground volume means a
small ball receives a comparable subject-loss scale to a large object instead
of being diluted by the background. Empty aligned subject tubes are fatal data
errors.

### Subject-supported latent temporal-difference loss

For consecutive clean-latent slices,

\[
\Delta\hat x_{0,t}=\hat x_{0,t}-\hat x_{0,t-1},\qquad
\Delta x_{0,t}=x_{0,t}-x_{0,t-1}.
\]

The temporal support is the union of the subject at both endpoints,

\[
q_t=\max(m_t,m_{t-1}).
\]

Using a channel-mean Charbonnier penalty
`rho(z)=sqrt(z^2+1e-6)`,

\[
L_{motion,b}=
\frac{\sum_{t,h,w}q_{b,t,h,w}\,
\operatorname{mean}_c\rho(
\Delta\hat x_{0,b,c,t,h,w}-\Delta x_{0,b,c,t,h,w})}
{\sum_{t,h,w}q_{b,t,h,w}+\varepsilon}.
\]

Because `x0_hat` is least reliable near pure noise, this term uses
`w(sigma)=1-sigma`. The subject Flow loss uses the native Wan scheduler weight
instead, since it is directly a velocity error.

### Combined objective

All auxiliary coefficients use a linear 100-step optimizer warmup:

\[
r=\min(1,(step+1)/100).
\]

The first experiment uses

\[
L = L_{base}
+r\,0.05L_{tube}
+r\,0.10L_{subject}
+r\,0.05L_{motion}.
\]

The weights are intentionally conservative for a warm-start run. A real
three-step enabled/disabled preflight must inspect finite gradients, relative
loss scales, step time, and CUDA peak memory before the two-epoch training. If
either new contribution exceeds `0.5 * L_base` after applying its coefficient,
its coefficient is reduced so the object-focused branch cannot immediately
erase the parent model.

## Training, Checkpoints, and Metrics

The trainable parameters are the existing rank-32 DiT LoRA targets
`q,k,v,o,ffn.0,ffn.2` and the fp32 occupancy head. Base Wan parameters and VAE
remain frozen. AdamW, `1e-4` learning rate, `0.01` weight decay, ConstantLR,
bf16 Wan execution, gradient checkpointing, and seed 42 are retained.

Warm start requires both the parent `step-2184.safetensors` and matching
`step-2184.st-head.safetensors`. New saves retain the isolation contract:

- `step-N.safetensors`: LoRA tensors only, accepted by stock Wan inference;
- `step-N.st-head.safetensors`: occupancy head and complete loss config;
- optimizer/scheduler plus per-rank RNG state for recovery.

Every synchronized optimizer step logs the previous Tube-IoU metrics plus
`subject_flow_loss`, `motion_delta_loss`, each effective coefficient, its
weighted contribution, subject support fraction, LoRA gradient norm,
occupancy-head gradient norm, step time, and CUDA peak memory. Non-finite loss
or gradient and missing required gradients are fatal.

## Inference and Evaluation

Inference deliberately ignores the occupancy head and all masks. It loads only
the new LoRA into stock WAN2.2-TI2V-5B and uses the sealed task prompt,
conditioning first frame, seed 42, 50 inference steps, CFG 5.0, and the normal
media contract. The 76 jobs run on persistent workers restricted to GPUs
0--3.

Generated videos undergo the canonical v14 evaluation without loss-specific
shortcuts. Comparison uses:

- CSTI observed macro mean and per-scene CSTI;
- evaluator coverage/status counts;
- canonical observed physical score and per-scene values;
- generation failures and media-contract integrity;
- training gradient, runtime, and memory diagnostics.

The main decision signal is whether collision, projectile, and pendulum CSTI
improve without a material collapse in circular and incline results. Official
aggregate scores remain subject to the benchmark's strict complete-coverage
policy.

## Testing and Failure Handling

Unit tests cover exact foreground normalization, scheduler weights, empty-mask
rejection, temporal support, identical/shifted temporal differences, gradient
flow to predicted velocity/clean estimate, disabled-loss equivalence, warm
start pairing, inference head exclusion, GPU restriction, cache validation,
registration, and managed-runtime discovery.

Runtime preparation is fail-closed for malformed masks, mismatched case IDs,
hash mismatches, invalid frame compression, missing warm-start pairs, and
unexpected GPU assignments. Canonical run artifacts are never overwritten;
each experiment uses a new run ID.

## Iteration Rule

After canonical evaluation, the next baseline is selected from measured
failure modes rather than precommitted:

- if subject Flow improves identity/coverage but not CSTI, strengthen explicit
  trajectory geometry rather than its coefficient;
- if motion loss helps projectile/pendulum but harms collision, add
  event-aware temporal weighting around high mask displacement;
- if all object-centric terms improve training proxies but not generated
  motion, remove the co-adaptive occupancy branch and test direct subject Flow
  alone;
- if quality collapses, lower auxiliary weights or shorten warm-start
  adaptation before adding new mechanisms.

The remaining 11-hour experiment window is used for as many complete
register-train-infer-evaluate-reflect cycles as runtime permits, always on GPUs
0--3.
