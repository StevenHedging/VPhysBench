# WAN2.2 Spatio-Temporal Tube IoU Baseline Design

## Goal

Register and execute a new WAN2.2-TI2V-5B LoRA baseline whose training objective
adds one differentiable, whole-video soft tube-IoU term to the existing
DiffSynth Flow Matching SFT loss. The inference graph remains the stock WAN2.2
TI2V graph with a standard LoRA checkpoint.

## Audited Starting Point

The deployed DiffSynth revision uses the Wan FlowMatch scheduler. For a sampled
flow noise level `sigma`, it constructs

\[
x_s=(1-\sigma)x_0+\sigma\epsilon,
\qquad v^*=\epsilon-x_0.
\]

The DiT predicts `v_hat`, so the exact one-step clean latent estimate is

\[
\hat{x}_0=x_s-\sigma\hat{v}.
\]

WAN2.2 TI2V fixes latent frame zero to the encoded conditioning frame and the
base loss excludes that model-output slice. The auxiliary path reconstructs
the complete latent video by retaining that known slice and using the formula
above for slices `1:`. No sampler or iterative denoising loop is introduced.

The WAN VAE maps `4n+1` pixel frames to `n+1` latent frames and downsamples each
spatial dimension by eight. The subject tube must therefore be aligned to the
actual clean-latent shape rather than to a hard-coded length.

## Alternatives Considered

1. Reuse existing model mask logits. This would be the cleanest integration,
   but WAN2.2-TI2V-5B exposes no segmentation or occupancy output.
2. Decode `x0_hat` through the frozen VAE and run a frozen pixel segmenter on
   every training step. This gives a recognizable pixel-space mask, but adds a
   large differentiable decoder/segmenter graph, has no suitable domain-frozen
   segmenter in the training runtime, and is prohibitively expensive at
   121x480x832 or 121x832x480.
3. Predict occupancy directly from `x0_hat` with a compact 3D convolutional
   head. This is the selected approach because it preserves gradients to the
   DiT LoRA, works at the native VAE tube resolution, and has bounded memory.

The selected representation is explicitly a learned latent-space occupancy
proxy, not a decoded RGB segmentation. The head is trained jointly from the
tube-IoU supervision. Its weights are saved separately for audit/resume and
are not consumed by inference. This separation keeps the exported inference
artifact a standard WAN LoRA checkpoint.

## Dataset Adaptation

VPhysBench 13.0.0 provides a canonical training video and one or more
first-frame instance masks, not precomputed full-video masks. It also correctly
removes evaluator-only mask assets from normal model inputs. To keep this
baseline self-contained, no compiler or input-contract change is made. During
training preparation only, the new adapter starts from the already authorized
supervised reference-video path, resolves its canonical sibling
`masks/manifest.json`, and requires the manifest `case_id` to equal the active
training case. It records the resolved path and content hash in its own audit.
This lookup does not run for evaluation cases and the path never appears in an
adaptation, native model input, or inference job.

Before optimization, the adapter creates a pseudo-ground-truth subject tube:

- union all physical-subject instance masks from the manifest;
- seed the frozen `facebook/sam2.1-hiera-tiny` video predictor on canonical
  frame zero using the manifest masks' boxes/points;
- propagate each instance through the canonical reference video and union the
  results per frame;
- apply exactly the same physical-time resampling, aspect-preserving contain,
  and padding geometry used for the normalized WAN training video;
- write a compressed uint8 `masks` array with layout `THW`, plus an audit
  record containing source hashes, propagation metadata, frame count, and
  spatial mapping.

SAM2 propagation is an offline target-materialization step and is never part of
the differentiable training graph. If a source manifest already supplies a
valid `THW` tube, the materializer consumes it directly and records that source
instead. Failures are fatal rather than silently repeating the first mask,
because a static mask would change the requested spatio-temporal objective.

## Auxiliary Model and Loss

`LatentOccupancyHead` maps `[B,16,T,H,W]` to one logit channel using
`Conv3d(16,32,3,padding=1)`, SiLU, and `Conv3d(32,1,1)`. The sigmoid output is
`P_pred`. Ground-truth uint8 tubes are converted to float and resized to the
actual latent `T,H,W` with nearest-neighbor interpolation. No framewise loss,
prefix accumulation, cumulative mask, distance transform, or temporal
pooling is used.

For each batch element independently,

\[
I_b=\sum_{t,h,w}P^{pred}_{b,t,h,w}P^{gt}_{b,t,h,w},
\]

\[
U_b=\sum P^{pred}_b+\sum P^{gt}_b-I_b,
\qquad
IoU_b=\frac{I_b+\varepsilon}{U_b+\varepsilon},
\]

\[
L_{ST}=\frac{1}{B}\sum_b(1-IoU_b).
\]

If both predicted and target occupancy are empty for a sample, its score is
defined as one; if only one is empty, the normal soft-IoU formula applies.

The default noise weighting is `linear_clean`, using `w(sigma)=1-sigma`
because Wan `sigma=1` is maximally noisy and `sigma=0` is clean. Supported
ablations are:

- `none`: `w=1`;
- `linear_clean`: `w=1-sigma`;
- `threshold`: `w=1[sigma <= st_noise_threshold]`.

The effective coefficient is linearly warmed up over
`st_loss_warmup_steps` optimizer steps:

\[
\lambda_{eff}=\lambda_{st}\min(1,(step+1)/warmup).
\]

The final optimized scalar is

\[
L=L_{base}+\lambda_{eff}\,\operatorname{mean}_b
\left[w(\sigma_b)(1-IoU_b)\right].
\]

The baseline uses one independently sampled timestep per sample so metrics and
weights have true batch semantics. WAN's dense TI2V-5B DiT has no MoE expert
routing; its forward path, block dispatch, and conditioning remain untouched.

## Training and Checkpoint Contract

The custom trainer follows the repository's audited four-GPU WAN loop:

- CUDA devices `0,1,2,3`, one process per device;
- micro-batch one per rank and global batch eight, hence two accumulation
  micro-steps per optimizer step;
- bf16 base model with fp32 LoRA and occupancy-head trainable parameters;
- AdamW and ConstantLR; the occupancy head uses the same configured learning
  rate and weight decay as LoRA;
- dataset scene balancing remains world-aligned and deterministic;
- gradient checkpointing and WAN TI2V conditioning remain unchanged.

Every model save emits:

- `step-N.safetensors`: LoRA tensors only, directly accepted by stock WAN
  inference;
- `step-N.st-head.safetensors`: occupancy-head tensors and loss configuration;
- optional `step-N.training-state.pt`: optimizer, scheduler, RNG, epoch and
  optimizer-step state when enabled.

Resume requires the matching LoRA, head, and training-state artifacts. The
driver validates that the paired head exists for training recovery. Inference
requires only the LoRA file and deliberately ignores the head.

## Metrics and Diagnostics

The main rank writes JSONL records for `base_loss`, weighted and unweighted
`st_loss`, `st_iou`, `noise_sigma`, `noise_weight`, `lambda_effective`,
`total_loss`, `gt_foreground_fraction`, and `pred_foreground_fraction`.
Distributed values are gathered before logging.

Verification includes:

- exact soft-IoU behavior for identical, disjoint, partial, oversized,
  disappeared, delayed, temporally shifted, both-empty and one-empty tubes;
- gradient flow from `L_ST` through `x0_hat` and the occupancy head;
- temporal alignment tests for 5, 9 and 121 pixel frames, including the known
  first latent slice and off-by-one rejection;
- adapter isolation tests showing mask lookup occurs only during training and
  mask paths never enter generation jobs;
- a real WAN single-forward/backward gradient audit on LoRA tensors;
- 1--3 optimizer-step smoke training;
- baseline-versus-enabled step-time and CUDA peak-memory profiling;
- full four-GPU training, stock inference, and canonical v14 evaluation.

## Registration and Execution

The registered identity is
`wan22_ti2v_5b_lora_r32_st_tube_iou_v1`. It supports the seven-scene v14
fine-tune/eval task and uses normal case prompts without structured-physics
injection. The new bundle owns portable trainer/loss settings; its ignored
`baseline.local.json` binds the existing DiffSynth deployment and GPUs 0--3.
All production-code changes are confined to new files owned by this baseline;
shared compiler contracts and existing baselines remain unchanged.

Execution uses the same Dataset 13.0.0 snapshot and
`tasks/experiments/seven_scene_entity_vector_finetune_eval.json` as the recent
comparable WAN baselines. Run artifacts live under a new, versioned directory
and never alter historical runs.

## Limitations

The supervision tube is SAM2-propagated pseudo-ground truth because the release
does not contain full-video human or simulator masks. It may drift under
occlusion or fast motion, and the latent head can co-adapt with the LoRA. Both
facts are disclosed in the baseline and run audit. A later dataset release with
native full tubes can replace materialization without changing the loss API.
