# WAN2.2 Pair-Text Encoder Ablation Design

## Objective

Test whether the learned analytic symbol/value encoder is the main cause of
poor generation quality. Replace only the construction of each physical token
with a frozen pretrained UMT5 encoding while preserving the Dataset 13 split,
registry-selected quantities, original caption path, post-UMT5 cross-attention,
WAN2.2 DiT LoRA topology, FlowMatch loss, optimizer, generation settings, and
VPhysBench evaluator.

## Compared Treatments

The control is `symbol_value_cross_attention_v1`: frozen symbol subword means
plus a trainable SI value/dimension/unit MLP, followed by the existing
cross-attention conditioner. The treatment renders each registry record as
`<symbol> = <rendered_value> <raw_unit>` (omitting the unit only for unit `1`),
encodes every string independently with the same frozen UMT5-XXL encoder, and
masked-mean-pools the final hidden states to one 4096-dimensional token per
record. No physical pair is appended to the original caption.

The pair encoder is frozen and runs under `torch.no_grad()`. The downstream
trainable path is unchanged: physics LayerNorm, 4096-to-512 projection,
eight-head text-to-physics attention, 512-to-4096 output projection, output
LayerNorm, and residual gate initialized to 0.1. The negative CFG branch remains
ordinary text-only UMT5 context.

## Isolation and Identity

Create a new baseline identity and new checkpoint namespace rather than mutate
the frozen control baseline. Reuse the existing registry and adapter contract,
but include the renderer and encoder mode in fingerprints and token audits.
Training and outputs use a new run directory. Existing checkpoints, runs, and
uncommitted workspace files are read-only controls.

## Training Protocol

Use Dataset 13 View A and the same seven-scene balancing policy: 2,184 rows per
epoch, eight GPUs, global batch 8, and 273 optimizer steps per epoch. Train from
random LoRA and random downstream conditioner initialization for eight complete
epochs, yielding 2,184 optimizer steps. This is the nearest exact epoch boundary
to the requested approximately 2,000 steps and avoids a biased partial epoch.

Keep seed 42, bf16 WAN computation, FP32 downstream conditioner parameters,
LoRA rank 32 and targets `q,k,v,o,ffn.0,ffn.2`, AdamW at `1e-4`, weight decay
`0.01`, the existing ConstantLR behavior, gradient accumulation 1, 121 frames,
and checkpoint interval 273. Do not add a new loss or change sampling.

## Validation and Evaluation

Before full training, require unit tests for exact pair rendering, per-pair
tokenization, masked pooling, frozen UMT5 gradients, cross-attention shape and
CFG isolation, checkpoint topology, and baseline discovery. Then run an
eight-GPU one-step smoke train and one-case generation.

Generate the frozen test jobs with seed 42, 50 denoising steps, CFG 5.0, and the
same canvas/frame settings. Evaluate with the current VPhysBench v10 scene
evaluators. Report strict Task score and coverage, per-scene scores, degraded
or failed cases, generation integrity, and qualitative samples. Compare against
the original symbol/value run only where task, steps, seeds, and evaluation
coverage match; otherwise label the comparison as diagnostic rather than causal.

## Decision Rule

The hypothesis is supported only if the treatment materially improves both
generation validity/subject preservation and VPhysBench physics scores under a
matched budget. A score increase caused solely by missing jobs, degraded-zero
policy differences, or changed evaluator coverage does not count. If visual
quality improves but physics scores do not, conclude that the analytic encoder
may damage text conditioning but is not the sole physical-learning bottleneck.

## Failure Handling

Prediction-side failures retain the v10 conservative zero policy; reference
failures remain unavailable and internal evaluator failures remain errors.
Loss, downstream-conditioner gradients, checkpoint hashes, and UMT5 gradient
counts are audited. The full run is stopped on non-finite loss/gradient,
checkpoint mismatch, or smoke-generation failure, while preserving logs and
partial artifacts.
