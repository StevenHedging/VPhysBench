# Causal Forcing++ two-step autoregressive I2V

This Bundle integrates the official ICML 2026 Causal Forcing++ frame-wise
two-step checkpoint as two schema-v5 managed direct-evaluation Baselines:

- `causal_forcing_pp_2step_i2v_generic` uses only the Dataset Case's canonical
  prompt and first-frame asset.
- `causal_forcing_pp_2step_i2v_physics` appends audited structured physical
  quantities to that same canonical prompt as plain English.

The two identities share the same checkpoint, driver, worker, spatial recipe,
temporal recipe and seeds. Evaluator/reference videos are inaccessible to both
models, and raw physics objects are never passed to the worker.

## Why this autoregressive model

The selection was refreshed on 2026-08-01 rather than being limited to the
2025 candidates that existed when MAGI-1 was released.

| Candidate | Autoregressive unit | Native I2V in released code/weights | License | Integration decision |
|---|---|---:|---|---|
| [Causal Forcing++](https://github.com/thu-ml/Causal-Forcing) ([paper](https://arxiv.org/abs/2605.15141)) | one latent frame | yes | Apache-2.0 | selected: newest high-quality two-step checkpoint, 1.3B Wan backbone, practical 8× single-GPU throughput |
| [MAGI-1](https://github.com/SandAI-org/MAGI-1) ([paper](https://arxiv.org/abs/2505.13211)) | 24-pixel-frame chunks | yes | Apache-2.0 | strong physics-oriented alternative, but 4.5B plus a separate T5-XXL stack is slower/heavier for 658 official direct-eval cases |
| [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) ([paper](https://arxiv.org/abs/2504.13074)) | diffusion-forcing chunks | yes | custom model license | not selected because redistribution/use terms are less permissive |
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) ([paper](https://arxiv.org/abs/2506.08009)) | causal chunks | no native released I2V baseline | Apache-2.0 | superseded by Causal Forcing on the same inference budget |
| [NOVA](https://github.com/baaivision/NOVA) ([paper](https://arxiv.org/abs/2409.11305)) | individual non-quantized frames | yes | Apache-2.0 | much lighter, but the released 0.6B model is not competitive with current few-step causal diffusion quality |
| [CausVid](https://github.com/tianweiy/CausVid) ([paper](https://arxiv.org/abs/2412.07772)) | causal chunks | repository still marks I2V checkpoint TODO | code/model terms are less clear | cannot provide a reproducible official I2V deployment |

URSA is a uniform discrete-diffusion model rather than a causal
autoregressive I2V generator; SkyReels-V3 is a multimodal generation suite,
not a released causal-AR I2V architecture. Newer Context Forcing and Diagonal
Distillation releases currently expose T2V inference rather than a verified
first-frame I2V checkpoint, so they are not drop-in candidates for this
Benchmark contract.

The integrated source and weight identities are frozen identically in
`baseline.json` and `physics.baseline.json`:

- Causal Forcing source commit:
  `1fc7bbc19a503c1bce80ecef08158b20e702f386`
- Causal Forcing Hugging Face revision:
  `2f8eb8bb6eeb1238da9d13e5420d342a74d634a6`
- checkpoint: `causal-forcing++/framewise-2step.pt`
- base model: `Wan-AI/Wan2.1-T2V-1.3B`

## Benchmark recipe

The model generates 21 latent frames autoregressively. Wan's temporal VAE
decodes those into 81 pixel frames at 16 FPS. Every generated latent uses two
denoising steps; the first generated latent uses the checkpoint's official
four-step stabilization schedule.

The official demo fixes 832×480 landscape output. Physics Video Benchmark
also contains portrait pendulum, parabolic-motion and uniform-circular-motion
captures, so the worker supports 480×832 portrait
output. Both orientations have the same pixel/token budget and the same 1,560
spatial tokens per latent frame. First frames are resized without cropping
using aspect-preserving contain resize and edge padding.

Jobs are deterministically assigned by job-ID hash. One process is started per
visible GPU, the transformer/text encoder/VAE are loaded once in that process,
and all assigned cases are then generated sequentially. Thus an eight-GPU run
has eight persistent model workers rather than reloading the model 658 times.

## Physics prompt fusion

The physics identity uses the standard `append_structured_text_v1` adapter and
the audited `six_scene_physics_clauses_v2` template. The adapter reads only
`case.physics` values marked `annotated=true`, validates every unit, renders a
deterministic English clause and records the exact values in
`used_parameters` together with their stable symbols. Appearance, background, capture setup and all
parameters marked non-conditionable remain excluded.

For collision cases, balls are mapped from left to right in the initial frame
and every velocity quantity is a non-negative magnitude. Leftward, rightward
and stationary roles come from the canonical Case prompt rather than a numeric
sign convention. Per-ball initial velocities are used instead of the redundant
audit-only `striker_initial_velocity` alias, so opposed two-ball incidents retain
both velocity magnitudes. The final fused prompt is sealed in
`native_inputs.text.prompt`; the shared driver forwards it unchanged to the
model's text encoder.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored
`baseline.local.json` and set the frozen source, checkpoint, base model,
Python environment and cache paths. Validation checks the source Git commit,
the complete Causal Forcing checkpoint SHA-256, all three Wan base-weight
SHA-256 values, tokenizer/config identities, and exact runtime package
versions.

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate \
  causal_forcing_pp_2step_i2v_generic

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate \
  causal_forcing_pp_2step_i2v_physics
```

Compile or dry-run the current five-evaluator-scene task:

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  task-build \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline causal_forcing_pp_2step_i2v_physics \
  --output /tmp/causal_forcing_pp_physics.task.json

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/11.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline causal_forcing_pp_2step_i2v_physics \
  --output-root runs_v2
```

Add `--execute` to generate. `--case-id CASE_ID` restricts an integration
smoke test to one case.
