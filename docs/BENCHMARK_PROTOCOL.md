# Benchmark protocol

## Dataset identity

- Dataset ID: `physics_video_seven_scene_v13`
- Release: `13.0.0`
- Cases: 916
- Views: `view_a` and `view_b`
- Bound Hub revision: `datasets/huggingface.json`

Each Case owns its canonical prompt, structured physical quantities, first
frame, masks, reference video, and split annotations. Baselines receive only
the input view authorized by their declared input policy. Reference media and
evaluator annotations are evaluator-only.

## Five scored scenes

The official direct- and finetune-evaluation Tasks score:

1. `pendulum`
2. `collision_1d`
3. `inclined_plane_slide`
4. `uniform_circular_motion`
5. `parabolic_motion`

## Two preview scenes

The Dataset also exposes:

1. `push_bottle`
2. `vertical_spring_oscillator`

These scenes are data-only in this near-release. Their Cases may be inspected
or used for private experiments, but they are excluded from official Task
selection, aggregation, and leaderboard claims.

## Official Tasks

- `tasks/official/five_scene_direct_eval.json`: no training partition; 658
  inference jobs across the five scored scenes.
- `tasks/official/five_scene_finetune_eval.json`: view-A training followed by
  76 held-out inference jobs across the same scenes.

Both Tasks use evaluation protocol `scene_default_v10`. The Task owns scene
selection, seeds, split and reporting policy. It does not own model
conditioning. A baseline owns whether structured physics is ignored, optional
or required.

## Media contract

Managed I2V jobs seal:

- the authorized first-frame asset;
- output canvas width and height;
- FPS and physical time zero;
- a fixed or bounded frame-count rule;
- the output path inside the current run.

Predictions that cannot be decoded or violate the canvas, FPS, start-time or
frame-count contract fail before scene scoring.

## Scoring and coverage

Scene evaluators compute per-case physical-consistency measurements and emit
case results under the run's `evaluation/` tree. Task aggregation is valid only
when canonical inference-job coverage is complete. A partial run may expose
diagnostic observed means, but it is not an official Task score.

Failures remain explicit. Missing predictions, invalid media, unsupported
scenes and evaluator errors are recorded rather than silently replaced with a
successful score.

For evaluator algorithms and scene-specific quantities, see
[EVALUATION.md](EVALUATION.md).
