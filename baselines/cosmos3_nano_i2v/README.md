# Cosmos3-Nano I2V baseline

This Bundle registers the base `nvidia/Cosmos3-Nano` checkpoint as
`cosmos3_nano_i2v`. It supports `direct_eval × {generic, physics}` for all five
benchmark scenes and requires each Case's immutable first-frame asset.

This is a schema-v4 managed Bundle. The Benchmark-owned compiler and standard
I2V DataAdapter build the canonical task, prompts, audit records and prediction
envelope. `driver.py` contains only Cosmos checkpoint validation, payload
rendering and `torchrun` execution.

The integration uses Cosmos-native resolution/aspect-ratio tokens and valid
`4n+1` frame counts. The default generation is 480p, 24 FPS and 121 frames.
The benchmark evaluator remains responsible for resampling and physical-time
alignment, so generated video resolution or duration is not coupled to the
ground-truth media.

This identity intentionally denotes the base pretrained snapshot at revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`. Nico's separately trained physics
SFT checkpoint is not silently substituted; it should be registered as a
distinct Bundle if it is later selected for comparison.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored `baseline.local.json` and
set the model snapshot, Cosmos framework, Python/torchrun executables and cache
paths. The two lightweight checkpoint identity files are SHA-256 verified
before building a task; the 35 GB checkpoint is not redundantly copied or
rehash-scanned.

Cosmos generation uses all GPU IDs listed in `cuda_visible_devices` for one job
at a time. A protocol dry run does not allocate model GPUs:

```bash
/root/miniconda3/envs/phybench/bin/physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/3.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_physics.json \
  --baseline cosmos3_nano_i2v \
  --scene-id collision_1d \
  --case-id collision_r2_medium_steel_medium_steel_medium_steel_v02818 \
  --output-root runs_v2
```
