# Cosmos3-Nano I2V baselines

This Bundle directory registers two schema-v5 baseline identities backed by
the same `nvidia/Cosmos3-Nano` checkpoint and managed driver:

- `cosmos3_nano_i2v_generic` uses the Case's canonical `text.prompt` and
  explicitly ignores structured physics.
- `cosmos3_nano_i2v_physics` appends the Case's annotated structured physics
  to that same prompt with `five_scene_physics_clauses_v1`.

The distinction is Baseline-owned through `input_policy` and
`adapter.physics_transform`; Tasks no longer select a generic/physics arm.
Both identities otherwise have identical model, runner, spatial and temporal
recipes.

The compiler and standard I2V DataAdapter build and seal canonical inputs.
`driver.py` only validates the Cosmos deployment, renders its inference
payload and invokes `torchrun`. Generation uses Cosmos-native
resolution/aspect-ratio tokens, 480p, 24 FPS and 121 frames. The evaluator
owns video resampling and physical-time alignment.

This identity denotes the base pretrained snapshot at revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`. A separately trained checkpoint
must be registered as another Baseline identity.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored
`baseline.local.json`, then configure the checkpoint, Cosmos framework,
Python/torchrun and cache paths. The single local override is intentionally
shared by both manifests in this directory. The lightweight checkpoint
identity files are SHA-256 verified before task compilation.

```bash
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  cosmos3_nano_i2v_generic
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  cosmos3_nano_i2v_physics
```

Choose the desired Baseline ID when running the same Task:

```bash
/root/miniconda3/envs/phybench/bin/physbench atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline cosmos3_nano_i2v_generic \
  --scene-id collision_1d \
  --output-root runs_v2
```
