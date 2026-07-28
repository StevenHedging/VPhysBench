# WAN2.2 G15 sparse-motion baselines

This Bundle directory registers the frozen epoch-20 G15 LoRA as two
schema-v5 Baseline identities:

- `wan22_g15_sparse_motion_r32_e20_generic` uses the Case's canonical prompt
  and explicitly ignores structured physics.
- `wan22_g15_sparse_motion_r32_e20_physics` appends the Case's annotated
  structured physics with `five_scene_physics_clauses_v1`.

The distinction is owned by each Baseline manifest's `input_policy` and
`adapter.physics_transform`, not by the Task. Both identities share the same
checkpoint, one-line `Wan22ManagedDriver`, runner, spatial and temporal
recipes. They support `direct_eval` only and do not retrain on View A.

## Comparability

G15 was trained before registration on a five-scene corpus that shares source
trials with benchmark v3. The source-aware audit in
`provenance/benchmark_overlap_v3.json` finds 176/214 seen-source cases:
all 121 View A training cases, 8/23 ID test cases and 20/43 OOD1 test cases.
Scores are therefore diagnostic and are not officially comparable to
leakage-free baselines.

The 38 source-unseen cases are the legacy collision, free-fall and pendulum
test cases. A clean-subset report must explicitly name that selection.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored
`baseline.local.json`, then configure the epoch-20 checkpoint, WAN project,
model root and Python environment. The same local override services both
manifests. The frozen checkpoint is verified against the portable SHA-256
before task compilation.

```bash
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  wan22_g15_sparse_motion_r32_e20_generic
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  wan22_g15_sparse_motion_r32_e20_physics
```
