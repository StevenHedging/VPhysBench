# Physics Video Benchmark Dataset 10.0.0

This is the minimal immutable runtime snapshot for
`physics_video_six_scene_v10`. It contains 799 Cases from Dataset 9.0.0
without changing their ordered facts. Each Case adds one locked
`assets.physics_annotation` reference to its Case-local `physics.json`.

Runtime entries:

- `dataset.json`: stable Dataset entry point.
- `release.json`: Dataset and asset-lock digests.
- `cases.jsonl`: ordered Case facts and asset references.
- `assets.lock.json`: size and SHA-256 for every referenced asset.
- `scenes/`: Scene and evaluator contracts.
- `views/`: frozen train/test and direct-evaluation memberships.
- `README.md`: this responsibility map.

Inline `case.physics` remains the runtime compatibility API and must equal
the referenced Case-local physics document exactly. Dataset 9.0.0's
`masks.jsonl` is intentionally absent because masks remain reachable through
each Case's mask manifest and object asset roles, and no runtime consumer uses
the release-wide index. Build and validation evidence is stored separately at
`datasets/provenance/releases/10.0.0/`.
