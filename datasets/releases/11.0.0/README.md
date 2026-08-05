# Physics Video Benchmark Dataset 11.0.0

This is the minimal immutable runtime snapshot for
`physics_video_six_scene_v11`. It preserves all 799 V10 Cases, media assets,
Views, and provenance facts while upgrading structured physics metadata.

Every quantity now has a stable `symbol`. Independent quantities use
`annotated=true` and their symbols appear in the English Case prompt without
numeric values. Derived, calibration, auxiliary, and duplicate-alias
quantities remain available to evaluators with `annotated=false`. Stored scalar
values are finite non-negative magnitudes; motion direction is expressed in the
prompt.

Each Case references a locked `physics.v11.json` document whose physics object
must equal inline `case.physics`. Dataset 10.0.0 and its `physics.json` files
remain immutable. Migration and validation evidence is stored separately at
`datasets/provenance/releases/11.0.0/`.
