# V14-only frozen-reference evaluation design

## Goal

Create branch `2026-08-17` directly from `2026-08-12` and adapt its single
current (`v1`) task and evaluation surface to Dataset 14.0.0.  Dataset V14 is
the only supported Dataset on this branch.

## Dataset contract

- The current Dataset is `physics_video_seven_scene_v14`, release `14.0.0`,
  descriptor schema `6.0`.
- Every Case must declare `reference_observation_manifest` and
  `reference_observation_visualization_manifest` in addition to the V13
  materialized Case assets.
- Loading is fail closed: missing, malformed, escaping, or hash-mismatched
  frozen observations are Dataset/reference failures.
- The two official v1 tasks bind only to the V14 Dataset ID.

## Evaluation contract

- Reference subject masks, trajectories, visibility states, and identities
  come only from the Case-owned frozen `reference_observation` bundle.
- Frozen samples are selected on the evaluator's physical-time grid and
  transformed through the exact reference spatial transform used for the
  sampled reference video.
- Prediction segmentation, tracking, identity binding, scoring, and media
  validation remain unchanged.
- Reference video decoding remains available for apparatus geometry,
  appearance, topology, visualizations, and media provenance.  Evaluators do
  not invoke SAM2 or a subject tracker to reconstruct the GT tube.
- There is no V13/live-reference fallback.  A Case without a valid frozen
  bundle is unavailable rather than being re-extracted online.

## Architecture

Add a small pure loading/alignment module under evaluation common code.  It
validates and loads the V14 storage contract, aligns entity observations to a
requested time grid and spatial transform, and exposes evaluator-native masks,
centroids, boxes, areas, and states.  Each of the six supported Scene
evaluators replaces only its reference subject-observation branch with this
adapter; its condition/apparatus and prediction branches remain intact.

## Validation

- Dataset schema-6 and V14-only current-Dataset tests.
- Frozen bundle integrity, timeline alignment, crop/resize, state, and entity
  matching tests.
- One regression per supported Scene proving the online reference subject
  extractor is not called for a V14 Case.
- Existing prediction/evaluator and task compilation suites remain green.

