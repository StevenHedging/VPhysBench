# Six-scene V15 benchmark sync design

## Goal

Bring the final task protocol and vertical-spring evaluator from branch
`2026-08-12` into the research `main` worktree without deleting integrated
baselines, historical protocols, or provenance.

## Compatibility strategy

The release branch reset public protocol numbering to V1 and removed research
history. Main already contains immutable protocols V1-V14, so copying that
release layout would invalidate old fingerprints. Main will instead publish the
same new behavior as `scene_default_v15` and retain every older file and
evaluator type.

The two release tasks become schema-4 main tasks:

- `six_scene_train_six_scene_eval_v15`: View A, 679 training instances and 96
  evaluation jobs.
- `six_scene_direct_eval_v15`: View B, 775 evaluation jobs.

Both evaluate pendulum, 1D collision, inclined-plane slide, uniform circular
motion, parabolic motion, and vertical spring oscillation. Push-bottle remains
outside the official task.

## Evaluator integration

Add the release branch's `vertical_spring_oscillator_v1` observation, scoring,
and evaluator package. Extend the existing main registry rather than replacing
its historical mappings. Port only shared entity-manifest, frozen-subject, and
robust failure-contract changes required by the new evaluator.

The V15 protocol copies the five current V14 scene definitions byte-for-byte,
then adds the release branch's final vertical-spring configuration. Thus V14
scores remain reproducible while V15 adds one scene.

## Current-version switch

Preserve all versioned task files. Update current official task entry files and
human-facing current-version documentation only where they represent the latest
benchmark. Baseline-specific frozen experiment files continue to reference the
protocol they were designed and trained against.

## Error handling and verification

Prediction-origin media, spatial, and observation failures follow the release
contract and produce evaluated-zero results. Reference failures remain
unavailable, while internal evaluator errors still raise. Artifact-write
failures are recorded in provenance rather than changing physics scores.

Verification covers protocol schema/loading, exact task counts, evaluator
registration, vertical-spring scoring and end-to-end behavior, legacy V14
immutability, task planning, CSTI aggregation, and integrated baseline tests.

