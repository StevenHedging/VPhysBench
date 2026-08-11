# Vertical Spring Oscillator Evaluator v1 Design

## Goal

Publish a fail-closed expert evaluator for `vertical_spring_oscillator` that:

- produces the shared exact CSTI metric from one frozen steel-ball identity;
- produces an interpretable expert score for vertical spring dynamics;
- follows the same media, robustness, provenance, and artifact contracts as the
  five existing public evaluators;
- does not change either official Task v1 selector and does not modify or remove
  Dataset assets.

## Scope boundary

This change adds a public evaluator type and a scene entry to
`scene_default_v1`. It deliberately does not add spring Cases to the official
five-scene evaluation plans. The current six-scene training/five-scene
evaluation Task remains byte-for-byte semantically unchanged.

The Dataset remains the source of the condition image, frozen subject mask,
physics annotations, and same-Case reference video. Evaluation never writes to
the Dataset tree.

## Considered approaches

### Copy the parabolic compact-ball observer

The ball is visually simple, but copying or importing the parabolic scene's
private tracker would couple two scene implementations and inherit projectile
assumptions about initial position, lifecycle, and possible exit. It would also
produce approximate circle/component masks for CSTI. Rejected.

### Template matching or translated frozen masks

This is deterministic and cheap, but it measures the motion of an appearance
patch rather than the generated object's actual silhouette. CSTI would then
partly score an evaluator-generated mask. Rejected as insufficiently faithful.

### Frozen identity plus the shared SAM2 video segmenter

Use the immutable first-frame steel-ball mask to form a condition-causal prompt,
then segment reference and prediction independently with the existing shared
SAM2 adapter. Apply strict mask quality and temporal coverage checks, and reuse
the exact samples when the two videos are identical. Selected because it is the
smallest design that preserves actual masks, scene symmetry, and the established
expert-evaluator dependency boundary.

## Architecture

### Entity contract

The legacy entity materializer gains one deterministic declaration:

- entity id `oscillator_ball`;
- role `spring_oscillator`;
- class `steel_ball`, matching the frozen mask manifests;
- persistent lifecycle and `spring` as a declared part;
- physical attributes for initial displacement, mass, and radius;
- condition anchors for spring id and release side when present.

The apparatus declaration is `spring_support`, class
`vertical_spring_and_support`, with gravity, natural length, and stiffness
attributes. Same-Case reference video gives the manifest `SAME_CASE_GT`
capability, so CSTI is applicable.

### Observation layer

The scene evaluator loads the frozen subject anchor after shared spatial
normalization. Its bounding box and centroid form a single-object SAM2 prompt.
Reference and prediction use the same prompt policy but are segmented
independently. Identical sampled videos reuse the reference result exactly.

Each mask is normalized to the evaluation canvas and rejected when it is empty,
too small, too large, or implausibly different in area from the frozen anchor.
Unavailable prediction samples are forced to empty masks. A trace contains:

- raw and interpolated centroid coordinates;
- per-frame validity and mask area;
- empirical equilibrium, amplitude, period, and release direction;
- horizontal drift normalized by vertical amplitude;
- mask coverage and observer provenance.

Reference observation failures are `unavailable`. Prediction observation or
identity failures are `evaluated_zero` through `robust_subject_v3`.

### Spring topology observation

For every valid ball mask, Canny edge support is measured in a narrow vertical
corridor from the fixed top support region to the ball's upper boundary. The
observer records row coverage and endpoint support. Comparing the prediction
series against the reference prevents a freely moving ball with no attached
spring from receiving a high expert score. This diagnostic uses only the
current frame and the frozen corridor; it does not use future reference pixels
to localize the prediction.

### Expert dynamics score

The pure scoring module compares both traces on the shared physical timeline,
without DTW or temporal realignment. Components are bounded in `[0, 1]`:

1. vertical displacement trajectory, normalized by the reference amplitude;
2. oscillation period, compared with both the empirical reference and
   `2*pi*sqrt(m/k)`;
3. amplitude and envelope agreement;
4. equilibrium and initial release-side/phase agreement;
5. vertical-axis confinement;
6. observable oscillation evidence, which prevents a static subject from
   earning a structurally plausible score.

The physics score uses explicit protocol weights. It is combined with common
position/shape/appearance subject comparison and spring topology using another
explicit set of weights. Missing coverage and identity ambiguity fail closed;
topology also acts as an integrity factor rather than an optional bonus.

The primary robust metric remains `scene_subject_state_similarity`, while the
scene-specific metric is `vertical_spring_oscillator_similarity` with all
components, weights, physical diagnostics, and failure reasons exposed.

### CSTI

The evaluator builds `CSTIInput` through
`build_csti_input_from_aligned_masks` for exactly one expected entity. It passes
the independently segmented reference and prediction mask tubes and a matched
track id only after the frozen identity/quality gate accepts the prediction.
The base evaluator attaches the existing `exact_full_tube_edt` metric, including
the standard first-three-sample exclusion and full-tube aggregation. No CSTI
algorithm or tolerance changes are introduced.

## Protocol and release surface

- Add `vertical_spring_oscillator_v1` to the lazy registry.
- Add a schema-validated scene configuration to `scene_default_v1`.
- Update protocol wording from “five-scene protocol” to “public scene
  evaluator protocol”; official Task docs continue to state five scored scenes.
- Update release contract tests to distinguish evaluator availability from Task
  selection.
- Keep heavy evaluator imports behind registry resolution, preserving the
  hub-only import boundary and clean-release CI.

## Artifacts and auditability

Each successful Case emits:

- per-frame centroid, validity, displacement, mask area, and topology CSV;
- reference/prediction vertical trajectory plot;
- subject similarity artifacts;
- a JSON audit containing manifest digest, frozen-anchor hashes, segmentation
  metadata, theoretical/empirical periods, scoring components, and failures.

Artifact write failures are recorded without changing the numerical result.

## Verification

Tests are introduced before implementation and cover:

- exact spring entity/apparatus materialization;
- trace extraction and theoretical period;
- identity scoring at one, static prediction degradation, wrong period/phase,
  horizontal drift, missing masks, and topology loss;
- identical and absent CSTI tubes;
- lazy registry routing and protocol/schema contracts;
- official Task plans remaining six-scene train/five-scene evaluation;
- two read-only real-Dataset identity smokes, one release-above and one
  release-below, when full assets and evaluator dependencies are available.

Final verification includes focused tests, evaluation tests, interface/release
tests, compile checks, clean archive checks, push, and hosted Actions status.
