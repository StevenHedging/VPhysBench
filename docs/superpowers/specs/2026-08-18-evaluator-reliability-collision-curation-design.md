# Evaluator Reliability and Collision Curation Design

## Scope

Harden the current `scene_default_v1` evaluator against the seven reference-side
failures observed in the DisCa evaluation, and manually audit every collision
test case selected by `six_scene_train_six_scene_eval_v1`. The work stays on the
current protocol and Dataset V14 identifiers. It does not create a new release
or a case-specific runtime exception.

The collision curation scope is the 20 canonical `collision_1d` evaluation jobs
in that Task. Training cases are not part of this pass.

## Considered approaches

1. Lower the existing reference coverage and proposal thresholds. This is
   rejected because it can turn an unavailable case into a silently incorrect
   score without establishing identity or physical observability.
2. Re-segment every reference video with SAM2. This is rejected because the
   Dataset already contains reviewed frozen masks, and SAM2 cannot by itself
   resolve transparent-ball identity, complete occlusion, or apparatus geometry.
3. Make reviewed frozen annotations authoritative, use lifecycle-aware evidence
   gates, and repair only demonstrably defective assets. This is the selected
   design because it separates Dataset evidence from prediction observation and
   makes every fallback auditable.

## Evaluator design

### Frozen-reference authority

When `reference_observation_policy` is
`frozen_dataset_reference_observation_v1`, a reviewed frozen entity observation
is the authority for reference identity and masks. A colour, Hough-circle, or
condition proposal detector may supply ancillary geometry, but failure of that
detector must not invalidate an otherwise sufficient frozen observation.

This does not weaken prediction evaluation. Prediction masks and identities
remain independently observed and fail closed under the existing robustness
policy.

### Collision lifecycle gate

Collision reference availability is based on metric sufficiency, not the ratio
of frames marked `visible` over the entire clip. `occluded` is a resolved
lifecycle state, while `unresolved` is missing evidence. A frozen collision
entity must:

- be visible in frame zero for identity binding;
- have at least three visible samples so a two-frame velocity window can be
  supported;
- have no unresolved run that removes all usable motion evidence around every
  contact in which it participates.

The evaluator records visible, resolved-occluded, and unresolved counts
separately. It never treats an occluded sample as a measured centroid. If the
reference is too sparse to extract its N-body state, it remains unavailable
rather than receiving an invented trajectory.

### Pendulum condition geometry

The frozen first-frame bob mask owns bob identity, center, and scale. Visual
circle proposals may not override it. Pivot selection ranks all plausible
upward-string proposals by their geometry relative to the frozen bob anchor and
the declared radius/length and initial-angle constraints. Proposal circle
containment is diagnostic rather than a terminal gate. A selected pivot must
still satisfy physical length, upward direction, string-edge support, and a
unique-margin check. If condition pixels remain ambiguous, the reviewed frozen
reference bob trajectory supplies a reference-only pivot fit; that fit is
audited and never derived from prediction pixels.

### Circular apparatus geometry

The frozen object tube is loaded before apparatus detection. The normal
adaptive-hue apparatus detector remains the primary path. If it fails, a
reference-only fallback fits the orbit center from reviewed frozen centroids,
then constrains an edge-based disk-radius search around that center. The fitted
geometry must meet finite residual, in-frame, support, and radius bounds before
it can be shared with prediction observation. There is no unconstrained
full-frame fallback.

## Collision Dataset audit

Each of the 20 official collision test cases receives a human audit of:

- frame-zero entity identity and first-frame anchor alignment;
- pre-contact, contact, and post-contact masks;
- identity continuity through overlaps and crossings;
- visible, occluded, out-of-frame, and unresolved lifecycle states;
- centroid continuity and physically plausible one-dimensional ordering;
- consistency between the contact sheet, trajectory plot, and full overlay
  video.

Scripts may generate case-specific candidate masks or tracks, but their output
is not accepted automatically. Corrections are made only after visual review of
the relevant frames. Temporary scripts and intermediate assets are deleted.
Permanent changes include only corrected canonical observation assets, their
hashes/manifests, regenerated quality and visualization artifacts, and explicit
human-review records.

The steel case with only one visible sample cannot be repaired by interpolation.
It is either re-annotated from verifiable source pixels or remains unavailable.
The glass case must not label tracker uncertainty as physical occlusion; any
recoverable transparent-ball trajectory is re-segmented and reviewed.

## Verification

- Add red tests that reproduce the current circular, pendulum, and collision
  failures before changing production code.
- Run focused scene evaluator tests after every fix.
- Run Dataset structural validation, frozen-observation QC, and asset-lock
  verification after any asset correction.
- Re-evaluate all 20 collision jobs and the seven previously unavailable jobs.
- Run the full 96-job Task and require no new unavailable/error cases.
- Compare scores for previously evaluated cases to detect unintended evaluator
  drift; any material change must be explained by the new reference evidence.
