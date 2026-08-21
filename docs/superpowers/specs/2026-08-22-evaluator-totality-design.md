# Evaluator Totality and Reference Preflight Design

## Objective

For every Case selected by an official Task and every legal, decodable
prediction video, evaluation must return `status="evaluated"` with finite
scene and CSTI scores.  Poor generation quality—including missing, duplicated,
merged, crossing, or stationary subjects—must reduce the score and must never
produce `unavailable` or `error`.

`unavailable` remains reserved for a broken Dataset/reference installation.
The official Dataset release must make that state unreachable by passing a
release preflight before it is published or used for a benchmark run.

## Current-State Findings

Historical six-scene runs failed only on the reference side.  Dataset V14 has
already repaired the old collision coverage, invalid mask-ID, spring, and
several pendulum identity defects.  Re-evaluating the five failures shared by
the foreground-loss sweep with current code reduced them to three pendulum
Cases:

- `pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a015deg_img1449`;
- `pendulum_s3_ltot0060mm_lrope0050mm_m031p5g_r010mm_a065deg_img1392`;
- `pendulum_s3_ltot0175mm_lrope0165mm_m031p5g_r010mm_a045deg_img1511`.

Their first-frame masks, frozen tubes, and videos are valid.  The failures are
caused by evaluator defects:

1. production computes `bob_radius / string_length`, while image geometry is
   pivot-to-bob-centre distance and therefore requires
   `bob_radius / (string_length + bob_radius)`;
2. the condition-only V7 observer raises on a near tie before the V8 observer
   can use the frozen bob anchor to resolve it; and
3. the V8 fragmented-string fallback groups all segments only by horizontal
   sign.  Unrelated support/background segments can therefore be fused into a
   false string and a false pivot.

The first complete reference preflight additionally exposed three vertical
spring Cases whose valid fundamental periods were present in the ACF peaks,
but the residual-based selector preferred a fourth or fifth harmonic and only
then rejected it as out of bounds.  Period bounds were therefore being used as
a late error check instead of as part of candidate selection.

A subsequent adversarial preflight, using contract-valid all-black videos for
all 96 jobs, exposed two more reference-only pendulum defects that a canonical
reference preflight did not reliably exercise:

1. two circle proposals with different centres but the same pivot survived the
   anchor gates in `img1369`; because the frozen annotation owns the final bob
   centre and radius, they describe the same final pendulum structure but were
   incorrectly treated as ambiguous; and
2. a weak visual proposal in `img1412` nearly tied the correct structure only
   because its extrapolated length happened to match the physical prior.  The
   internal candidate score assigned 90% to this geometry prior and omitted
   the V7 visual score, allowing geometry-only evidence to manufacture a
   reference failure.

The videos, first-frame masks, and frozen tubes for both Cases are valid.  The
failures are deterministic observer errors, not Dataset annotation defects.

The six official scene configurations already use `robust_subject_v3`.
Prediction media/observation failures are conservatively converted to an
evaluated zero.  The observed holes are reference-analysis failures, not a
missing prediction degradation policy.

## Considered Approaches

### A. Convert every exception to an evaluated zero

This would make result files superficially complete, but a broken GT asset or
evaluator installation would silently penalize a Baseline.  It destroys the
reference/prediction fault boundary and is rejected.

### B. Add manually curated pendulum pivots to the Dataset

This removes runtime pivot perception but expands the Dataset schema and
requires another full-Case visual curation pass.  The current GT tubes are
correct and the three failures have a general algorithmic cause, so a new
manual annotation is unnecessary for this repair.

### C. Repair the general observer and certify references before evaluation

This is the selected approach.  It preserves the evaluator's semantics,
removes the known false failure paths, and makes reference readiness a release
invariant rather than something discovered after an expensive Baseline run.

## Design

### 1. Correct pendulum geometry

Use the total pivot-to-bob-centre length:

```text
expected_radius_length_ratio = r_bob / (l_string + r_bob)
```

Reject missing, non-finite, or non-positive physics exactly as today.  Add a
unit test that calls the production helper so fixtures cannot accidentally use
the correct formula while production uses a different one.

### 2. Give V8 ownership of anchor-aware ambiguity

Separate proposal production from condition-only finality.  V7 keeps its
existing fail-closed default for V7 callers.  Its proposal result can also be
requested without enforcing the condition-only ambiguity gate.  V8 uses that
mode because its frozen subject anchor is stronger identity evidence and must
make the final uniqueness decision.

Weak or missing V7 proposals are not automatically accepted.  They may only
be supplemented by independently supported, anchor-guided line candidates and
must still pass V8 containment, source-agreement, geometry, and uniqueness
gates.

### 3. Replace sign-only string fusion with geometric clustering

Anchor-guided line segments are clustered by signed orientation and distance
to a common line through the frozen bob anchor.  A cluster must have coherent
angle, transverse offset, and useful span/coverage.  Each cluster contributes
its own pivot candidate; candidates are not mixed across clusters.

V8 audits every candidate against the corrected physical radius/length ratio.
The true long string in `img1511` is then distinct from short bob highlights
and from support/background edges.  Provenance records the selected cluster,
supporting segments, coverage, angle spread, and geometry score.

After binding the frozen bob annotation, compare final candidate structures by
their pivots rather than by the detector's provisional circle centres.  The
anchor owns the final bob centre and radius, so same-pivot candidates are
duplicate evidence; genuinely different pivots remain subject to the
fail-closed ambiguity margin.

Rank eligible candidates with 75% physical geometry agreement, 10% frozen
anchor identity evidence, and 15% V7 visual evidence.  Geometry remains the
largest signal, but a low-identity, low-visual proposal can no longer create a
near tie merely by matching the expected length.  These internal observer
weights and the selection policy are recorded in provenance.

### 4. Bound spring period selection before harmonic ranking

Constrain spring period selection to the configured physical interval before
ranking the fundamental/harmonic recurrence chain.  If the shortest
repeat-supported peak (the fundamental) itself is outside the interval, return
no period evidence; never relabel one of its harmonics as the fundamental.
This keeps a valid trace evaluable while allowing the period component to
score zero when no admissible repeat is observable.  Spring scoring tests must
therefore assert `period_s is None` for an unsupported slow trace rather than
expecting a `period_out_of_bounds` exception; harmonic-selection tests that
exercise the residual gate keep every candidate inside the configured window.

### 5. Preserve the fault boundary

- Prediction-side media, segmentation, tracking, or identity failure under
  `robust_subject_v3` returns `evaluated`, score `0.0`, finite zero CSTI, and a
  degradation reason.
- Reference-side corruption remains `unavailable`; it is never translated to
  a Baseline penalty.
- Unexpected implementation exceptions remain `error` so programming defects
  are visible.

The guarantee is obtained by eliminating reference defects before runs, not by
hiding them in result aggregation.

### 6. Add an official-Task reference preflight

Provide a deterministic command that materializes the selected official jobs
and evaluates each Case using its canonical reference video as a known-legal
prediction.  It fails nonzero unless every job is `evaluated`, every primary
score is finite, and every applicable CSTI score is finite.  Its JSON summary
lists every failure and the evaluator fingerprint.

Fast tests cover the historical real pendulum Cases and prediction degradation
contract.  The full 96-job six-scene preflight is the authoritative release and
completion gate.

## Verification Gates

1. New focused tests fail on the unmodified evaluator.
2. Pendulum unit and integration suites pass after the repair.
3. Existing evaluator, dataset-contract, and task-protocol suites pass.
4. The official six-scene preflight returns 96/96 `evaluated`, all finite.
5. A 96-job adversarial preflight using contract-valid all-black videos also
   returns 96/96 `evaluated`, all finite.  This explicitly separates visual
   quality degradation from evaluator/reference failure.
6. Historical pure-LoRA and foreground-sweep predictions are re-evaluated with
   current Dataset/code; each run returns 96/96 `evaluated` and no
   `unavailable`, `error`, or non-finite score.

## Non-Goals

- Do not change published scene/CSTI metric weights or improve generated-video
  quality in this repair.  Internal reference-observer evidence weights may be
  corrected when they cause deterministic reference failures.
- Do not map malformed Dataset/reference assets to zero.
- Do not add Case-ID-specific branches or coordinates.
- Do not rewrite canonical GT tubes that have already passed the V14 audit.
