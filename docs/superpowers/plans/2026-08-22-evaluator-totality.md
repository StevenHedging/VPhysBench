# Evaluator Totality Implementation Plan

**Goal:** Eliminate reference-side holes from the official six-scene Task and
prove that every legal prediction produces a finite evaluated result.

**Architecture:** Repair the general pendulum condition observer, preserve the
reference/prediction fault boundary, and add a release preflight that evaluates
the official Task with canonical videos as legal predictions.  Follow red,
green, refactor for each production change.

**Runtime:** Python 3.11, NumPy, OpenCV, SciPy, existing scene evaluators and
SAM2 evaluation environment.

---

## Task 1: Correct the pendulum physical geometry ratio

**Files:**

- Modify: `tests/test_pendulum_identity_v8.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v6_evaluator.py`

1. Add a focused test that passes a short-pendulum Case to the production
   `_pendulum_radius_length_ratio` helper and expects `0.01 / 0.06`.
2. Run only that test and record the expected failure (`0.20 != 1/6`).
3. Change the helper to divide by `string_length + bob_radius` and document the
   centre-to-pivot geometry.
4. Re-run the focused test and existing pendulum geometry tests.

## Task 2: Defer V7 ambiguity when V8 has a frozen identity anchor

**Files:**

- Modify: `tests/test_pendulum_open_world_v7.py`
- Modify: `tests/test_pendulum_identity_v8.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v7_open_world.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v8_evaluator.py`

1. Add a unit test proving the V7 default remains fail-closed on a spatial
   near tie.
2. Add a test for proposal mode proving the same hypotheses are returned with
   ambiguity diagnostics instead of raising.
3. Add real-Case integration assertions for `img1392` and `img1511`; confirm
   they fail before implementation.
4. Introduce an explicit proposal/finality option with a fail-closed default.
   V8 alone requests proposal mode and still performs its own absolute gates
   and uniqueness decision.
5. Re-run the focused V7/V8 suites.

## Task 3: Repair anchor-guided fragmented-string fusion

**Files:**

- Modify: `tests/test_pendulum_identity_v8.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v8_open_world.py`
- Modify: `configs/evaluation/protocols/scene_default_v1.json` only if a new
  general threshold is required

1. Add a synthetic regression with two same-sign but non-collinear line
   families; assert the selected pivot comes from the anchor-connected string.
2. Add a real `img1511` assertion that the recovered pivot lies near the long
   visible string endpoint and not at the left image edge.
3. Confirm both tests fail with sign-only grouping.
4. Cluster supported segments by signed orientation and transverse line
   distance.  Emit one audited candidate per coherent cluster and preserve
   cluster diagnostics.
5. Run all pendulum V7/V8 tests and visually inspect the three repaired real
   Case identity overlays.

## Task 4: Constrain spring fundamental-period selection

**Files:**

- Modify: `tests/test_vertical_spring_evaluator.py`
- Modify: `src/physbench/evaluation/scenes/vertical_spring_oscillator/scoring.py`

1. Add a deterministic noisy-oscillator regression whose unconstrained ACF
   residual ranking selects an out-of-bounds harmonic.
2. Confirm the public trace extractor raises `period_out_of_bounds`.
3. Apply the configured physical period bounds while selecting among the
   repeat-supported fundamental and its harmonics; never promote an invalid
   fundamental's harmonic to a valid period.
4. Re-run the spring suite and the affected real reference cases.

## Task 5: Add official-Task reference evaluability preflight

**Files:**

- Create: `src/physbench/evaluation/preflight.py`
- Create: `scripts/evaluation_preflight.py`
- Create: `tests/test_evaluation_preflight.py`
- Modify: `Makefile`
- Modify: `docs/EVALUATION.md`

1. Write unit tests for summary validation: reject non-evaluated statuses,
   null/non-finite primary scores, and null/non-finite applicable CSTI scores;
   accept complete finite results.
2. Run the tests and confirm the module is absent/failing.
3. Implement deterministic prediction materialization from each selected
   Case's canonical reference video, invoke the existing evaluator, and write
   a machine-readable summary.
4. Expose a script command and a `Makefile` target.  Document that this is a
   Dataset/evaluator release gate, not a score-estimation run.
5. Re-run unit tests and script help/smoke tests.

## Task 6: Focused and repository regression

1. Run `tests.test_pendulum_identity_v8` and
   `tests.test_pendulum_open_world_v7`.
2. Run all scene-evaluation and CSTI integration suites.
3. Run dataset contract, current Dataset, task protocol, and interface suites.
4. Inspect `git diff --check`, worktree status, and changed-file scope.

## Task 7: Authoritative 96-Case and historical-run verification

1. Run the new preflight against Dataset 14.0.0 and
   `six_scene_train_six_scene_eval_v1`; require 96/96 evaluated and finite.
2. Re-evaluate the pure-LoRA 96 predictions with the fixed evaluator/current
   Dataset into a new immutable evaluation directory.
3. Re-evaluate the foreground `lambda=0.2` and `lambda=2.0` prediction sets the
   same way.
4. Programmatically assert each result file has exactly 96 rows, all
   `status="evaluated"`, finite scores, and finite applicable CSTI values.
5. Record evaluator fingerprints, output locations, and any score changes.

## Task 8: Review and integration readiness

1. Review the diff against the design's non-goals: no Case-ID branches, no GT
   masking of failures, no metric-weight changes.
2. Run final verification once more from the isolated worktree.
3. Commit the isolated branch with the design, tests, implementation, and
   documentation.  Keep the main benchmark worktree untouched until the
   verified branch is ready for integration.
