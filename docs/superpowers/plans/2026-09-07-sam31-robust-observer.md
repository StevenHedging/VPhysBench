# SAM3.1 Robust Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prediction-only SAM segmentation and fixed initial identities more robust without changing CSTI mathematics or hiding errors.

**Architecture:** A small instance compatibility layer protects object birth conditioning; a pure constrained matcher binds frame-zero identities; explicit initial-only proposal gates expose previously hidden detector decisions. Historical inputs and outputs remain outside the source tree.

**Tech Stack:** Python 3.11+, NumPy, SciPy, OpenCV, pinned SAM3.1 multiplex runtime; unittest/pytest for regression.

**Spec:** `docs/superpowers/specs/2026-09-07-sam31-robust-observer-design.md`

## Global Constraints

- No case identifier may select evaluator behavior.
- GT masks participate only in post-detection frame-zero matching, never SAM prompting or retry selection.
- Identity remains fixed after frame zero; initial failure remains null and internal failure remains error.
- Do not modify `common/csti/metric.py`, shared vendor sources, environments, videos or historical scores.
- Only benchmark source, tests, portable configuration and documentation may be committed.
- Preserve legacy configurations and record every new behavior policy in provenance.

### Task 1: Protect multiplex birth conditioning and empty initialization

**Files:**
- Create: `src/physbench/evaluation/common/masks/sam31_compat.py`
- Modify: `src/physbench/evaluation/common/masks/sam31_text.py`
- Create: `tests/test_sam31_compat.py`
- Modify: `tests/test_sam31_text_adapter.py`
- Create: `tests/integration/test_sam31_state_regression.py`

**Interfaces:**
- Consumes: predictor instance with `predictor.model.tracker.model.add_new_masks` and the existing adapter lock.
- Produces: `install_birth_conditioning_fix(predictor) -> bool`, returning whether the compatible instance has the invariant installed; adapter provenance includes `birth_conditioning_policy`.

- [ ] Write RED tests for the real state transition: initial object at frame 5, late object at already-tracked frame 6, remove early object at frame 16, propagate frame 17. Neural forward work may be substituted, but use native state/index/removal logic. Assert surviving inputs and conditioning memory exist; cover shared-frame deletion, remove-all, existing-object updates, idempotence and exception restoration.

```python
assert survivor_id in state['obj_ids']
assert state['output_dict']['cond_frame_outputs']
assert state['mask_inputs_per_obj'][state['obj_id_to_idx'][survivor_id]]
```

- [ ] Run `python -m unittest tests.test_sam31_compat tests.test_sam31_text_adapter -v` and record the expected RED failures, not dependency errors.
- [ ] Implement instance-only wrapping, preserving the bound signature and restoring the flag on failure. The controlling condition is:

```python
new_ids = any(obj_id not in state['obj_id_to_idx'] for obj_id in obj_ids)
force_conditioning = new_ids and not reconditioning
```

- [ ] Add empty-ID early return after initial `consume`, inside existing `try/finally`, and provenance of the installed fix. Existing objects must not use the temporary new-birth policy.
- [ ] Run focused GREEN tests and checkpoint-marked real-vendor state tests; report precise commands. Controller separately executes raw-predictor GPU replay and verifies unchanged vendor hashes.
- [ ] Commit only these files with `fix: preserve SAM3 conditioning for newly born objects` and write the task report.

### Task 2: Threshold-feasible initial matching and auditable failure matrices

**Files:**
- Modify: `src/physbench/evaluation/common/csti/observation.py`
- Modify: `tests/test_csti_observation.py`
- Modify if required: `tests/test_csti_case_integration.py`

**Interfaces:**
- Consumes: candidate tubes, GT frame-zero masks, configured IoU threshold and ambiguity margin.
- Produces: optional `initial_matching_policy` in `CSTIObserverConfig`, legacy default `maximum_total_iou_v1`, new value `threshold_feasible_v2`; existing outcome types with complete initial IoU diagnostics.

- [ ] Write a realizable-mask RED fixture: disjoint GT A and B with 100 pixels each; candidate X contains all A plus 55 B pixels; Y contains 26 A pixels. At threshold 0.25, the feasible policy must select Y for A and X for B, whereas legacy behavior stays unchanged.

```python
assert match.initial_matching == {'a': 'y', 'b': 'x'}
assert match.matching_iou == {'a': 0.26, 'b': 0.275}
```

- [ ] Add RED tests for infeasible Hall conflict, rectangular extra candidates, impossible/all-zero rows, threshold equality, valid second-best ambiguity missed by unconstrained optimization, preserved legacy default, invalid policy names, full matrix diagnostics and immutable later IDs.
- [ ] Run `python -m unittest tests.test_csti_observation -v` and record expected failures.
- [ ] Implement one feasibility-constrained solver reused by best and edge-excluded alternative matching. Infeasibility must become an explicit initialization failure, not an uncaught SciPy exception. Never reuse one candidate for multiple entities.
- [ ] Run `python -m unittest tests.test_csti_observation tests.test_csti_metric tests.test_csti_case_integration -v`; verify unchanged numerical CSTI for identical tubes.
- [ ] Commit with `fix: constrain initial identity assignment to valid IoU edges` and write the task report.

### Task 3: Explicit first-frame proposal gates and versioned public observer

**Files:**
- Modify: `src/physbench/evaluation/common/masks/sam31_text.py`
- Modify: `tests/test_sam31_text_adapter.py`
- Modify: `configs/evaluation/protocols/scene_default_v1.json`
- Modify as required: `schemas/evaluation_protocol.schema.json`, `tests/test_evaluation_protocol_v1.py`
- Create: `docs/SAM31_OBSERVER.md`

**Interfaces:**
- Consumes: controller's fixed-policy calibration result, Task 1 adapter and Task 2 matcher.
- Produces: optional `initial_detection` mapping with validated `score_threshold` and `new_object_threshold`; descriptive observer revision and resolved initial/backend gates in `describe()`; public scene configurations use `threshold_feasible_v2` and explicitly recorded calibration-selected gates.

- [ ] Write RED tests that initial-only gates are visible to the real adapter's initial text request, then restored before propagation, after failed prompt and on a reused predictor. Reject nonfinite/out-of-range thresholds and a birth threshold below the detector threshold. Legacy omitted mapping must not override vendor gates. No prompt request may contain GT input.

```python
assert segmenter.describe()['initial_detection']['score_threshold'] == 0.3
assert model.score_threshold_detection == original_detection_threshold
assert model.new_det_thresh == original_birth_threshold
```

- [ ] Run focused adapter tests and record RED output.
- [ ] Implement one initial-prompt context manager with `finally` restoration. Keep current output probability, prompt semantics, image geometry and temporal propagation unchanged. New behavior cannot branch on GT match outcome.
- [ ] Record observer revision 2 and both initial and propagation gates in configuration/provenance; preserve public protocol entrypoint while documenting that observer revision/config fingerprint differ from legacy results. Preserve the old behavior for configurations without the new optional fields.
- [ ] Add portable documentation of gate meanings, fixed-ID semantics, state fix, errors and calibration limits. Never put generation model names or local deployment paths in committed prose.
- [ ] Run focused adapter, protocol and integration GREEN tests, then controller-run all-video and independent-control validation. A chosen gate must improve correct matching without silently accepting ambiguous identities.
- [ ] Commit with `feat: expose calibrated first-frame SAM candidate policies` and write the task report.

### Task 4: Whole-branch verification and benchmark-only publication

**Files:**
- Review all changed benchmark files; experimental audit/replay/comparison files are run-owned and never staged.

**Interfaces:**
- Consumes: reviewed Tasks 1–3, individual failure audits, real state replay and regression results.
- Produces: verified benchmark-only main worktree at dated branch and matching remote commit; concise handoff with coverage and limitations.

- [ ] Check all historical failure keys are individually accounted for and inspect every failure overlay, including unrecovered examples.
- [ ] Run all 720 initializations with frozen new policy and full-video validation on failures and normal controls; preserve separate old/new artifacts and report remaining failures honestly.
- [ ] Run `make test`, `make smoke-interface`, `make release-check`, relevant evaluation tests and `make release-archive-check`; use the pinned evaluation environment for heavy tests and retain dependency-light interface checks.
- [ ] Generate a whole-branch review package and fix verified findings through the implementer.
- [ ] Recheck main-worktree cleanliness and remote branch state; fast-forward onto `2026-09-07`, push without force, and confirm `git ls-remote` equals the local commit. If unrelated remote work appears, preserve it and integrate non-destructively.
- [ ] Verify no generated assets, baseline code, checkpoints, local environment paths or experiment reports entered the commit. Preserve the isolated worktree until handoff succeeds.
