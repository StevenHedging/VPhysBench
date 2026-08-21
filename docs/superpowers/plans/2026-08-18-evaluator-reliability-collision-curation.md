# Evaluator Reliability and Collision Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable reference-side evaluator failures and produce manually verified frozen observations for all 20 official collision test cases.

**Architecture:** Reviewed frozen reference observations are authoritative for Dataset-side identity. Scene-specific geometry detectors become constrained ancillary evidence, while collision availability becomes lifecycle- and metric-aware. Dataset corrections use temporary per-case candidate generation followed by manual review and canonical artifact regeneration.

**Tech Stack:** Python 3.11, NumPy, OpenCV, SAM2 only where visually justified, unittest, PhysBench Dataset V14 contracts.

**Spec:** `docs/superpowers/specs/2026-08-18-evaluator-reliability-collision-curation-design.md`

## Global Constraints

- Keep `scene_default_v1`, Task IDs, and Dataset V14 identifiers unchanged.
- Do not add permanent case-ID branches to evaluator code.
- Do not infer reference trajectories through visually unresolved intervals.
- Temporary case-specific scripts and intermediate files must be deleted.
- Every permanent mask/track correction requires human visual review.
- Preserve unrelated working-tree changes and `.local/`.

---

### Task 1: Freeze the failure inventory and collision audit ledger

**Files:**
- Create: `docs/audits/collision_v14_official_test_audit_20260818.md`
- Test: `tests/test_current_dataset.py`

**Interfaces:**
- Consumes: canonical jobs from `plan_atomic_task(...)`.
- Produces: a 20-case ledger with QC values, reviewer decision, defect class, and disposition.

- [ ] Extract the exact 20 collision case IDs from the canonical plan.
- [ ] Add a test asserting that the ledger contains every canonical collision job exactly once.
- [ ] Run the test and verify that it fails while the ledger is absent.
- [ ] Create the ledger and populate machine-readable facts from each frozen observation.
- [ ] Manually inspect every contact sheet, trajectory plot, and overlay video; record the visual decision.
- [ ] Run the ledger coverage test and verify that it passes.

### Task 2: Make collision reference sufficiency lifecycle-aware

**Files:**
- Modify: `src/physbench/evaluation/scenes/collision/v5_evaluator.py`
- Modify: `configs/evaluation/protocols/scene_default_v1.json`
- Test: `tests/test_collision_v5_evaluator.py`
- Test: `tests/test_collision_v6_evaluator.py`

**Interfaces:**
- Consumes: frozen entity `visible`, `occluded`, and unresolved states.
- Produces: `validate_frozen_collision_evidence(...)` with per-entity audit diagnostics.

- [ ] Write a failing test in which 12 visible plus 60 resolved-occluded samples pass the reference gate.
- [ ] Write a failing test in which one visible plus 94 unresolved samples remains unavailable.
- [ ] Run both tests and confirm the first fails for the old 0.2 ratio while the second already fails for insufficient evidence.
- [ ] Implement the minimum-visible-sample and unresolved-evidence gate without filling hidden centroids.
- [ ] Add the gate diagnostics to scene analysis artifacts.
- [ ] Run collision evaluator, N-body, CSTI, and visualization tests.

### Task 3: Anchor-first pendulum condition geometry

**Files:**
- Modify: `src/physbench/evaluation/scenes/pendulum/v8_open_world.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v8_evaluator.py`
- Test: `tests/test_pendulum_identity_v8.py`
- Test: `tests/test_scene_evaluation.py`

**Interfaces:**
- Consumes: `PendulumSubjectAnchor`, V7 pivot hypotheses, frozen bob centroids, and Case physics.
- Produces: an audited `AnnotatedConditionDecision` whose bob identity always comes from the anchor.

- [ ] Add one failing regression for each of the four real pendulum condition frames.
- [ ] Confirm the failures reproduce missing, misaligned, and ambiguous proposal states.
- [ ] Rank pivot hypotheses relative to the anchor rather than requiring proposal-circle ownership of the bob.
- [ ] Add a constrained frozen-trajectory pivot fallback with residual and geometry gates.
- [ ] Verify the four cases evaluate and existing adversarial identity tests remain fail closed.

### Task 4: Frozen-track fallback for circular apparatus geometry

**Files:**
- Modify: `src/physbench/evaluation/scenes/circular_motion/open_world.py`
- Modify: `src/physbench/evaluation/scenes/circular_motion/v6_evaluator.py`
- Test: `tests/test_circular_open_world_v6.py`
- Test: `tests/test_scene_evaluation.py`

**Interfaces:**
- Consumes: frozen object centroids/masks and the condition frame.
- Produces: a validated `FrozenCircularApparatus` from primary colour geometry or constrained reference-track fallback.

- [ ] Add a failing regression using `circular_r1_wood04cm_img_0390`.
- [ ] Confirm adaptive-hue apparatus discovery reproduces the current failure.
- [ ] Implement orbit-center fitting and center-constrained disk-edge selection.
- [ ] Reject fallback geometry with insufficient arc, excessive residual, or invalid disk support.
- [ ] Verify the real case and existing hue/adversarial apparatus tests.

### Task 5: Manually correct defective collision observations

**Files:**
- Modify only as needed: `datasets/assets/collision_1d/**/canonical/masks/**`
- Modify only as needed: `datasets/assets/collision_1d/**/canonical/reference_observation/**`
- Modify as required: `datasets/releases/14.0.0/assets.lock.json`
- Modify: `docs/audits/collision_v14_official_test_audit_20260818.md`

**Interfaces:**
- Consumes: visually approved candidate masks and lifecycle states.
- Produces: canonical frozen observation manifests and regenerated audit visualizations.

- [ ] Inspect frame-zero anchors for all 20 cases at native resolution.
- [ ] Inspect pre-contact/contact/post-contact frames and label every defect.
- [ ] For each defective case, generate candidate masks with the most reliable per-case combination of colour, edge, template, optical-flow, or SAM2 prompts.
- [ ] Review candidates visually frame by frame; edit or reject inaccurate intervals.
- [ ] Regenerate trajectories, quality reports, contact sheets, overlay videos, manifests, and hashes.
- [ ] Delete temporary scripts and intermediate assets.
- [ ] Run Dataset validation and verify every changed asset against the lock.

### Task 6: End-to-end evaluation verification

**Files:**
- Modify if required: `docs/audits/collision_v14_official_test_audit_20260818.md`
- Produce outside the release tree: a new evaluation run directory under `/public/vphysbench-cluster/runs/`.

**Interfaces:**
- Consumes: corrected evaluator and Dataset assets plus existing DisCa predictions.
- Produces: a 96-job evaluation report and before/after failure comparison.

- [ ] Re-evaluate the seven formerly unavailable cases first.
- [ ] Require that repaired cases are evaluated and genuinely unobservable cases remain explicitly unavailable.
- [ ] Re-evaluate all 20 collision jobs and inspect their diagnostic visualizations.
- [ ] Re-evaluate all 96 canonical jobs using the validated existing predictions.
- [ ] Compare coverage, status counts, scene CSTI, and per-case score drift against the prior run.
- [ ] Run focused and repository regression suites, compileall, JSON validation, Dataset validation, and `git diff --check`.
- [ ] Report exact remaining failures and do not claim completion if any unexplained regression remains.
