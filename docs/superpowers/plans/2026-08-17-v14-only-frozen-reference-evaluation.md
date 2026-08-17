# V14-only frozen-reference evaluation implementation plan

**Goal:** Make the `2026-08-17` branch a V14-only evolution of
`2026-08-12`, with frozen Dataset reference observations replacing online GT
subject-tube extraction.

**Architecture:** Extend the current Dataset loader to schema 6, add one
hash-verifying frozen-reference alignment adapter, then route only the
reference half of each supported Scene evaluator through it.  Preserve all
prediction-side algorithms and the branch's unified v1 naming.

---

### Task 1: Bind the branch to Dataset V14

**Files:**
- Modify: `src/physbench/data_layout.py`
- Modify: `src/physbench/datasets/loader.py`
- Modify: `tasks/official/six_scene_direct_eval_v1.json`
- Modify: `tasks/official/six_scene_train_six_scene_eval_v1.json`
- Test: `tests/test_dataset_contract_v6.py`
- Test: `tests/test_task_protocol_20260817.py`

1. Write failing tests for schema-6 required assets and V14-only task IDs.
2. Add the smallest schema-6 loader extension and V14 current pointer.
3. Update both official v1 tasks and make the tests pass.

### Task 2: Add the frozen-reference adapter

**Files:**
- Create: `src/physbench/evaluation/common/frozen_reference.py`
- Test: `tests/test_frozen_reference_observation.py`

1. Write failing tests for manifest/file hashes, Case/Scene/entity identity,
   physical-time selection, spatial crop/resize, and visibility states.
2. Implement strict loading and alignment without importing extraction code.
3. Verify malformed or missing V14 data fails closed.

### Task 3: Route simple single-object evaluators

**Files:**
- Modify: `src/physbench/evaluation/scenes/inclined_plane/evaluator.py`
- Modify: `src/physbench/evaluation/scenes/parabolic_motion/v2_evaluator.py`
- Modify: `src/physbench/evaluation/scenes/vertical_spring_oscillator/evaluator.py`
- Test: focused Scene evaluator tests

Write extractor-not-called regressions first, then replace only reference
subject masks/traces.  Keep apparatus and prediction analysis unchanged.

### Task 4: Route multi-object and topology evaluators

**Files:**
- Modify: `src/physbench/evaluation/scenes/collision/v6_evaluator.py`
- Modify: `src/physbench/evaluation/scenes/circular_motion/v7_evaluator.py`
- Modify: `src/physbench/evaluation/scenes/pendulum/v8_evaluator.py`
- Test: focused Scene evaluator tests

Use frozen entity IDs as authoritative reference identities and construct the
existing evaluator-native reference tracks from aligned masks.  Continue to
derive non-subject apparatus/topology evidence from decoded GT frames.

### Task 5: Freeze the v1 protocol and verify

**Files:**
- Modify: `configs/evaluation/protocols/scene_default_v1.json`
- Modify: evaluator protocol tests and documentation as required

1. Declare the V14 frozen-reference policy in every supported Scene.
2. Run focused Dataset, task, adapter, and six-Scene suites.
3. Run the full test suite and repository diff checks.
4. Commit and push `2026-08-17` to `origin` only after all required checks
   pass.
