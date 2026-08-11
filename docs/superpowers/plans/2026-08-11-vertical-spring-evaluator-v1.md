# Vertical Spring Oscillator Evaluator v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed vertical spring oscillator expert evaluator that emits the shared exact CSTI metric and an auditable scene-specific physics/subject/topology score without changing official Task v1 selection or Dataset assets.

**Architecture:** Extend the common entity contract with one frozen steel-ball identity, then keep observation, pure spring-dynamics scoring, and evaluator orchestration in separate scene files. The evaluator reuses shared media normalization, frozen-mask loading, SAM2 video segmentation, subject comparison, CSTI construction, and robustness behavior; the registry remains lazy.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, SciPy-compatible evaluator environment, existing SAM2 adapter, unittest/pytest, JSON Schema, Make/GitHub Actions.

## Global Constraints

- Do not modify either official Task v1 selector, Task counts, Dataset metadata, Distribution binding, or any Dataset asset.
- Keep `vertical_spring_oscillator` train-only in the official six-scene-train/five-scene-evaluation Task while making its evaluator publicly resolvable.
- Use `evaluator_contract=robust_subject_v3`: reference defects are unavailable; prediction observation defects are evaluated zero.
- Use the existing `exact_full_tube_edt` CSTI configuration unchanged and provide actual aligned ball masks.
- Do not import OpenCV, NumPy, SciPy, torch, SAM2, or a scene evaluator at the public package/registry module import boundary.
- All scoring functions return finite values in `[0, 1]`; no DTW, future-reference localization, or missing-frame reward is allowed.
- Every behavior change follows RED-GREEN-REFACTOR and each task ends in a focused commit.

---

## File map

- `src/physbench/evaluation/common/entities/manifest.py`: deterministic spring entity and apparatus declarations.
- `src/physbench/evaluation/scenes/vertical_spring_oscillator/observation.py`: frozen prompt, mask validation, identity gate, and spring-edge topology observation.
- `src/physbench/evaluation/scenes/vertical_spring_oscillator/scoring.py`: immutable trace model, period extraction, and pure expert scoring.
- `src/physbench/evaluation/scenes/vertical_spring_oscillator/evaluator.py`: lifecycle orchestration, SAM2 calls, subject composition, artifacts, and CSTI input.
- `src/physbench/evaluation/registry.py`: lazy evaluator resolution only.
- `configs/evaluation/protocols/scene_default_v1.json`: thresholds and explicit weights.
- `schemas/evaluation_protocol.schema.json`: sixth public scene configuration contract.
- `tests/test_vertical_spring_scoring.py`: pure synthetic trace/scoring coverage.
- `tests/test_vertical_spring_evaluator.py`: observation, robust evaluator, CSTI, and read-only real-asset smoke coverage.
- `tests/test_entity_manifest.py`: materializer contract.
- `tests/test_release_v1_contract.py`, `tests/test_evaluation_protocol_v1.py`: public protocol/registry boundary and unchanged Task selection.
- `docs/EVALUATION.md`, `docs/BENCHMARK_PROTOCOL.md`: public behavior and Task/evaluator distinction.

---

### Task 1: Materialize the spring physical identity

**Files:**
- Modify: `src/physbench/evaluation/common/entities/manifest.py`
- Modify: `tests/test_entity_manifest.py`

**Interfaces:**
- Consumes: Dataset v5 grouped physics through `_case_physics(case)`.
- Produces: `materialize_entity_manifest(case) -> EntityManifest` with one `steel_ball` entity and one `vertical_spring_and_support` apparatus.

- [ ] **Step 1: Write the failing materializer tests**

Add a minimal grouped-physics Case and assert the canonical declarations:

```python
def test_vertical_spring_materializes_frozen_ball_and_apparatus(self):
    case = spring_case()
    manifest = materialize_entity_manifest(case)
    self.assertEqual("oscillator_ball", manifest.entities[0].entity_id)
    self.assertEqual("spring_oscillator", manifest.entities[0].role_id)
    self.assertEqual("steel_ball", manifest.entities[0].entity_class)
    self.assertEqual(("spring",), manifest.entities[0].parts)
    self.assertEqual(
        {"initial_displacement", "mass", "radius"},
        {value.name for value in manifest.entities[0].physical_attributes},
    )
    self.assertEqual(
        "vertical_spring_and_support",
        manifest.apparatus[0].apparatus_class,
    )
    self.assertIs(manifest.reference_capability, ReferenceCapability.SAME_CASE_GT)
```

Also assert missing required mass, radius, displacement, stiffness, natural
length, or gravity raises `ValueError` instead of guessing.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_entity_manifest.py -k vertical_spring -q
```

Expected: failure containing `has no legacy entity materializer`.

- [ ] **Step 3: Add deterministic entity and apparatus branches**

Add this entity branch to `_legacy_scene_entities`:

```python
if scene_id == "vertical_spring_oscillator":
    return _single_entity_defaults(
        case,
        entity_id="oscillator_ball",
        role_id="spring_oscillator",
        entity_class="steel_ball",
        bindings=(
            ("initial_displacement", "initial_displacement"),
            ("mass", "oscillator_mass"),
            ("radius", "ball_radius"),
        ),
        lifecycle=LifecyclePolicy.PERSISTENT,
        parts=("spring",),
        appearance_bindings=(
            ("spring_id", "spring_id"),
            ("release_side", "release_side"),
        ),
    )
```

Add a `_default_apparatus` branch using `_attributes_from_case` for
`natural_spring_length`, `spring_stiffness`, and `gravity_acceleration`, all
required, with selector `fixed_support_and_vertical_spring_axis`.

- [ ] **Step 4: Run the entity suite and confirm GREEN**

Run:

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_entity_manifest.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/physbench/evaluation/common/entities/manifest.py tests/test_entity_manifest.py
git commit -m "feat: declare vertical spring entity contract"
```

---

### Task 2: Implement pure spring trace extraction and expert scoring

**Files:**
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/__init__.py`
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/scoring.py`
- Create: `tests/test_vertical_spring_scoring.py`

**Interfaces:**
- Produces: `SpringTrace`, `extract_spring_trace(...)`, `theoretical_period_s(...)`, and `score_spring_traces(...) -> dict[str, Any]`.
- `SpringTrace` fields: `times_s`, `xy`, `area_px2`, `valid`, `valid_ratio`, `equilibrium_y_px`, `amplitude_px`, `envelope_px`, `period_s`, `horizontal_drift_ratio`, and `release_sign`.

- [ ] **Step 1: Write RED tests from analytic sinusoidal masks**

Generate circle masks with
`y(t)=equilibrium + amplitude*cos(2*pi*t/period)` and assert:

```python
trace = extract_spring_trace(masks, times, quality_config=QUALITY)
self.assertAlmostEqual(period, trace.period_s, delta=1 / 24)
self.assertAlmostEqual(amplitude, trace.amplitude_px, delta=2.0)
self.assertGreater(trace.release_sign, 0)
```

Add comparisons asserting identity is `1.0`, a static tube is below `0.25`, a
wrong period is below the matching period, an opposite release phase is below
the correct phase, horizontal oscillation loses verticality, missing masks fail
the required-valid-ratio check, and
`theoretical_period_s(mass_kg=0.5156, stiffness_n_m=32.6213467096774)` matches
`2*pi*sqrt(m/k)`.

- [ ] **Step 2: Run the new test module and confirm RED**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_vertical_spring_scoring.py -q
```

Expected: import failure for the missing spring scoring module.

- [ ] **Step 3: Implement immutable trace extraction**

Use mask centroids, explicit valid-area bounds, linear interpolation only after
the minimum valid ratio passes, a robust median equilibrium, 5/95-percentile
half range amplitude, absolute-displacement envelope, and bounded
autocorrelation lag search:

```python
@dataclass(frozen=True)
class SpringTrace:
    times_s: np.ndarray
    xy: np.ndarray
    area_px2: np.ndarray
    valid: np.ndarray
    valid_ratio: float
    equilibrium_y_px: float
    amplitude_px: float
    envelope_px: np.ndarray
    period_s: float | None
    horizontal_drift_ratio: float
    release_sign: int
```

Reject non-monotonic time, shape mismatch, fewer than three observations,
insufficient valid ratio, amplitude below `minimum_amplitude_px`, and period
outside configured `minimum_s`/`maximum_s` with `SpringTraceError(code, message)`.

- [ ] **Step 4: Implement bounded component scoring**

`score_spring_traces` must return:

```python
{
    "score": float,
    "components": {
        "vertical_trajectory": float,
        "period": float,
        "amplitude_envelope": float,
        "equilibrium_release_phase": float,
        "vertical_axis_confinement": float,
        "oscillation_evidence": float,
    },
    "weights": dict[str, float],
    "diagnostics": {
        "reference_period_s": float | None,
        "prediction_period_s": float | None,
        "theoretical_period_s": float,
        "reference_amplitude_px": float,
        "prediction_amplitude_px": float,
    },
}
```

Normalize weights, use exponential similarities, compare positions on the exact
common timeline, and define the period component as the geometric mean of
prediction-vs-reference and prediction-vs-theory similarity. A missing
prediction period scores zero.

- [ ] **Step 5: Run focused scoring tests and confirm GREEN**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_vertical_spring_scoring.py -q
```

Expected: all tests pass and no runtime warnings.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/physbench/evaluation/scenes/vertical_spring_oscillator tests/test_vertical_spring_scoring.py
git commit -m "feat: score vertical spring dynamics"
```

---

### Task 3: Implement condition-causal mask and spring-topology observation

**Files:**
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/observation.py`
- Create: `tests/test_vertical_spring_evaluator.py`

**Interfaces:**
- Consumes: `FrozenSubjectAnchor`, `MaskPrompt`, normalized BGR frames, masks, and availability.
- Produces: `prompt_from_anchor(anchor) -> MaskPrompt`, `validate_mask_tube(...) -> tuple[np.ndarray, ...]`, `validate_prediction_identity(...) -> dict[str, Any]`, and `observe_spring_topology(...) -> SpringTopology`.

- [ ] **Step 1: Write RED observation tests**

Construct a frozen circular anchor and synthetic frames containing a vertical
zig-zag spring. Assert the prompt box contains the mask with one positive
centroid point; identity accepts the exact mask and rejects empty, displaced,
or extreme-area masks; unavailable frames become empty; topology is near one
for a spring reaching the ball and near zero after removing the corridor edges.

```python
decision = validate_prediction_identity(
    anchor_mask=anchor,
    prediction_mask=prediction,
    config=IDENTITY,
)
self.assertTrue(decision["accepted"])
self.assertGreater(decision["anchor_iou"], 0.99)
```

- [ ] **Step 2: Run focused observation tests and confirm RED**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_vertical_spring_evaluator.py -k 'prompt or identity or topology' -q
```

Expected: import failure for `observation`.

- [ ] **Step 3: Implement prompt and fail-closed mask validation**

Build an expanded bounding box from `cv2.boundingRect(anchor.mask)`, clamp it to
the canvas, and use the anchor centroid as the positive point. Normalize every
mask to binary uint8, reject wrong canvases and special values, filter by
minimum pixels, maximum frame-area ratio, anchor-area ratio, and availability.
The identity decision reports IoU, centroid distance in anchor radii, area
ratio, accepted status, and exact threshold provenance.

- [ ] **Step 4: Implement topology observation**

For each valid mask, compute Canny edges in a corridor of
`corridor_half_width_radius_ratio * equivalent_radius` centered on the ball.
Measure the fraction of rows with at least `minimum_edge_pixels_per_row` and
whether the last `endpoint_height_radius_ratio` above the ball contains edge
support. Return immutable arrays and their valid means:

```python
@dataclass(frozen=True)
class SpringTopology:
    row_coverage: np.ndarray
    endpoint_support: np.ndarray
    valid: np.ndarray
    score: float
```

- [ ] **Step 5: Run focused observation tests and confirm GREEN**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_vertical_spring_evaluator.py -k 'prompt or identity or topology' -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/physbench/evaluation/scenes/vertical_spring_oscillator/observation.py tests/test_vertical_spring_evaluator.py
git commit -m "feat: observe spring oscillator identity"
```

---

### Task 4: Orchestrate the evaluator, CSTI, registry, and protocol

**Files:**
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/evaluator.py`
- Modify: `src/physbench/evaluation/scenes/vertical_spring_oscillator/__init__.py`
- Modify: `src/physbench/evaluation/registry.py`
- Modify: `configs/evaluation/protocols/scene_default_v1.json`
- Modify: `schemas/evaluation_protocol.schema.json`
- Modify: `tests/test_vertical_spring_evaluator.py`
- Modify: `tests/test_release_v1_contract.py`
- Modify: `tests/test_evaluation_protocol_v1.py`

**Interfaces:**
- Produces: `VerticalSpringOscillatorCaseEvaluator(config)` implementing `SceneCaseEvaluator` and registry type `vertical_spring_oscillator_v1`.
- Consumes: Tasks 1-3 plus `Sam2VideoSegmenter`, `compare_subjects`, `compose_subject_and_state_score`, and `build_csti_input_from_aligned_masks`.

- [ ] **Step 1: Add RED evaluator and release-contract tests**

Use a fake segmenter returning deterministic synthetic masks. Assert a direct
`analyze` identity case returns score one, CSTI input contains one
`oscillator_ball` tube and one matched id, identity rejection yields no matched
prediction tube, identical sampled videos call segmentation once, prediction
segmentation errors become `SceneAnalysisError`, and reference errors become
`ReferenceAnalysisError`.

Update release assertions so the protocol and registry contain:

```python
"vertical_spring_oscillator": "vertical_spring_oscillator_v1"
```

while `SCORED_SCENES`, 658 direct jobs, 679 training Cases, and 76 fine-tune
evaluation jobs remain unchanged.

- [ ] **Step 2: Run protocol/evaluator tests and confirm RED**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_vertical_spring_evaluator.py \
  tests/test_evaluation_protocol_v1.py \
  tests/test_release_v1_contract.py -q
```

Expected: unknown evaluator type or missing protocol scene.

- [ ] **Step 3: Implement the evaluator lifecycle**

Define:

```python
class VerticalSpringOscillatorCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "vertical_spring_oscillator_expert"
    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"
    scene_id = "vertical_spring_oscillator"
    primary_score = "vertical_spring_oscillator_similarity"
    allow_partial_prediction = True
```

In `analyze`, validate the Task 1 manifest, load the frozen `steel_ball` anchor,
segment reference and prediction independently, force unavailable prediction
masks empty, validate identity, extract Task 2 traces, observe Task 3 topology,
run common subject comparison, combine explicit content weights, and apply the
integrity factor. Convert reference-side exceptions to
`ReferenceAnalysisError`; convert prediction-side failures to
`SceneAnalysisError` so `robust_subject_v3` returns evaluated zero.

Build CSTI only for `SAME_CASE_GT`; pass the prediction tube and
`("bound_spring_ball",)` only after identity acceptance. Attach metrics,
quality, provenance, CSV/JSON/curve artifacts, and segmentation metadata.

- [ ] **Step 4: Add the lazy registry route and protocol configuration**

Import the evaluator only inside `SceneEvaluatorRegistry.resolve`. Add a schema
definition requiring `type=vertical_spring_oscillator_v1`, timeline, spatial,
SAM2, quality, period, scoring, subject scoring, identity, topology, and content
weights. Use 24 fps, 480x832 maximum no-pad normalization, the existing tiny
SAM2 model, minimum valid ratio 0.90, and explicit normalized weights from the
design.

- [ ] **Step 5: Run focused and boundary tests and confirm GREEN**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_vertical_spring_scoring.py \
  tests/test_vertical_spring_evaluator.py \
  tests/test_entity_manifest.py \
  tests/test_evaluation_protocol_v1.py \
  tests/test_release_v1_contract.py -q
make test-interface
```

Expected: all pass; the hub-only interface process does not import NumPy,
OpenCV, SciPy, torch, SAM2, or the evaluator implementation.

- [ ] **Step 6: Commit Task 4**

```bash
git add configs/evaluation/protocols/scene_default_v1.json \
  schemas/evaluation_protocol.schema.json \
  src/physbench/evaluation/registry.py \
  src/physbench/evaluation/scenes/vertical_spring_oscillator \
  tests/test_vertical_spring_evaluator.py \
  tests/test_release_v1_contract.py tests/test_evaluation_protocol_v1.py
git commit -m "feat: evaluate vertical spring oscillator v1"
```

---

### Task 5: Calibrate, document, review, and publish

**Files:**
- Modify: `tests/test_vertical_spring_evaluator.py`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/BENCHMARK_PROTOCOL.md`
- Modify: `docs/superpowers/plans/2026-08-11-vertical-spring-evaluator-v1.md`

**Interfaces:**
- Consumes: complete evaluator and read-only assets at `/root/Steven/VPhysBench/datasets/assets` when present.
- Produces: calibrated thresholds, public documentation, review evidence, pushed commit, and successful hosted CI.

- [ ] **Step 1: Add guarded real-asset identity smokes**

Select one `release_side=above` and one `release_side=below` Case from the
canonical Dataset. When full videos, frozen masks, SAM2 dependencies, and model
weights are available, evaluate each reference video as its own prediction and
assert status `evaluated`, total score at least `0.98`, scene physics score at
least `0.98`, and CSTI exactly `1.0`. Otherwise skip with the exact missing
dependency/asset reason.

- [ ] **Step 2: Run calibration and adjust only protocol thresholds**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_vertical_spring_evaluator.py -m real_assets -vv
```

Run at least one additional non-identity perturbation test (static, wrong
frequency, horizontal motion, or removed spring) and preserve the ordering
assertions from Task 2. Threshold changes must remain in protocol JSON; do not
add Case-specific ids or pixel coordinates.

- [ ] **Step 3: Update public documentation**

Document the six publicly resolvable scene evaluators, the spring expert
components, exact CSTI input, failure semantics, and artifacts. State explicitly
that official Task v1 still evaluates only five scenes and retains spring data
for training/future Task versions.

- [ ] **Step 4: Run the full relevant verification matrix**

```bash
/root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_vertical_spring_scoring.py tests/test_vertical_spring_evaluator.py -q
make test-evaluation
make test-interface
make test-release
make check-release-archive
/root/Steven/.venvs/wan22-pair-text/bin/python -m compileall -q src tests
git diff --check
```

Expected: every command exits zero. Record exact pass/skip counts before making
any completion claim.

- [ ] **Step 5: Request an independent code review and address findings**

Use `superpowers:requesting-code-review`. The reviewer must inspect numerical
correctness, CSTI identity binding, reference/prediction exception origin,
future-reference leakage, lazy imports, Task invariance, and release scope.
Resolve every Critical/Important finding with a focused RED-GREEN regression;
re-run the affected matrix.

- [ ] **Step 6: Commit documentation and calibrated tests**

```bash
git add docs/EVALUATION.md docs/BENCHMARK_PROTOCOL.md \
  docs/superpowers/plans/2026-08-11-vertical-spring-evaluator-v1.md \
  tests/test_vertical_spring_evaluator.py \
  configs/evaluation/protocols/scene_default_v1.json
git commit -m "docs: publish vertical spring evaluation contract"
```

- [ ] **Step 7: Verify, push, and inspect hosted checks**

Use `superpowers:verification-before-completion` and
`superpowers:finishing-a-development-branch`, then:

```bash
git status --short
git push origin 2026-08-11
gh run list --branch 2026-08-11 --limit 5
gh run watch <new-run-id> --exit-status
```

Expected: clean worktree, remote branch equals local HEAD, and the new GitHub
Actions run completes successfully. If hosted CI fails, use
`github:gh-fix-ci`, reproduce the root cause, fix with a regression, and repeat
this step.
