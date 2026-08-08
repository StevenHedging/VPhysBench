# VPhysBench CSTI Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CSTI as an independent, manifest-complete physical-trajectory score while preserving every existing Scene expert score and all v10 output semantics.

**Architecture:** Each official Scene evaluator reuses its existing observation and identity matching to return an in-memory `CSTIInput`. `ReferenceCaseEvaluator` validates and scores that input with one exact prefix-EDT implementation, while `task_evaluator` aggregates expert and CSTI as separate dimensions. A new `scene_default_v11` and new official CSTI Task variants enable the feature append-only.

**Tech Stack:** Python 3.10+, NumPy 1.26+, SciPy `ndimage.distance_transform_edt`, existing OpenCV/open-world entity contracts, JSON Schema Draft 2020-12, pytest/unittest.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-07-csti-evaluation-design.md` at commit `2633ecc`.
- Public metric name/key is `CSTI` / `csti`; do not publish the former long field name.
- Keep `CaseEvaluationResult.score` and current Task top-level scores as expert scores; never combine expert and CSTI.
- Keep `scene_default_v10` and the two existing official Task files byte-for-byte unchanged.
- Enable CSTI only through `scene_default_v11` and new `*_csti.json` official Tasks.
- Evaluate only `same_case_gt`; physics-parent and annotation-only are CSTI `not_applicable`.
- Use GT manifest entities as the denominator; unmatched GT entities score exactly zero; extra predictions receive no direct CSTI penalty.
- Use the physical-overlap common timeline, arithmetic mean over prefix scores, and arithmetic mean over GT entities.
- Use native shared Scene mask resolution; never resize or temporally stretch CSTI masks.
- Normalize existing integer `{0,255}` observer masks to bool at the adapter boundary with `mask > 0`; reject float probability masks.
- Fix `temporal_weight=1.0`, `tolerance_radius=0.05`, `algorithm=exact_prefix_edt` in v11.
- Recompute exact EDT for every prefix; never reuse future-aware distance fields or replace CSTI with cumulative hard IoU.
- Do not serialize raw mask Tubes and do not add matplotlib to the core scorer.
- Use test-driven development and commit after every task.
- Run repository tests with `PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest ...`; this environment has cv2, scipy, and pytest.

---

## File Structure

### New production files

- `src/physbench/evaluation/common/csti/__init__.py` — stable exports for the CSTI package.
- `src/physbench/evaluation/common/csti/contracts.py` — fixed configuration, in-memory Tube contracts, and strict validation exceptions.
- `src/physbench/evaluation/common/csti/metric.py` — exact prefix EDT scorer and zero/not-applicable result builders.
- `src/physbench/evaluation/common/csti/adapters.py` — existing match/aligned-mask results to manifest-ordered bool Tubes.
- `configs/evaluation/protocols/scene_default_v11.json` — v10 expert protocol plus fixed CSTI configuration.
- `tasks/official/five_scene_direct_eval_csti.json` — v13 View B expert+CSTI Task.
- `tasks/official/five_scene_finetune_eval_csti.json` — v13 View A expert+CSTI Task.
- `scripts/benchmark_csti.py` — reproducible exact-backend timing/RSS benchmark.
- `docs/experiments/CSTI_REFERENCE_PERFORMANCE_20260807.md` — measured reference-backend report.

### New tests

- `tests/test_csti_metric.py` — numerical definition, invariants, causality, types, and empty policies.
- `tests/test_csti_adapters.py` — mask normalization, frame-match materialization, manifest order, and unmatched semantics.
- `tests/test_csti_case_integration.py` — shared evaluator lifecycle, result attachment, degraded zero, and v10 opt-out.
- `tests/test_csti_aggregation.py` — Case→Scene→Task aggregation, N/A, strict coverage, and preflight zeros.
- `tests/test_evaluation_protocol_v11.py` — v11 schema, fingerprint, Task identity, and v10 immutability.
- `tests/test_csti_benchmark.py` — benchmark CLI smoke coverage without running the maximum case in CI.

### Existing production files to modify

- `src/physbench/evaluation/common/base.py` — add `SceneAnalysis.csti_input`, common scoring hook, and degraded CSTI zero.
- `src/physbench/evaluation/registry.py` — inject protocol-level general metric configuration into evaluator instances.
- `src/physbench/evaluation/scenes/pendulum/v7_evaluator.py` — expose the matched bob Tube.
- `src/physbench/evaluation/scenes/collision/v5_evaluator.py` — expose separate Tubes for every collision entity.
- `src/physbench/evaluation/scenes/rigid_body_open_world.py` — expose the inclined-plane entity Tube.
- `src/physbench/evaluation/scenes/circular_motion/v6_evaluator.py` — expose v7-inherited frozen per-entity circular Tubes.
- `src/physbench/evaluation/scenes/parabolic_motion/evaluator.py` — expose the bound projectile Tube.
- `src/physbench/evaluation/task_evaluator.py` — normalize score dimensions, aggregate CSTI, and attach preflight-zero CSTI.
- `src/physbench/orchestration/evaluation_variants.py` — reaggregate v11 dimensions during native evaluation validation.
- `scripts/summarize_quantity_run.py` — verify dimensions when a protocol enables general metrics.
- `schemas/v3/evaluation_protocol.schema.json` — define strict top-level `general_metrics.csti`.
- `schemas/v3/task_evaluation.schema.json` — document the optional v11 `dimensions` result object.
- `docs/EVALUATION.md`, `docs/TASKS.md`, and `README.md` — document v10 reproduction and v11 CSTI usage.

---

### Task 1: Exact CSTI contracts and numerical scorer

**Files:**
- Create: `src/physbench/evaluation/common/csti/__init__.py`
- Create: `src/physbench/evaluation/common/csti/contracts.py`
- Create: `src/physbench/evaluation/common/csti/metric.py`
- Create: `tests/test_csti_metric.py`

**Interfaces:**
- Produces: `CSTIConfig.from_mapping(value) -> CSTIConfig`.
- Produces: `CSTIEntityTube`, `CSTIInput`, and `CSTIContractError`.
- Produces: `cumulative_soft_tube_iou(reference_masks, prediction_masks, *, config) -> tuple[float, ...]`.
- Produces: `evaluate_csti(value, *, expected_entities, config) -> dict[str, Any]`.
- Produces: `zero_csti_metric(*, expected_entities, config, degradation_code, degradation_reason) -> dict[str, Any]`.
- Produces: `not_applicable_csti_metric(*, config, reason_code) -> dict[str, Any]`.

- [ ] **Step 1: Write the core failing tests**

Create helpers and explicit tests rather than random property tests:

```python
import unittest

import numpy as np

from physbench.evaluation.common.csti import (
    CSTIConfig,
    CSTIContractError,
    CSTIEntityTube,
    CSTIInput,
    cumulative_soft_tube_iou,
    evaluate_csti,
)
from physbench.evaluation.common.entities import ReferenceCapability


CONFIG = CSTIConfig.from_mapping({
    "enabled": True,
    "algorithm": "exact_prefix_edt",
    "temporal_weight": 1.0,
    "tolerance_radius": 0.05,
    "prefix_aggregation": "arithmetic_mean",
    "case_aggregation": "mean_gt_entities",
    "timeline_policy": "physical_overlap",
    "mask_resolution": "scene_analysis_native",
})


def point_tube(*, x_offset: int = 0, delay: int = 0) -> tuple[np.ndarray, ...]:
    tube = np.zeros((4, 21, 21), dtype=bool)
    for frame in range(delay, 4):
        tube[frame, 10, 5 + frame - delay + x_offset] = True
    return tuple(tube)


class CSTIMetricTest(unittest.TestCase):
    def test_identity_curve_and_score_are_one(self) -> None:
        masks = point_tube()
        curve = cumulative_soft_tube_iou(masks, masks, config=CONFIG)
        self.assertEqual((1.0, 1.0, 1.0, 1.0), curve)

    def test_both_empty_is_one_and_one_empty_is_zero(self) -> None:
        empty = tuple(np.zeros((5, 7), dtype=bool) for _ in range(3))
        nonempty = list(empty)
        nonempty[1] = np.pad(np.ones((1, 1), dtype=bool), ((2, 2), (3, 3)))
        self.assertEqual((1.0, 1.0, 1.0), cumulative_soft_tube_iou(empty, empty, config=CONFIG))
        self.assertEqual((1.0, 0.0, 0.0), cumulative_soft_tube_iou(empty, tuple(nonempty), config=CONFIG))

    def test_near_offset_scores_above_far_offset(self) -> None:
        reference = point_tube()
        near = cumulative_soft_tube_iou(reference, point_tube(x_offset=1), config=CONFIG)
        far = cumulative_soft_tube_iou(reference, point_tube(x_offset=6), config=CONFIG)
        self.assertGreater(float(np.mean(near)), float(np.mean(far)))

    def test_larger_delay_scores_no_higher(self) -> None:
        reference = point_tube()
        one = np.mean(cumulative_soft_tube_iou(reference, point_tube(delay=1), config=CONFIG))
        two = np.mean(cumulative_soft_tube_iou(reference, point_tube(delay=2), config=CONFIG))
        self.assertGreaterEqual(one, two)

    def test_future_masks_cannot_change_an_existing_prefix(self) -> None:
        reference = list(point_tube())
        first = list(point_tube())
        second = list(point_tube())
        second[3] = np.flip(second[3], axis=1)
        curve_a = cumulative_soft_tube_iou(tuple(reference), tuple(first), config=CONFIG)
        curve_b = cumulative_soft_tube_iou(tuple(reference), tuple(second), config=CONFIG)
        self.assertEqual(curve_a[:3], curve_b[:3])

    def test_direct_uint8_255_and_float_inputs_are_rejected(self) -> None:
        bad_255 = tuple(mask.astype(np.uint8) * 255 for mask in point_tube())
        bad_float = tuple(mask.astype(np.float32) for mask in point_tube())
        with self.assertRaises(CSTIContractError):
            cumulative_soft_tube_iou(bad_255, bad_255, config=CONFIG)
        with self.assertRaises(CSTIContractError):
            cumulative_soft_tube_iou(bad_float, bad_float, config=CONFIG)
```

Add named tests for spatial-offset monotonicity, temporal-weight effect, tolerance-radius monotonicity, object width, tracking gaps, `T/H/W == 1`, bool/uint8-01 equality, invalid dimensions, shape mismatch, non-finite config, range, manifest order, unmatched-zero short-circuit, and a manually computed tiny EDT result.

- [ ] **Step 2: Run the tests and confirm the package is absent**

Run:

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_metric.py -q
```

Expected: collection fails with `ModuleNotFoundError: physbench.evaluation.common.csti`.

- [ ] **Step 3: Implement strict contracts and fixed config parsing**

Use these exact public fields and reject unsupported config values in `CSTIConfig.from_mapping`:

```python
@dataclass(frozen=True)
class CSTIConfig:
    enabled: bool
    algorithm: str
    temporal_weight: float
    tolerance_radius: float
    prefix_aggregation: str
    case_aggregation: str
    timeline_policy: str
    mask_resolution: str


@dataclass(frozen=True)
class CSTIEntityTube:
    entity_id: str
    role_id: str
    reference_masks: tuple[np.ndarray, ...]
    prediction_masks: tuple[np.ndarray, ...] | None
    matched_prediction_track_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CSTIInput:
    reference_capability: ReferenceCapability
    times_s: tuple[float, ...]
    frame_shape: tuple[int, int]
    entities: tuple[CSTIEntityTube, ...]
```

Require the complete config key set, `enabled is True`, the five fixed string values from `CONFIG`, finite `temporal_weight > 0`, and finite `tolerance_radius > 0`. `CSTIContractError` subclasses `ValueError` and carries a stable `.code` string.

- [ ] **Step 4: Implement exact prefix scoring**

The core loop must have this causal structure:

```python
def cumulative_soft_tube_iou(reference_masks, prediction_masks, *, config):
    reference = _stack_strict_binary_masks(reference_masks, label="reference")
    prediction = _stack_strict_binary_masks(prediction_masks, label="prediction")
    if reference.shape != prediction.shape:
        raise CSTIContractError("csti_tube_shape_mismatch", "...")
    frame_count, height, width = reference.shape
    spacing = (
        config.temporal_weight / max(frame_count - 1, 1),
        1.0 / max(height - 1, 1),
        1.0 / max(width - 1, 1),
    )
    curve = []
    for end in range(1, frame_count + 1):
        gt_prefix = reference[:end]
        pred_prefix = prediction[:end]
        gt_any = bool(np.any(gt_prefix))
        pred_any = bool(np.any(pred_prefix))
        if not gt_any and not pred_any:
            curve.append(1.0)
            continue
        if gt_any != pred_any:
            curve.append(0.0)
            continue
        gt_soft = _soft_membership(gt_prefix, spacing, config.tolerance_radius)
        pred_soft = _soft_membership(pred_prefix, spacing, config.tolerance_radius)
        numerator = float(np.minimum(gt_soft, pred_soft).sum(dtype=np.float64))
        denominator = float(np.maximum(gt_soft, pred_soft).sum(dtype=np.float64))
        curve.append(float(np.clip(numerator / denominator, 0.0, 1.0)))
    return tuple(curve)
```

Implement `_soft_membership` with `scipy.ndimage.distance_transform_edt(~tube, sampling=spacing)` and linear clipping. Do not call EDT for explicit empty cases.

- [ ] **Step 5: Implement manifest-complete result builders**

`evaluate_csti` receives `expected_entities: Sequence[tuple[str, str]]`, reorders input by those IDs, validates exact ID and role coverage, returns objects in that order, returns `curve=None`/score 0 for `prediction_masks is None`, and computes the two arithmetic means. `zero_csti_metric` must list every expected `(entity_id, role_id)` with zero. `not_applicable_csti_metric` must return status, null score, and `csti_requires_same_case_gt`.

- [ ] **Step 6: Run the numerical test module**

Run:

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_metric.py -q
```

Expected: all CSTI numerical tests pass.

- [ ] **Step 7: Commit the core**

```bash
git add src/physbench/evaluation/common/csti tests/test_csti_metric.py
git commit -m "feat: add exact CSTI metric core"
```

---

### Task 2: Existing-match to Tube adapters

**Files:**
- Create: `src/physbench/evaluation/common/csti/adapters.py`
- Modify: `src/physbench/evaluation/common/csti/__init__.py`
- Create: `tests/test_csti_adapters.py`

**Interfaces:**
- Consumes: Task 1 `CSTIEntityTube` and `CSTIInput`.
- Consumes: existing `EntityMatch`, `EntityManifest`, `OpenWorldObservation`, and `ReferenceCapability`.
- Produces: `normalize_observer_mask(mask, *, frame_shape) -> np.ndarray` bool.
- Produces: `build_csti_input_from_frame_matches(...) -> CSTIInput`.
- Produces: `build_csti_input_from_aligned_masks(...) -> CSTIInput`.

- [ ] **Step 1: Write adapter tests with real open-world contracts**

Use a two-entity manifest-order fixture, a `{0,255}` `ObjectDetection.mask`, and actual `EntityMatch` values:

```python
def test_frame_matches_preserve_gt_entities_and_normalize_255_masks(self):
    observed = np.zeros((9, 11), dtype=np.uint8)
    observed[3:5, 4:7] = 255
    observation = OpenWorldObservation(
        tracks=(OpenWorldTrack(
            track_id="track_b",
            detections=(ObjectDetection(
                frame_index=1,
                detection_id="d1",
                xy=np.array([5.0, 4.0]),
                area_px2=6.0,
                entity_class="block",
                mask=observed,
            ),),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        ),),
        overflow_counts=np.zeros(2),
    )
    value = build_csti_input_from_frame_matches(
        reference_capability=ReferenceCapability.SAME_CASE_GT,
        times_s=(0.0, 0.0625),
        frame_shape=(9, 11),
        expected_entities=(("a", "left"), ("b", "right")),
        reference_masks_by_entity={"a": empty_masks(), "b": gt_masks()},
        prediction_observation=observation,
        matches=(EntityMatch(1, "b", "track_b", 1.0, 0.0),),
    )
    self.assertEqual(("a", "b"), tuple(item.entity_id for item in value.entities))
    self.assertIsNone(value.entities[0].prediction_masks)
    self.assertEqual(np.dtype(bool), value.entities[1].prediction_masks[1].dtype)
    np.testing.assert_array_equal(value.entities[1].prediction_masks[1], observed > 0)
```

Add tests for float-mask rejection, out-of-range frame indices, unknown entity/track IDs, duplicate entity-frame or track-frame slots, a matched detection with `mask=None`, empty per-frame masks, aligned-mask input, and stable sorted unique track IDs.

- [ ] **Step 2: Run the adapter tests and confirm missing imports**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_adapters.py -q
```

Expected: collection fails because the adapter exports do not exist.

- [ ] **Step 3: Implement lossless observer-mask normalization**

`normalize_observer_mask` must require a two-dimensional bool/integer array of the exact frame shape, reject floats, and return a read-only bool copy using `array > 0`. This is the only place `{0,255}` is accepted.

- [ ] **Step 4: Implement frame-match materialization**

Build detection lookup by `(track_id, frame_index)`, allocate one bool empty sequence per expected entity, reject duplicate reference/prediction slots, and set `prediction_masks=None` only when an entity has no `EntityMatch` anywhere. A match whose detection has no mask still establishes identity but leaves that frame empty.

- [ ] **Step 5: Implement aligned-mask materialization**

Use this exact signature for rigid/circular/parabolic callers:

```python
def build_csti_input_from_aligned_masks(
    *,
    reference_capability: ReferenceCapability,
    times_s: Sequence[float],
    frame_shape: tuple[int, int],
    expected_entities: Sequence[tuple[str, str]],
    reference_masks_by_entity: Mapping[str, Sequence[np.ndarray]],
    prediction_masks_by_entity: Mapping[str, Sequence[np.ndarray] | None],
    matched_track_ids_by_entity: Mapping[str, Sequence[str]],
) -> CSTIInput:
    ...
```

Require exact key coverage for all three mappings. Normalize all non-null masks and preserve manifest order.

- [ ] **Step 6: Run core and adapter tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_metric.py tests/test_csti_adapters.py -q
```

Expected: both modules pass.

- [ ] **Step 7: Commit adapters**

```bash
git add src/physbench/evaluation/common/csti tests/test_csti_adapters.py
git commit -m "feat: adapt matched entity masks for CSTI"
```

---

### Task 3: Shared Case evaluator lifecycle and degraded results

**Files:**
- Modify: `src/physbench/evaluation/common/base.py:20`
- Modify: `src/physbench/evaluation/registry.py:28`
- Create: `tests/test_csti_case_integration.py`

**Interfaces:**
- Consumes: Tasks 1–2 `CSTIInput`, `CSTIConfig`, `evaluate_csti`, and zero/N/A builders.
- Produces: `SceneAnalysis.csti_input: CSTIInput | None = None`.
- Produces: `ReferenceCaseEvaluator.csti_enabled: bool`.
- Produces: one shared `_attach_csti_metric(request, analysis)` post-analysis hook.
- Produces: evaluator descriptions/fingerprints that include enabled general metrics.

- [ ] **Step 1: Write shared lifecycle tests**

Create a minimal `ReferenceCaseEvaluator` subclass whose `analyze` returns expert score `0.625` and a two-frame identity Tube. Assert:

```python
result = evaluator.evaluate(request)
self.assertEqual(0.625, result.score)
self.assertEqual(1.0, result.metrics["csti"]["score"])
self.assertNotIn("csti_input", result.to_dict())
```

Add tests that the same fake evaluator without `general_metrics.csti` leaves metrics unchanged, a synthetic physics-parent `CSTIInput` yields metric status `not_applicable`, invalid/missing v11 Tube input returns Case status `error` with its stable CSTI contract code such as `csti_input_missing`, and `_degraded_output` emits manifest-complete CSTI zero.

- [ ] **Step 2: Run the shared integration tests and confirm failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_case_integration.py -q
```

Expected: tests fail because `SceneAnalysis` has no `csti_input` and the base has no CSTI hook.

- [ ] **Step 3: Inject protocol-level general metrics in the registry**

Copy each selected Scene config before evaluator construction and insert the protocol's top-level `general_metrics` only into the in-memory copy:

```python
raw_config = self.protocol["scenes"].get(scene_id, {"type": "unsupported"})
config = dict(raw_config)
if "general_metrics" in self.protocol:
    config["general_metrics"] = self.protocol["general_metrics"]
```

Never mutate `self.protocol`. The existing evaluator fingerprint will then include the effective CSTI config automatically.

- [ ] **Step 4: Add the SceneAnalysis field and common scoring hook**

Parse `self.csti_config` once in `ReferenceCaseEvaluator.__init__`. After `analyze` and before provenance/result construction:

```python
if self.csti_enabled:
    analysis.metrics["csti"] = self._attach_csti_metric(request, analysis)
```

The hook rematerializes the manifest, builds ordered `(entity_id, role_id)` descriptors, returns N/A for non-same-case capability, otherwise requires `analysis.csti_input` and calls `evaluate_csti`. Add algorithm/config/materializer provenance without storing masks.

- [ ] **Step 5: Map CSTI contract failures to the existing internal-error policy**

Catch only `CSTIContractError` around the hook and return Case `status="error"`, `reason_code=exc.code`, and a reason prefixed with `CSTI contract failed:`. Do not catch SciPy process failures as prediction errors and do not turn contract errors into zero.

- [ ] **Step 6: Attach CSTI to degraded prediction-zero results**

When `_degraded_output` runs under v11, materialize the GT manifest and insert `zero_csti_metric(...)` beside the existing degraded expert metric. Reference-side exceptions that return `unavailable` continue to omit CSTI.

- [ ] **Step 7: Run shared and existing base/media tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_csti_case_integration.py \
  tests/test_scene_evaluation.py \
  tests/test_evaluation_media_v2.py \
  tests/test_evaluation_media_partial_v5.py -q
```

Expected: new tests pass and existing lifecycle tests remain green.

- [ ] **Step 8: Commit shared integration**

```bash
git add src/physbench/evaluation/common/base.py src/physbench/evaluation/registry.py tests/test_csti_case_integration.py
git commit -m "feat: integrate CSTI in case evaluation lifecycle"
```

---

### Task 4: Pendulum v7 CSTI adapter

**Files:**
- Modify: `src/physbench/evaluation/scenes/pendulum/v7_evaluator.py:443`
- Modify: `tests/test_pendulum_open_world_v7.py`

**Interfaces:**
- Consumes: `build_csti_input_from_frame_matches`.
- Produces: one manifest-complete bob `CSTIInput` when `self.csti_enabled`.

- [ ] **Step 1: Extend the pendulum self-comparison test**

Enable fixed CSTI config in the test evaluator config, evaluate the existing exact GT/self fixture, and assert:

```python
csti = result.metrics["csti"]
self.assertEqual("evaluated", csti["status"])
self.assertEqual(1, len(csti["objects"]))
self.assertEqual(manifest.entities[0].entity_id, csti["objects"][0]["entity_id"])
self.assertAlmostEqual(1.0, csti["score"], places=12)
self.assertEqual(result.score, result.metrics["scene_subject_state_similarity"]["score"])
```

Add a fail-closed prediction observation assertion that the bob object is unmatched and CSTI is zero.

- [ ] **Step 2: Run the targeted tests and observe missing CSTI input**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_pendulum_open_world_v7.py -k 'self_trace or csti' -q
```

Expected: the v11-style test returns `csti_internal_contract_error` or lacks `metrics.csti`.

- [ ] **Step 3: Build the bob Tube from existing matches**

Immediately before returning `SceneAnalysis`, conditionally call the frame-match adapter with:

```python
reference_masks_by_entity={entity.entity_id: reference_bobs}
prediction_observation=prediction_observation
matches=comparison.matches
expected_entities=((entity.entity_id, entity.role_id),)
```

Assign the result to `SceneAnalysis(csti_input=csti_input, ...)`. Do not use full subject/string union masks.

- [ ] **Step 4: Run all pendulum v7 tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_pendulum_open_world_v7.py -q
```

Expected: all tests pass and existing expert assertions are unchanged.

- [ ] **Step 5: Commit pendulum wiring**

```bash
git add src/physbench/evaluation/scenes/pendulum/v7_evaluator.py tests/test_pendulum_open_world_v7.py
git commit -m "feat: expose pendulum tubes to CSTI"
```

---

### Task 5: Collision v5 per-entity CSTI adapter

**Files:**
- Modify: `src/physbench/evaluation/scenes/collision/v5_evaluator.py:439`
- Modify: `tests/test_collision_v5_evaluator.py`

**Interfaces:**
- Consumes: `build_csti_input_from_frame_matches`.
- Produces: one independent CSTI Tube per collision manifest entity.

- [ ] **Step 1: Write a multi-entity regression test**

Use the existing two-body collision fixture and assert that the v11 analysis exposes two objects in manifest order. Then remove all matches for one entity and assert that only that object's score is zero and the Case score is the mean of the two object scores. Also assert that OR-ing both GT bodies into one Tube would produce a different value, proving CSTI did not reuse `_matched_subject_masks` union output.

- [ ] **Step 2: Run the collision CSTI test and confirm failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_collision_v5_evaluator.py -k 'csti' -q
```

Expected: v11 CSTI is missing because `SceneAnalysis.csti_input` is unset.

- [ ] **Step 3: Expose per-entity reference and prediction Tubes**

Build mappings without the existing union helper:

```python
expected_entities = tuple((entity.entity_id, entity.role_id) for entity in entities)
reference_masks_by_entity = {
    entity_id: tuple(reference_masks[index])
    for index, entity_id in enumerate(entity_ids)
}
csti_input = build_csti_input_from_frame_matches(
    reference_capability=manifest.reference_capability,
    times_s=times_s,
    frame_shape=reference_union[0].shape,
    expected_entities=expected_entities,
    reference_masks_by_entity=reference_masks_by_entity,
    prediction_observation=prediction_objects,
    matches=comparison.matches,
)
```

Attach only when enabled. Keep `_matched_subject_masks` untouched for existing appearance/subject diagnostics.

- [ ] **Step 4: Run collision tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_collision_v5_evaluator.py \
  tests/test_collision_open_world.py \
  tests/test_collision_nbody.py -q
```

Expected: all pass, including multi-entity CSTI isolation.

- [ ] **Step 5: Commit collision wiring**

```bash
git add src/physbench/evaluation/scenes/collision/v5_evaluator.py tests/test_collision_v5_evaluator.py
git commit -m "feat: expose collision entity tubes to CSTI"
```

---

### Task 6: Inclined-plane v7 CSTI adapter

**Files:**
- Modify: `src/physbench/evaluation/scenes/rigid_body_open_world.py:8215`
- Modify: `tests/test_rigid_open_world_v6.py`

**Interfaces:**
- Consumes: `build_csti_input_from_aligned_masks`.
- Produces: one CSTI Tube for the manifest sliding block; v7 inherits this shared base implementation.

- [ ] **Step 1: Add CSTI assertions to the rigid-body shared evaluator test**

For an identical inclined-plane reference/prediction fixture, require one evaluated object with score 1. For the existing legal-exit fixture, assert no new masks exist after GT lifecycle exit and no time-axis extension occurs.

- [ ] **Step 2: Run targeted rigid tests and observe failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_rigid_open_world_v6.py -k 'csti or legal_exit' -q
```

Expected: CSTI assertions fail because no aligned input is attached.

- [ ] **Step 3: Build the aligned rigid-body input**

At the final `SceneAnalysis` construction, pass:

```python
reference_masks_by_entity={reference.entity_id: reference.masks}
prediction_masks_by_entity={reference.entity_id: result.matched_prediction_masks}
matched_track_ids_by_entity={
    reference.entity_id: tuple(sorted({
        match.track_id
        for match in result.comparison.matches
        if match.entity_id == reference.entity_id
    }))
}
```

If the track-ID tuple is empty, pass `prediction_masks_by_entity[entity_id]=None` so an all-empty unmatched array cannot earn double-empty prefix credit.

- [ ] **Step 4: Run the full rigid/open-world test module**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_rigid_open_world_v6.py -q
```

Expected: all existing v6/v7 inherited behavior and CSTI tests pass.

- [ ] **Step 5: Commit inclined-plane wiring**

```bash
git add src/physbench/evaluation/scenes/rigid_body_open_world.py tests/test_rigid_open_world_v6.py
git commit -m "feat: expose inclined-plane tube to CSTI"
```

---

### Task 7: Circular-motion v7 per-entity CSTI adapter

**Files:**
- Modify: `src/physbench/evaluation/scenes/circular_motion/v6_evaluator.py:418`
- Modify: `tests/test_circular_open_world_v6.py`

**Interfaces:**
- Consumes: `build_csti_input_from_aligned_masks`.
- Produces: manifest-ordered frozen circular entity Tubes; v7 inherits v6 `analyze`.

- [ ] **Step 1: Add circular CSTI identity/order tests**

Use the existing normal-result fixture with v7 config. Require object IDs equal `frozen_reference.entity_ids`, a self-comparison score of 1, and per-object curves. In the existing missing-object fixture, require exactly the missing GT orbiter to have `matched=false`, `curve=None`, and score 0.

- [ ] **Step 2: Run targeted circular tests and observe failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_circular_open_world_v6.py -k 'normal_result or csti' -q
```

Expected: no CSTI input is present.

- [ ] **Step 3: Materialize frozen aligned Tubes**

Reuse the already computed `prediction_tracks`:

```python
reference_masks_by_entity = dict(zip(
    frozen_reference.entity_ids,
    frozen_reference.instance_masks,
))
prediction_masks_by_entity = {
    entity_id: (
        prediction_tracks.instance_masks[index]
        if any(match.entity_id == entity_id for match in comparison.matches)
        else None
    )
    for index, entity_id in enumerate(frozen_reference.entity_ids)
}
```

Build track-ID sets from `comparison.matches`, use manifest roles for `expected_entities`, and attach the aligned CSTI input. Do not use `matched_prediction_union`.

- [ ] **Step 4: Run all circular tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_circular_open_world_v6.py -q
```

Expected: all hue/apparatus, identity, integrity, and CSTI tests pass.

- [ ] **Step 5: Commit circular wiring**

```bash
git add src/physbench/evaluation/scenes/circular_motion/v6_evaluator.py tests/test_circular_open_world_v6.py
git commit -m "feat: expose circular entity tubes to CSTI"
```

---

### Task 8: Parabolic-motion bound-projectile CSTI adapter

**Files:**
- Modify: `src/physbench/evaluation/scenes/parabolic_motion/evaluator.py:997`
- Modify: `tests/test_parabolic_motion_evaluator.py`

**Interfaces:**
- Consumes: `build_csti_input_from_aligned_masks`.
- Produces: one projectile Tube only when existing frame-zero binding is accepted.

- [ ] **Step 1: Add accepted/rejected binding CSTI tests**

Extend `test_identical_observation_scores_exactly_one` to enable CSTI and assert one object/score 1. Extend the shifted or missing-projectile fixture so a rejected `scored["binding"]["accepted"]` produces unmatched CSTI zero even when `prediction.masks` contains candidate pixels.

- [ ] **Step 2: Run the targeted tests and observe failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_parabolic_motion_evaluator.py -k 'identical or binding or csti' -q
```

Expected: CSTI is absent.

- [ ] **Step 3: Gate the aligned prediction Tube on existing binding**

Use:

```python
bound = bool(scored["binding"]["accepted"])
prediction_masks = prediction.masks if bound else None
track_ids = ("bound_projectile",) if bound else ()
```

Pass the manifest entity ID/role, `reference.masks`, and the gated prediction into the aligned adapter. Do not derive masks from the fitted parabola and do not add interpolation.

- [ ] **Step 4: Run all parabolic tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_parabolic_motion_evaluator.py -q
```

Expected: all pass, including existing soft integrity composition.

- [ ] **Step 5: Commit parabolic wiring**

```bash
git add src/physbench/evaluation/scenes/parabolic_motion/evaluator.py tests/test_parabolic_motion_evaluator.py
git commit -m "feat: expose parabolic projectile tube to CSTI"
```

---

### Task 9: Dimension-aware Task aggregation and preflight zeros

**Files:**
- Modify: `src/physbench/evaluation/task_evaluator.py:32`
- Modify: `src/physbench/orchestration/evaluation_variants.py:459`
- Modify: `scripts/summarize_quantity_run.py:2622`
- Create: `tests/test_csti_aggregation.py`
- Modify: `tests/test_evaluation_variants.py`
- Modify: `tests/test_summarize_quantity_run.py`

**Interfaces:**
- Consumes: Case `metrics.csti` from Tasks 3–8.
- Produces: internal `DimensionCaseRecord` and `_aggregate_dimension(plan, records)`.
- Changes: `aggregate_task_results(..., general_metrics: Mapping[str, Any] | None = None)`.
- Produces: v11 `task_result.dimensions.expert` and `.csti` without changing v10 output.

- [ ] **Step 1: Write pure aggregation tests**

Construct Case records directly. Cover unequal Scene case counts:

```python
aggregation = aggregate_task_results(
    plan=plan_for_scenes("pendulum", "collision_1d"),
    case_results=[
        case("p1", "pendulum", expert=0.8, csti=0.2),
        case("c1", "collision_1d", expert=0.4, csti=0.8),
        case("c2", "collision_1d", expert=0.6, csti=1.0),
    ],
    general_metrics={"csti": FIXED_CSTI_CONFIG},
)
self.assertAlmostEqual(0.65, aggregation["score"])
self.assertAlmostEqual(0.65, aggregation["dimensions"]["expert"]["score"])
self.assertAlmostEqual(0.55, aggregation["dimensions"]["csti"]["score"])
```

Here expert Scene means are `0.8` and `0.5`; CSTI Scene means are `0.2` and `0.9`, proving both Task values are Scene macro means. Add tests for incomplete coverage/null strict score, observed mean, finetune partitions, one N/A Case excluded, all-N/A Task, and v10 call without `general_metrics` producing the exact former key set.

- [ ] **Step 2: Write preflight-zero tests**

Run `evaluate_task` with one known Case and no prediction record under v11 robustness. Assert global Case status remains `evaluated`, expert score is 0, `metrics.csti.score` is 0, and its objects exactly match manifest entity IDs.

- [ ] **Step 3: Run aggregation tests and confirm failure**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_aggregation.py -q
```

Expected: `aggregate_task_results` rejects `general_metrics` and no dimensions exist.

- [ ] **Step 4: Normalize Case results into dimension records**

Add an internal frozen dataclass or typed dict with job ID, Scene, partition, status, and score. Build expert records from Case top-level. Build CSTI records as follows:

```python
if case_result["status"] != "evaluated":
    status, score = case_result["status"], None
else:
    metric = case_result.get("metrics", {}).get("csti")
    if metric is None:
        status, score = "error", None
    else:
        status, score = metric["status"], metric["score"]
```

Validate that evaluated dimension records have finite `[0,1]` scores and non-evaluated/N/A records have null scores.

- [ ] **Step 5: Extract and reuse the existing strict aggregator**

Move the current partition, Scene, Task macro, generalization, coverage, and status-count logic into `_aggregate_dimension`. First call it for expert and construct the existing top-level dictionary exactly as before. Only when `general_metrics.csti.enabled` is true, call it for CSTI and append:

```python
result["dimensions"] = {
    "expert": {
        "score": expert["score"],
        "source": "top_level_score",
    },
    "csti": csti,
}
```

For CSTI, exclude N/A records from expected counts. If a Scene has zero applicable records, mark it N/A and exclude it from the Task Scene macro; all-N/A yields Task N/A.

- [ ] **Step 6: Attach CSTI to prediction preflight zeros**

Change `_prediction_zero_result` to accept the Case and optional fixed CSTI config. Materialize its manifest and merge `zero_csti_metric` into metrics. Pass the Case/config from both missing-record and incomplete-record branches. Leave protocol-error and reference-unavailable results non-evaluated.

- [ ] **Step 7: Pass general metrics from evaluate_task**

Call:

```python
aggregation = aggregate_task_results(
    plan=plan,
    case_results=results,
    include_degraded_diagnostics=bool(protocol.get("robustness")),
    general_metrics=protocol.get("general_metrics"),
)
```

- [ ] **Step 8: Preserve reevaluation and reporting audits**

In `evaluation_variants.py`, load the protocol identified by the native Task result, verify its fingerprint as already required, and pass its `general_metrics` when recomputing. In `summarize_quantity_run.py`, use the verified protocol path/ID to pass the same mapping and add `dimensions` to compared aggregate fields only when enabled. Extend both test modules with a one-Case CSTI fixture.

- [ ] **Step 9: Run aggregation and audit regressions**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_csti_aggregation.py \
  tests/test_evaluation_variants.py \
  tests/test_summarize_quantity_run.py \
  tests/test_scene_evaluation.py -q
```

Expected: all pass; v10 fixtures remain structurally unchanged.

- [ ] **Step 10: Commit aggregation**

```bash
git add src/physbench/evaluation/task_evaluator.py src/physbench/orchestration/evaluation_variants.py scripts/summarize_quantity_run.py tests/test_csti_aggregation.py tests/test_evaluation_variants.py tests/test_summarize_quantity_run.py
git commit -m "feat: aggregate CSTI as an independent task dimension"
```

---

### Task 10: v11 protocol, schemas, official CSTI Tasks, and user documentation

**Files:**
- Create: `configs/evaluation/protocols/scene_default_v11.json`
- Modify: `schemas/v3/evaluation_protocol.schema.json`
- Modify: `schemas/v3/task_evaluation.schema.json`
- Create: `tasks/official/five_scene_direct_eval_csti.json`
- Create: `tasks/official/five_scene_finetune_eval_csti.json`
- Create: `tests/test_evaluation_protocol_v11.py`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/TASKS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all runtime interfaces from Tasks 1–9.
- Produces: loadable `scene_default_v11` with strict fixed CSTI config.
- Produces: append-only v13 official CSTI Tasks.

- [ ] **Step 1: Write protocol and Task identity tests**

Test exact IDs and immutability:

```python
protocol = load_evaluation_protocol("scene_default_v11")
self.assertEqual("exact_prefix_edt", protocol["general_metrics"]["csti"]["algorithm"])
self.assertEqual(0.05, protocol["general_metrics"]["csti"]["tolerance_radius"])

direct = load_task(ROOT / "tasks/official/five_scene_direct_eval_csti.json")
self.assertEqual("five_scene_direct_eval_v13_csti", direct.task_id)
self.assertEqual("scene_default_v11", direct.value["evaluation"]["protocol"])

legacy = load_task(ROOT / "tasks/official/five_scene_direct_eval.json")
self.assertEqual("scene_default_v10", legacy.value["evaluation"]["protocol"])
```

Validate v11 with Draft 2020-12 JSON Schema. Mutate each fixed enum/value and assert schema or config parsing rejects it. Compare v10/v11 Scene blocks after removing `protocol_id`, `description`, and `general_metrics`; require equality.

- [ ] **Step 2: Run tests and confirm files are absent**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_evaluation_protocol_v11.py -q
```

Expected: v11 protocol and CSTI Task files are missing.

- [ ] **Step 3: Extend the protocol schema**

Add optional root `general_metrics` whose only supported property is required `csti`. Require all eight fixed CSTI fields, use constants for the six string/bool choices, require positive numeric weights/radius, and set `additionalProperties: false` inside both objects. Do not require `general_metrics` globally because v10 remains valid.

- [ ] **Step 4: Add scene_default_v11**

Create it from the complete v10 document with these only semantic changes:

```json
"protocol_id": "scene_default_v11",
"description": "Five-scene expert protocol with independent exact CSTI trajectory scoring, sealed no-padding views, and physical-overlap time.",
"general_metrics": {
  "csti": {
    "enabled": true,
    "algorithm": "exact_prefix_edt",
    "temporal_weight": 1.0,
    "tolerance_radius": 0.05,
    "prefix_aggregation": "arithmetic_mean",
    "case_aggregation": "mean_gt_entities",
    "timeline_policy": "physical_overlap",
    "mask_resolution": "scene_analysis_native"
  }
}
```

Keep every Scene block identical to v10.

- [ ] **Step 5: Add official CSTI Task variants**

Copy each existing v13 official Task semantically, change only `task_id` to the approved `*_v13_csti` ID and `evaluation.protocol` to `scene_default_v11`. Keep Dataset, View, selections, seeds, reporting, and subgroup minimums identical.

- [ ] **Step 6: Document optional Task dimensions in schema**

Add an optional `dimensions` object to `schemas/v3/task_evaluation.schema.json`, with required `expert` and `csti` only when present. Document expert score/source and CSTI status/score/coverage/by_scene/breakdown. Do not make it required for v10 results.

- [ ] **Step 7: Update public docs**

Add a focused CSTI section to `docs/EVALUATION.md` containing the formula, prefix causality, fixed parameters, empty policies, GT denominator, physical overlap, and result paths. Update `docs/TASKS.md` and `README.md` tables/commands to show both historical v10 Tasks and new CSTI Tasks; state explicitly that top-level score remains expert.

- [ ] **Step 8: Run protocol, task, and schema tests**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_evaluation_protocol_v11.py \
  tests/test_evaluation_protocol_v6.py \
  tests/test_task_builder.py \
  tests/test_task_runtime_contracts.py -q
```

Expected: all pass and existing v10 tests remain green.

- [ ] **Step 9: Commit protocol and documentation**

```bash
git add configs/evaluation/protocols/scene_default_v11.json schemas/v3/evaluation_protocol.schema.json schemas/v3/task_evaluation.schema.json tasks/official/five_scene_direct_eval_csti.json tasks/official/five_scene_finetune_eval_csti.json tests/test_evaluation_protocol_v11.py docs/EVALUATION.md docs/TASKS.md README.md
git commit -m "feat: add v11 expert and CSTI evaluation protocol"
```

---

### Task 11: Reference-backend benchmark and final verification

**Files:**
- Create: `scripts/benchmark_csti.py`
- Create: `tests/test_csti_benchmark.py`
- Create: `docs/experiments/CSTI_REFERENCE_PERFORMANCE_20260807.md`

**Interfaces:**
- Consumes: `CSTIConfig` and `cumulative_soft_tube_iou`.
- Produces: deterministic synthetic native-size Tube benchmark with JSON output.
- Produces: recorded wall time and peak RSS for required shapes/object counts.

- [ ] **Step 1: Write a benchmark CLI smoke test**

Invoke `scripts/benchmark_csti.py` through `subprocess.run` with `--frames 3 --height 9 --width 11 --objects 1 --output <tmp>/result.json`. Assert exit 0 and exact JSON keys:

```python
self.assertEqual(
    {
        "algorithm", "frames", "height", "width", "objects",
        "scores", "wall_time_s", "peak_rss_kib",
    },
    set(payload),
)
self.assertEqual("exact_prefix_edt", payload["algorithm"])
self.assertEqual(1, len(payload["scores"]))
self.assertTrue(0.0 <= payload["scores"][0] <= 1.0)
```

- [ ] **Step 2: Run the smoke test and confirm script absence**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_benchmark.py -q
```

Expected: failure because `scripts/benchmark_csti.py` does not exist.

- [ ] **Step 3: Implement deterministic benchmark generation**

The CLI accepts positive integer frames/height/width/objects and an output path. Generate one moving disk per object in bool GT Tubes; produce prediction Tubes with a deterministic one-pixel spatial offset and one-frame gap. Time each object with `time.perf_counter`, capture process peak RSS with `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`, and atomically write the JSON through existing `write_json`.

- [ ] **Step 4: Run the benchmark smoke test**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest tests/test_csti_benchmark.py -q
```

Expected: pass.

- [ ] **Step 5: Run focused CSTI and Scene suites**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest \
  tests/test_csti_metric.py \
  tests/test_csti_adapters.py \
  tests/test_csti_case_integration.py \
  tests/test_csti_aggregation.py \
  tests/test_csti_benchmark.py \
  tests/test_pendulum_open_world_v7.py \
  tests/test_collision_v5_evaluator.py \
  tests/test_rigid_open_world_v6.py \
  tests/test_circular_open_world_v6.py \
  tests/test_parabolic_motion_evaluator.py -q
```

Expected: all pass.

- [ ] **Step 6: Run the maximum-shape exact benchmark under `/usr/bin/time`**

First derive the largest actual v11 sampled shape/frame count from the protocol and a representative Case probe; record those values. Then run the corresponding explicit command, for example the v10 maximum bounds and a typical 5-second/16-fps timeline:

```bash
PYTHONPATH=src /usr/bin/time -v /root/Steven/.venvs/wan22-pair-text/bin/python \
  scripts/benchmark_csti.py \
  --frames 81 \
  --height 832 \
  --width 480 \
  --objects 1 \
  --output /tmp/csti-max-benchmark.json
```

Expected: exit 0 without OOM. Copy the exact command, hardware/runtime context, JSON scores/wall time, and `/usr/bin/time` maximum resident set into `docs/experiments/CSTI_REFERENCE_PERFORMANCE_20260807.md`. If the probed maximum differs, use the probed numbers in both the command and report rather than the example bounds.

- [ ] **Step 7: Run the full repository test suite**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 8: Run structural and whitespace verification**

```bash
git diff --check
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m compileall -q src scripts
```

Expected: both commands exit 0.

- [ ] **Step 9: Inspect v10 and v11 result compatibility**

Run one identical synthetic/frozen Case through v10 and v11. Assert in a small audit script that v10 and v11 expert `score`, expert metric subtree, status, and quality are equal; v10 lacks `dimensions`; v11 alone has `metrics.csti` and Task `dimensions`. Save no generated videos or raw masks to Git.

- [ ] **Step 10: Commit benchmark evidence**

```bash
git add scripts/benchmark_csti.py tests/test_csti_benchmark.py docs/experiments/CSTI_REFERENCE_PERFORMANCE_20260807.md
git commit -m "test: benchmark exact CSTI evaluation"
```

- [ ] **Step 11: Confirm the final worktree**

```bash
git status --short
git log --oneline -12
```

Expected: clean worktree and one intentional commit per task.

---

## Completion Evidence

Before reporting implementation complete, collect these exact facts:

- Commit IDs for Tasks 1–11.
- Focused CSTI/Scene pytest command and pass count.
- Full pytest command and pass count.
- `git diff --check` and `compileall` exit status.
- v10/v11 compatibility audit outcome.
- Maximum exact-backend benchmark dimensions, wall time, and peak RSS.
- Paths to `scene_default_v11`, both CSTI Task files, one Case `metrics.csti` result, and one Task `dimensions.csti` result.

Do not claim completion if the maximum exact benchmark OOMs, any official Scene omits a manifest entity, a robust prediction zero lacks CSTI, or a v10 result gains new fields.
