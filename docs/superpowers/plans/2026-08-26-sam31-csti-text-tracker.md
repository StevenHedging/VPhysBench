# SAM 3.1 CSTI Text Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile CSTI prediction-side identity binding with SAM 3.1 text detection, frame-zero Hungarian binding, locked identities, and three-frame terminal confirmation while preserving the existing CSTI mathematics.

**Architecture:** A lazy shared SAM adapter emits semantic candidate Tubes from aligned generated frames. A pure CSTI observation layer matches candidates to frozen GT masks once on frame zero, constructs locked binary Tubes, and adds audit metadata around the unchanged CSTI metric. The common evaluator base invokes this observer independently of scene expert tracking, and task aggregation excludes visible evaluator initialization failures while reporting initialization coverage.

**Tech Stack:** Python 3.11/3.12, NumPy, SciPy, OpenCV, PyTorch, official Meta SAM 3.1 multiplex predictor, unittest, JSON Schema, A100 GPU.

**Spec:** `docs/superpowers/specs/2026-08-26-sam31-csti-text-tracker-design.md`

## Global Constraints

- Do not modify `src/physbench/evaluation/common/csti/metric.py` or its mathematical definition.
- SAM receives text prompts only; never send GT masks, boxes, points, trajectories, or future annotations.
- Match identities only at frame zero and never re-identify or rematch later.
- Preserve the full physical-overlap timeline and empty masks after termination.
- Keep scene expert scoring behavior independent from the new CSTI observer.
- Keep SAM optional and lazily imported so lightweight interfaces work without Torch or SAM.
- Do not publish absolute Dataset/checkpoint paths or GPU IDs.

---

### Task 1: Observation contracts, configuration, and frame-zero matching

**Files:**
- Create: `src/physbench/evaluation/common/csti/observation.py`
- Modify: `src/physbench/evaluation/common/csti/__init__.py`
- Create: `tests/test_csti_observation.py`

**Interfaces:**
- Produces `CSTIObserverConfig.from_mapping`, `SemanticCandidateTube`, `CSTIObservationResult`, and `match_initial_identities(...)`.
- Consumes manifest entity IDs/classes, first-frame GT masks, and namespaced SAM candidate Tubes.

- [ ] **Step 1: Write failing matching tests**

Use hand-built 8x12 binary masks. Assert a crossed candidate order maps by IoU,
not list order; extra candidates are ignored; different semantic groups cannot
cross-match; candidate shortage and low IoU return explicit initialization
failure; and an alternative complete assignment inside the configured margin
is ambiguous.

```python
result = match_initial_identities(
    entities=(entity("left", "ball"), entity("right", "ball")),
    reference_masks={"left": left_mask, "right": right_mask},
    candidates=(candidate("ball:9", right_mask), candidate("ball:4", left_mask)),
    config=config(initial_match_iou_threshold=0.5),
)
self.assertEqual({"left": "ball:4", "right": "ball:9"}, result.initial_matching)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src:tests:. /home/user/Steven/VPhysBench/.venv/bin/python -m unittest tests.test_csti_observation.CSTIInitialMatchingTest -v`

Expected: import failure for `common.csti.observation`.

- [ ] **Step 3: Implement validated immutable contracts and Hungarian matching**

Normalize all masks to read-only bool arrays, require exact THW/metadata shapes,
map every entity class to exactly one prompt group, and use
`scipy.optimize.linear_sum_assignment(1 - iou_matrix)`. Compute the best valid
alternative by forbidding each selected edge in turn; only alternatives whose
individual IoUs pass the threshold participate in the ambiguity margin.

- [ ] **Step 4: Verify GREEN**

Run the matching test class and `tests.test_csti_adapters`.

- [ ] **Step 5: Commit**

```bash
git add src/physbench/evaluation/common/csti tests/test_csti_observation.py
git commit -m "feat: bind CSTI identities on frame zero"
```

### Task 2: Locked Tube lifecycle and K-frame termination

**Files:**
- Modify: `src/physbench/evaluation/common/csti/observation.py`
- Modify: `tests/test_csti_observation.py`

**Interfaces:**
- Produces `is_valid_track_observation(...)` and `build_locked_prediction_tubes(...)`.
- Consumes one immutable entity-to-candidate mapping and per-frame candidate observations.

- [ ] **Step 1: Write failing lifecycle tests**

Add separate tests proving one and two invalid frames recover; three invalid
frames terminate and backdate to the first; masks remain empty through video
end even if the backend ID returns; an end-of-video one/two-frame pending gap
stays empty with no termination frame; and wrong frame shape or incomplete
mapping raises an explicit contract error.

```python
tube = build_locked_prediction_tubes(..., termination_patience=3)
self.assertEqual(2, tube.termination_frame_per_subject["body"])
self.assertFalse(np.any(np.stack(tube.prediction_masks["body"])[2:]))
```

- [ ] **Step 2: Verify RED**

Run the new lifecycle test class. Expected: missing lifecycle functions.

- [ ] **Step 3: Implement ACTIVE/TERMINATED/VIDEO_END state machine**

Validity checks object presence, exact shape, binary mask, minimum pixels, and
minimum confidence. Provisional invalid frames are emitted empty. On patience
confirmation, empty the Tube from the candidate termination frame onward and
ignore every later observation.

- [ ] **Step 4: Verify GREEN and mutation cases**

Run all `tests.test_csti_observation`; temporarily reason through patience 2/4
to ensure no literal `3` controls behavior outside the default.

- [ ] **Step 5: Commit**

```bash
git add src/physbench/evaluation/common/csti/observation.py tests/test_csti_observation.py
git commit -m "feat: terminate locked CSTI tracks after confirmed loss"
```

### Task 3: CSTI input adapter and result audit without metric changes

**Files:**
- Modify: `src/physbench/evaluation/common/csti/observation.py`
- Modify: `src/physbench/evaluation/common/csti/__init__.py`
- Modify: `tests/test_csti_observation.py`
- Modify: `tests/test_csti_metric.py`

**Interfaces:**
- Produces `observe_csti_tubes(...)`, `decorate_csti_metric(...)`, and `evaluator_init_failure_metric(...)`.
- Consumes an aligned `CSTIInput`, manifest entities, semantic candidates, and observer config.

- [ ] **Step 1: Write failing adapter and regression tests**

Assert the new input preserves reference masks/times/shape/order and changes
only prediction masks/track IDs. Evaluate one literal fixture before and after
the adapter and assert exact equality of video and object scores. Assert video
score is the arithmetic mean of subject scores, not a union/micro score. Assert
initialization failure returns `status="evaluator_init_failure"`, null score,
and complete diagnostics rather than zero.

- [ ] **Step 2: Verify RED**

Run the new adapter tests and record failures for absent functions.

- [ ] **Step 3: Implement adapters and audit decoration**

Create `CSTIEntityTube` values with original reference fields and locked
prediction Tubes. Decorate the mapping returned by existing `evaluate_csti`
with `csti_video`, `csti_per_subject`, subject count, initialization mapping,
IoUs, and termination frames. Do not import or duplicate private metric math.

- [ ] **Step 4: Verify GREEN and frozen metric suite**

Run `tests.test_csti_observation`, `tests.test_csti_metric`, and
`tests.test_csti_adapters`.

- [ ] **Step 5: Prove the metric source is untouched**

Run: `git diff 1c2ed39 -- src/physbench/evaluation/common/csti/metric.py`

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/physbench/evaluation/common/csti tests/test_csti_observation.py tests/test_csti_metric.py
git commit -m "feat: adapt SAM tubes to unchanged CSTI metric"
```

### Task 4: Official SAM 3.1 text-video adapter

**Files:**
- Create: `src/physbench/evaluation/common/masks/sam31_text.py`
- Modify: `src/physbench/evaluation/common/masks/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_sam31_text_adapter.py`
- Create: `tests/integration/test_sam31_text_checkpoint.py`

**Interfaces:**
- Produces `Sam31TextVideoSegmenter.segment(frames, prompt_groups)`, `get_shared_sam31_text_segmenter(config)`, and backend provenance.
- Returns `tuple[SemanticCandidateTube, ...]` using namespaced IDs.

- [ ] **Step 1: Write failing fake-predictor tests**

The fake implements complete official request/response structures. Assert
`add_prompt` contains exactly type/session/frame/text/threshold fields and no
mask/box/point fields; propagation is forward from zero; frame-zero response
and streamed frames retain IDs, masks, normalized boxes, and probabilities;
extra candidates survive; sessions close after success and raised propagation;
and the shared factory calls its predictor factory once.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src:tests:. /home/user/Steven/VPhysBench/.venv/bin/python -m unittest tests.test_sam31_text_adapter -v`

- [ ] **Step 3: Implement lazy checkpoint-verified adapter**

Port only predictor loading/session compatibility mechanics from the reviewed
`sam31-role-observer-v2` branch. Write aligned frames to a temporary JPEG
directory, use one text-only session per concept, normalize every response,
close sessions in `finally`, and serialize predictor calls with a lock. Pin
official source revision `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` in the
optional dependency and checkpoint SHA-256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.

- [ ] **Step 4: Verify unit GREEN and lightweight imports**

Run adapter, CSTI observation, and lightweight import tests. Confirm importing
public CSTI contracts does not add `torch` or `sam3` to `sys.modules`.

- [ ] **Step 5: Add opt-in real checkpoint smoke test**

The test requires `VPHYSBENCH_SAM31_CHECKPOINT` and skips only when unset or no
CUDA device exists. It validates the digest, runs a tiny real VPhysBench clip,
and asserts complete response shapes and closed session state.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/physbench/evaluation/common/masks tests/test_sam31_text_adapter.py tests/integration/test_sam31_text_checkpoint.py
git commit -m "feat: add text-only SAM3.1 video adapter"
```

### Task 5: Common evaluator integration independent of expert tracking

**Files:**
- Modify: `src/physbench/evaluation/common/base.py`
- Modify: `src/physbench/evaluation/common/frozen_reference.py`
- Modify: `tests/test_csti_case_integration.py`

**Interfaces:**
- Consumes sampled generated frames, common times, reference spatial transform, entity manifest, and scene `csti_observer` config.
- Produces an observed/decorated CSTI metric even when scene prediction observation degrades.

- [ ] **Step 1: Write failing integration tests with injected segmenter**

Extend the fake case evaluator so a scene analysis can succeed or raise a
prediction-side `SceneAnalysisError`. Assert successful scene analysis has its
old prediction Tube replaced; expert score is unchanged; prediction-side scene
degradation still retains the independently successful CSTI; reference failure
stays unavailable; and observer initialization failure is not converted to
zero.

- [ ] **Step 2: Verify RED**

Run the focused CSTI case integration tests.

- [ ] **Step 3: Add an injectable common observer hook**

Load aligned frozen reference masks with
`load_frozen_reference_observation`, build the reference-only aligned
`CSTIInput`, invoke the shared segmenter, and retain the result across scene
analysis. Add a private constructor injection used by CPU tests. Pass a CSTI
override into degraded-output construction without changing expert degradation.

- [ ] **Step 4: Verify GREEN across scene integration tests**

Run CSTI case integration plus collision, pendulum, rigid-body, circular,
parabolic, and spring evaluator test modules.

- [ ] **Step 5: Commit**

```bash
git add src/physbench/evaluation/common/base.py src/physbench/evaluation/common/frozen_reference.py tests/test_csti_case_integration.py
git commit -m "feat: observe CSTI independently from scene tracking"
```

### Task 6: Dataset aggregation and initialization coverage

**Files:**
- Modify: `src/physbench/evaluation/task_evaluator.py`
- Modify: `tests/test_csti_aggregation.py`

**Interfaces:**
- Consumes case CSTI statuses `evaluated`, `evaluator_init_failure`, and `not_applicable`.
- Produces equal-video CSTI mean over initialized videos plus explicit coverage/failure counts.

- [ ] **Step 1: Write failing aggregation tests**

Create three applicable records with scores 0.2 and 0.8 plus one initialization
failure, and one not-applicable record. Assert dataset score 0.5,
`init_coverage=2/3`, valid count two, failure count one, and reason count one.
Add a scene-imbalanced fixture proving the dataset score is a video mean rather
than a scene macro mean. Assert init failures never become zero.

- [ ] **Step 2: Verify RED**

Run `tests.test_csti_aggregation -v` and observe the old strict-coverage/null
result.

- [ ] **Step 3: Implement dedicated CSTI aggregation semantics**

Keep expert aggregation untouched. Exclude `not_applicable` from the applicable
denominator, exclude init failures only from the mean, preserve them in status
and reason diagnostics, and calculate per-scene success means and coverage.

- [ ] **Step 4: Verify GREEN and task evaluator regressions**

Run CSTI aggregation, case integration, task evaluator, and preflight tests.

- [ ] **Step 5: Commit**

```bash
git add src/physbench/evaluation/task_evaluator.py tests/test_csti_aggregation.py
git commit -m "feat: report CSTI initialization coverage"
```

### Task 7: Protocol configuration, schema, CLI compatibility, and docs

**Files:**
- Modify: `configs/evaluation/protocols/scene_default_v1.json`
- Modify: `schemas/evaluation_protocol.schema.json`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/RUN_LAYOUT.md`
- Modify: `tests/test_evaluation_protocol_v1.py`
- Modify: `tests/test_release_documentation.py`

**Interfaces:**
- Produces one validated `csti_observer` block per scored scene and documents migration/runtime environment variables.

- [ ] **Step 1: Write failing protocol tests**

Assert every scene declares the SAM 3.1 backend, prompt groups covering its
manifest entity classes, threshold fields, patience three, and no absolute
checkpoint/Dataset/GPU paths. Validate malformed prompt overlap, patience zero,
and out-of-range thresholds fail JSON Schema validation.

- [ ] **Step 2: Verify RED**

Run evaluation protocol and release documentation tests.

- [ ] **Step 3: Add compact scene configurations and schema definition**

Use prompts: collision `ball`; pendulum `pendulum bob`; inclined plane `sliding
block`; circular motion `orbiting metal block`; parabolic `ball`; spring `steel
ball`. Counts come from manifest classes. Set the common checkpoint environment
variable, bfloat16 precision, configurable thresholds, patience three, and
debug output false.

- [ ] **Step 4: Document status/output and migration**

Explain text-only prompting, frame-zero GT matching, locked IDs, pending
end-of-video gaps, initialization coverage, optional dependency installation,
checkpoint environment variable, and unchanged CSTI formula.

- [ ] **Step 5: Verify GREEN and protocol fingerprint behavior**

Run protocol/schema/docs tests and registry tests; confirm the new config
changes evaluator fingerprints deterministically without changing CLI flags.

- [ ] **Step 6: Commit**

```bash
git add configs/evaluation/protocols/scene_default_v1.json schemas/evaluation_protocol.schema.json docs/EVALUATION.md docs/RUN_LAYOUT.md tests/test_evaluation_protocol_v1.py tests/test_release_documentation.py
git commit -m "docs: configure SAM3.1 CSTI observation"
```

### Task 8: Full verification, real-model smoke, and representative replay

**Files:**
- Modify only if a failing test first demonstrates a production defect.
- Write run artifacts outside Git under `/public/vphysbench-cluster/runs/`.

**Interfaces:**
- Produces test logs, real SAM diagnostics, six-scene coverage measurements, and reviewed overlays.

- [ ] **Step 1: Run focused CPU suite**

```bash
PYTHONPATH=src:tests:. /home/user/Steven/VPhysBench/.venv/bin/python -m unittest \
  tests.test_csti_observation tests.test_sam31_text_adapter \
  tests.test_csti_metric tests.test_csti_adapters tests.test_csti_aggregation \
  tests.test_csti_case_integration tests.test_evaluation_protocol_v1 -v
```

- [ ] **Step 2: Run repository gates**

Run `make test-interface`, `make test-evaluation`, JSON Schema validation, and
`git diff --check`. Run any configured lint/type checker discovered in the
repository; if none exists, record that explicitly.

- [ ] **Step 3: Run real SAM 3.1 smoke on one local A100**

```bash
VPHYSBENCH_SAM31_CHECKPOINT=/public/model/sam3.1/sam3.1_multiplex.pt \
PYTHONPATH=src:tests:. \
/public/user/workspace/vphysbench-sam31/envs/evaluator-v1/bin/python \
  -m unittest tests.integration.test_sam31_text_checkpoint -v
```

Record elapsed time, peak allocated/reserved memory, candidate count, matching
IoU, and session cleanup.

- [ ] **Step 4: Replay representative VPhysBench cases**

Use one GPU per worker across the local eight A100s and at least one reference
video plus one generated video from each scored scene. Write overlays and
intermediate Tubes only to a timestamped `/public/vphysbench-cluster/runs/`
directory. Report total cases, initialization success/failure reasons, coverage,
runtime, and qualitative mismatches.

- [ ] **Step 5: Run CSTI source and repository cleanliness audits**

Confirm `metric.py` has no diff, no weights/cache/videos are tracked, all files
are intentional, and the original main worktree's untracked CSTI documentation
is untouched.

- [ ] **Step 6: Commit any test-led fixes and final documentation evidence**

Do not commit generated artifacts. Finish with a clean feature worktree and a
linear list of commits ready for integration into the primary benchmark branch.

