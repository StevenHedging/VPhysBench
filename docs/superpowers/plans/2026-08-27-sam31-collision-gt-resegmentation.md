# SAM 3.1 Collision GT Resegmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and publish identity-correct SAM 3.1 anchors and frozen tubes for all 330 collision Cases.

**Architecture:** A pure curation core discovers and orders candidates without consulting old masks, while a GPU adapter owns official SAM 3.1 sessions and forward propagation. Candidate bundles remain staged until structural, geometric, lifecycle, and visual gates pass, then the existing atomic installer closes all hashes.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PyTorch/CUDA, official SAM 3.1 multiplex predictor, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-sam31-collision-gt-resegmentation-design.md`

## Global Constraints

- Rebuild all 330 `collision_1d` Cases and no other scene in this run.
- Never use existing anchor or tube pixels as SAM prompts or candidate geometry.
- Bind identities row-major: top-to-bottom rows, then left-to-right within a row.
- Preserve the existing 24 Hz timeline and its exact sample count.
- Keep per-object masks independent; overlapping subject pixels are allowed.
- Use pinned SAM 3.1 source revision `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` and checkpoint digest `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
- Do not overwrite canonical assets until candidate validation and recorded review succeed.
- Use eight local A100 GPUs for the production run.

---

### Task 1: Pure identity and candidate-selection contracts

**Files:**
- Create: `src/physbench/reference_observations/curation/sam31_gt.py`
- Create: `tests/test_reference_observation_sam31_gt.py`
- Modify: `src/physbench/reference_observations/curation/__init__.py`

**Interfaces:**
- Consumes: `physics.objects`, SAM frame-zero candidate masks, boxes, probabilities, and optional prompt labels.
- Produces: `GtCandidate`, `OrderedGtIdentity`, `row_major_candidates(...)`, `select_collision_candidates(...)`, and `validate_physics_caption_binding(...)`.

- [ ] **Step 1: Write failing tests for row-major order and metadata binding**

```python
def test_row_major_candidates_groups_rows_before_sorting_x():
    ordered = row_major_candidates((bottom_left, top_right, top_left))
    assert [item.candidate_id for item in ordered] == ["top-left", "top-right", "bottom-left"]

def test_binding_rejects_missing_caption_symbol():
    with pytest.raises(ValueError, match="caption misses v_2"):
        validate_physics_caption_binding(physics_two_objects, caption_without_v2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=src .../bin/python -m pytest tests/test_reference_observation_sam31_gt.py -q`

Expected: collection fails because `sam31_gt` does not exist.

- [ ] **Step 3: Implement immutable candidate types, row grouping, and binding validation**

```python
def row_major_candidates(candidates):
    tolerance = max(1.0, 0.5 * median(item.height for item in candidates))
    # stable y-row clustering followed by stable x ordering
```

- [ ] **Step 4: Add failing selector tests for duplicates, exact counts, and ambiguity**

```python
def test_selector_ignores_high_confidence_fastener_outside_collision_row():
    result = select_collision_candidates(candidates, expected_radii=(0.01, 0.0125))
    assert [item.candidate_id for item in result] == ["left-ball", "right-ball"]
```

- [ ] **Step 5: Implement deterministic deduplication and exact-count global selection**

- [ ] **Step 6: Run focused tests and commit**

Run: `PYTHONPATH=src .../bin/python -m pytest tests/test_reference_observation_sam31_gt.py -q`

### Task 2: Official SAM 3.1 curation adapter

**Files:**
- Create: `src/physbench/reference_observations/curation/sam31_predictor.py`
- Create: `tests/test_reference_observation_sam31_predictor.py`

**Interfaces:**
- Consumes: equal-size BGR frames, text prompt ensemble, selected candidate points, expected subject count.
- Produces: `Sam31GtPrediction` containing backend IDs, THW masks, per-frame boxes/probabilities, prompt provenance, and closed-session diagnostics.

- [ ] **Step 1: Write a fake-predictor test proving text-only proposal requests**

```python
def test_discovery_uses_frame_zero_text_and_closes_session(fake_predictor):
    result = adapter.discover(frames, prompts=("small round object",))
    assert fake_predictor.requests[1]["frame_index"] == 0
    assert fake_predictor.requests[1]["text"] == "small round object"
    assert fake_predictor.requests[-1]["type"] == "close_session"
```

- [ ] **Step 2: Verify RED, then implement model loading and discovery sessions**

- [ ] **Step 3: Write failing tests for locked-ID point initialization, forward-only propagation, overlapping masks, malformed outputs, and exception cleanup**

- [ ] **Step 4: Implement stateful propagation while preserving independent masks**

- [ ] **Step 5: Run adapter tests and existing evaluator adapter regression tests**

Run: `PYTHONPATH=src .../bin/python -m pytest tests/test_reference_observation_sam31_predictor.py tests/test_sam31_text_adapter.py -q`

### Task 3: Per-Case rebuild orchestration and lifecycle gates

**Files:**
- Create: `src/physbench/reference_observations/curation/sam31_rebuild.py`
- Create: `tests/test_reference_observation_sam31_rebuild.py`

**Interfaces:**
- Consumes: `CurationCase`, `Sam31GtPredictor`, and `Sam31GtConfig`.
- Produces: `Sam31GtCaseCandidate` with ordered anchors, entity observations, QA findings, provenance, and rendered-event indices.

- [ ] **Step 1: Write failing tests proving old masks are not read and timeline sample indices are preserved**

- [ ] **Step 2: Implement reference-frame decoding and primary prompt discovery**

- [ ] **Step 3: Write failing tests for prompt fallback, border-crop recovery, internal drop, legal boundary exit, reappearance rejection, order inversion, and near-duplicate tube detection**

- [ ] **Step 4: Implement fallback discovery and deterministic lifecycle/geometry gates**

- [ ] **Step 5: Add a regression test that contact overlap remains present in both object tubes**

- [ ] **Step 6: Run focused tests and commit**

### Task 4: Candidate bundle and review artifacts

**Files:**
- Create: `src/physbench/reference_observations/curation/sam31_bundle.py`
- Create: `tests/test_reference_observation_sam31_bundle.py`
- Modify: `src/physbench/reference_observations/curation/visualization.py`

**Interfaces:**
- Consumes: `Sam31GtCaseCandidate` and original Case metadata.
- Produces: complete staged masks, tubes, trajectories, manifests, review record, visualizations, and `candidate_bundle.json` accepted by `validate_candidate_bundle(...)`.

- [ ] **Step 1: Write failing tests for anchor/tube-zero equality and row-major manifest IDs**

- [ ] **Step 2: Implement deterministic anchor and entity serialization**

- [ ] **Step 3: Write failing tests for generator provenance, hash closure, and rejected QA status**

- [ ] **Step 4: Implement manifests, quality ledger, contact sheet, trajectory image, and overlay-video records**

- [ ] **Step 5: Run bundle and existing storage/install tests**

### Task 5: Sharded CLI and configuration

**Files:**
- Create: `scripts/reference_observations/rebuild_collision_sam31.py`
- Create: `configs/reference_observations/collision_sam31_v1.json`
- Create: `tests/test_reference_observation_sam31_cli.py`

**Interfaces:**
- Consumes: Dataset descriptor, candidate root, shard index/count, checkpoint environment, optional Case list.
- Produces: per-shard JSONL ledger, failure ledger, progress records, and staged Case bundles.

- [ ] **Step 1: Write failing CLI tests for deterministic disjoint sharding and collision-only selection**

- [ ] **Step 2: Implement argument/config validation and one-predictor-per-worker execution**

- [ ] **Step 3: Add resume tests proving accepted Case bundles are checksum-validated before skipping**

- [ ] **Step 4: Implement atomic progress and failure ledgers**

- [ ] **Step 5: Run CLI tests and `--help` smoke test**

### Task 6: GPU smoke and the 13-case regression gate

**Files:**
- Modify only if a smoke-discovered bug first receives a failing regression test in Tasks 1--5.
- Create run artifacts under `/public/vphysbench-cluster/datasets/.local/sam31-collision-gt-20260827/`.

**Interfaces:**
- Consumes: the diagnosed 13-Case list and one local GPU.
- Produces: candidate bundles, overlays, ledger, timing, and exact failure diagnostics.

- [ ] **Step 1: Verify checkpoint digest, source revision, CUDA visibility, and one-Case inference**

- [ ] **Step 2: Run the 13-Case shard and inspect every anchor/contact sheet**

- [ ] **Step 3: For each defect, add a failing regression test before changing code**

- [ ] **Step 4: Repeat until all semantically valid candidates pass or have an explicit non-promoted failure record**

### Task 7: Eight-GPU production rebuild and review

**Files:**
- Create run artifacts and the authoritative 330-row review ledger under the shared candidate root.

**Interfaces:**
- Consumes: all 330 collision Cases split into eight deterministic shards.
- Produces: 330 staged candidate bundles plus a merged audit report.

- [ ] **Step 1: Launch eight local workers with `CUDA_VISIBLE_DEVICES=0..7` and shard indices `0..7`**

- [ ] **Step 2: Monitor per-shard throughput, GPU utilization, failures, and ETA**

- [ ] **Step 3: Merge ledgers and assert exactly 330 unique Case IDs**

- [ ] **Step 4: Run structural and QA gates over every candidate and visually inspect the flagged review atlas**

- [ ] **Step 5: Record pass/failure decisions without silently promoting rejected Cases**

### Task 8: Atomic publication and re-evaluation

**Files:**
- Modify: `datasets/releases/14.0.0/assets.lock.json`
- Modify: `datasets/releases/14.0.0/reference_observation_curation.json`
- Modify generated Case assets only through candidate-bundle installation.

**Interfaces:**
- Consumes: accepted 330-row ledger and validated bundles.
- Produces: hash-closed canonical collision assets and a new SAM 3.1 evaluation report for the existing 330 predictions.

- [ ] **Step 1: Snapshot exact pre-install digests and verify every accepted bundle**

- [ ] **Step 2: Install accepted bundles atomically and refresh dependent hashes inside-out**

- [ ] **Step 3: Run Dataset schema/current-state tests and `validate-dataset --check-assets`**

- [ ] **Step 4: Re-run the latest evaluator over all existing 330 generated videos using available local GPUs**

- [ ] **Step 5: Report CSTI, initialization coverage, per-case changes, remaining failures, runtime, and shared-SAM bias**

- [ ] **Step 6: Run the full relevant test suite, inspect `git diff`, and commit code/metadata changes**

