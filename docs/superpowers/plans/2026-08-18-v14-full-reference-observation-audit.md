# V14 Full Reference-Observation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible all-Case identity, anchor, and tracking audit; visually review all 916 V14 Cases; and atomically repair every confirmed defect in place.

**Architecture:** A new `physbench.reference_observations.curation` package separates catalog/binding checks, scene-aware diagnostics, independent anchor evidence, GPU tracking, candidate bundles, and atomic installation. Thin scripts orchestrate eight GPU workers and generate a one-row-per-Case review ledger; Dataset assets are replaced only after candidate validation and visual acceptance.

**Tech Stack:** Python 3.11, NumPy, OpenCV, SciPy, PyTorch, SAM2.1, FFmpeg/ffprobe, unittest/pytest-compatible tests, eight NVIDIA A100 GPUs.

**Spec:** `docs/superpowers/specs/2026-08-18-v14-full-reference-observation-audit-design.md`

## Global Constraints

- Audit all 916 Dataset 14.0.0 Cases across all seven scenes, including training Cases and `push_bottle`.
- Do not create a second Dataset release or canonical `fixed`/`backup` assets.
- Candidate bytes may exist only under `.local/` or an OS temporary directory and are installed with atomic replacement.
- Existing review decisions and automatic metrics never auto-approve semantic identity.
- Every Case requires a recorded visual anchor decision and full-tube decision.
- Preserve unrelated dirty-worktree changes, especially the pre-existing `datasets/releases/14.0.0/assets.lock.json` edits.
- Use `facebook/sam2.1-hiera-large` for repair candidates unless a recorded Case-specific reason selects another model.
- A declarative override is preferred to Case-specific Python; custom repair code is the final escalation only.
- No production behavior is added without first observing its focused test fail for the intended reason.

---

### Task 1: Catalog and Structural/Identity Audit

**Files:**
- Create: `src/physbench/reference_observations/curation/__init__.py`
- Create: `src/physbench/reference_observations/curation/catalog.py`
- Create: `src/physbench/reference_observations/curation/identity.py`
- Create: `tests/test_reference_observation_curation_identity.py`

**Interfaces:**
- Produces: `CurationCase`, `CurationEntity`, `AuditIssue`, `load_curation_cases(dataset_path)`, `audit_case_bindings(case)`, and `audit_visual_order(case)`.
- Consumes: existing `load_dataset`, `load_reference_observation`, first-frame mask manifests, caption and physics documents.

- [ ] **Step 1: Write failing tests for ordering and Case-wide bindings**

```python
def test_collision_visual_order_rejects_reversed_physics_identity() -> None:
    entities = [
        identity.EntityIdentity("object_1", "01", (812.0, 301.0), 0.0125),
        identity.EntityIdentity("object_2", "02", (774.0, 304.0), 0.0100),
    ]
    issues = identity.audit_entity_order("collision_1d", entities)
    self.assertEqual(["collision_left_to_right_order"], [x.code for x in issues])

def test_mask_physics_keys_must_exactly_cover_object_quantities() -> None:
    issues = identity.audit_physics_key_binding(
        "object_1",
        {"mass", "radius", "initial_velocity"},
        ["objects.object_1.mass", "objects.object_1.radius"],
    )
    self.assertEqual("mask_physics_keys_mismatch", issues[0].code)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_identity.py -q`

Expected: import failure for the absent `curation` package or missing interface, not fixture/setup failure.

- [ ] **Step 3: Implement typed catalog and literal audit rules**

Implement immutable dataclasses with resolved paths but keep dataset-relative references in output records. `audit_entity_order` uses strict x ordering for collision and height-derived row groups for circular motion. `audit_case_bindings` checks exact entity sets, mask IDs, physics keys, caption symbols, timeline lengths, review entity sets, and file-record shapes.

```python
@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: Literal["error", "warning", "review"]
    message: str
    object_id: str | None = None
    observation_indices: tuple[int, ...] = ()
```

`audit_case_bindings(case)` and `audit_visual_order(case)` return immutable
tuples of these records and return an empty tuple only when every applicable
invariant has been evaluated without an issue.

- [ ] **Step 4: Run focused tests and the current-dataset characterization**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_identity.py tests/test_current_dataset.py -q`

Expected: PASS; a read-only invocation over V14 reports the two already identified collision order contradictions without changing assets.

- [ ] **Step 5: Commit the independently testable catalog layer**

```bash
git add src/physbench/reference_observations/curation tests/test_reference_observation_curation_identity.py
git commit -m "feat: audit reference observation identity bindings"
```

### Task 2: Scene-Aware Fast and Pixel Diagnostics

**Files:**
- Create: `src/physbench/reference_observations/curation/quality.py`
- Create: `tests/test_reference_observation_curation_quality.py`
- Modify: `src/physbench/reference_observations/curation/__init__.py`

**Interfaces:**
- Consumes: `CurationEntity`, timeline physical times, trajectory arrays, optionally unpacked masks.
- Produces: `EntityDiagnostics`, `EventRange`, `fast_entity_diagnostics(entity, scene_id)`, and `pixel_entity_diagnostics(masks, states, scene_id)`.

- [ ] **Step 1: Write failing tests that distinguish physical motion from tracker failure**

```python
def test_fast_diagnostics_flags_long_unresolved_run() -> None:
    result = fast_entity_diagnostics(
        area=np.array([100, 100, 0, 0, 0], np.int64),
        centroid_xy=np.array([[0, 0], [1, 0], [np.nan, np.nan],
                              [np.nan, np.nan], [np.nan, np.nan]], np.float32),
        state=np.array([0, 0, 3, 3, 3], np.uint8),
        physical_time=np.arange(5, dtype=np.float64) / 24.0,
        scene_id="collision_1d",
    )
    self.assertEqual((2, 4), result.events[0].index_range)
    self.assertIn("unresolved_run", result.review_reasons)

def test_projectile_translation_alone_is_not_area_instability() -> None:
    result = fast_entity_diagnostics(
        area=np.full(5, 100, np.int64),
        centroid_xy=np.array([[0, 0], [20, 2], [40, 6], [60, 12], [80, 20]], np.float32),
        state=np.zeros(5, np.uint8),
        physical_time=np.arange(5, dtype=np.float64) / 24.0,
        scene_id="parabolic_motion",
    )
    self.assertNotIn("area_instability", result.review_reasons)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_quality.py -q`

Expected: FAIL because diagnostics are not implemented.

- [ ] **Step 3: Implement diagnostics and event localization**

Compute contiguous lifecycle runs, robust log-area deltas, component counts,
holes, border contact, inter-entity overlap, and scene-specific motion
residuals. Return observed statistics and review reasons; do not return a
semantic pass decision.

- [ ] **Step 4: Verify GREEN and run storage tests**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_quality.py tests/test_reference_observation_storage.py -q`

Expected: PASS.

- [ ] **Step 5: Commit diagnostics**

```bash
git add src/physbench/reference_observations/curation tests/test_reference_observation_curation_quality.py
git commit -m "feat: add scene-aware observation diagnostics"
```

### Task 3: Review Evidence, Ledger, and Renderers

**Files:**
- Create: `src/physbench/reference_observations/curation/visualization.py`
- Create: `src/physbench/reference_observations/curation/review.py`
- Create: `scripts/reference_observations/render_review_queue.py`
- Create: `tests/test_reference_observation_curation_review.py`

**Interfaces:**
- Consumes: catalog Cases, diagnostics, existing/candidate masks, source frames.
- Produces: `render_anchor_sheet`, `render_contact_sheet`, `render_dense_event_sheet`, `ReviewDecision`, `read_review_ledger`, and `write_review_ledger`.

- [ ] **Step 1: Write failing ledger and renderer behavior tests**

```python
def test_review_ledger_rejects_duplicate_or_missing_case_rows(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"case_id":"a","anchor_decision":"pass","tube_decision":"pass"}\n'
                    '{"case_id":"a","anchor_decision":"pass","tube_decision":"pass"}\n')
    with self.assertRaisesRegex(ValueError, "duplicate Case"):
        read_review_ledger(path, expected_case_ids={"a", "b"})

def test_anchor_sheet_contains_distinct_existing_and_candidate_boundaries() -> None:
    image = np.zeros((32, 32, 3), np.uint8)
    existing = np.zeros((32, 32), np.uint8); existing[4:12, 4:12] = 1
    candidate = np.zeros((32, 32), np.uint8); candidate[5:13, 5:13] = 1
    rendered = render_anchor_sheet(image, {"object_1": existing},
                                   {"object_1": candidate}, metadata={})
    self.assertGreater(np.unique(rendered.reshape(-1, 3), axis=0).shape[0], 2)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_review.py -q`

Expected: FAIL for missing review/visualization modules.

- [ ] **Step 3: Implement deterministic visual evidence and ledger validation**

Sheets include Case ID, scene, object labels, physics/appearance summary,
observation/source indices, state, area, and colored boundaries. Ledger writes
canonical JSONL sorted by release Case order and requires explicit
`anchor_decision`, `tube_decision`, evidence paths, and reviewer notes.

- [ ] **Step 4: Run focused tests and inspect a generated two-Case fixture**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_review.py -q`

Expected: PASS; generated images are nonempty and deterministic.

- [ ] **Step 5: Commit review tooling**

```bash
git add src/physbench/reference_observations/curation scripts/reference_observations/render_review_queue.py tests/test_reference_observation_curation_review.py
git commit -m "feat: render observation review evidence"
```

### Task 4: Independent Anchor Evidence and Curated SAM2 Tracking

**Files:**
- Create: `src/physbench/reference_observations/curation/anchors.py`
- Create: `src/physbench/reference_observations/curation/tracking.py`
- Create: `src/physbench/reference_observations/curation/overrides.py`
- Create: `configs/reference_observations/audit_v1.json`
- Create: `tests/test_reference_observation_curation_tracking.py`

**Interfaces:**
- Consumes: source frames, scene/appearance/physics facts, existing anchors as comparison-only evidence, validated override JSON.
- Produces: `AnchorCandidate`, `CorrectionPrompt`, `LifecycleOverride`, `build_independent_anchor_candidates`, and `CuratedSam2Tracker.track`.

- [ ] **Step 1: Write failing tests for independent evidence and correction prompts**

```python
def test_anchor_localizer_does_not_use_existing_mask_as_candidate_seed() -> None:
    frames = synthetic_departing_disk_frames()
    wrong = np.zeros(frames[0].shape[:2], np.uint8); wrong[:8, :8] = 1
    candidates = build_independent_anchor_candidates(
        frames, scene_id="parabolic_motion", expected_count=1,
        comparison_masks={"object_1": wrong},
    )
    self.assertGreater(candidates[0].centroid_xy[0], 12.0)

def test_override_rejects_duplicate_object_prompt_on_same_frame() -> None:
    with self.assertRaisesRegex(ValueError, "duplicate correction prompt"):
        validate_override({"case_id": "case", "corrections": [
            {"object_id": "object_1", "frame": 3, "points": [[5, 5, 1]]},
            {"object_id": "object_1", "frame": 3, "points": [[6, 6, 1]]},
        ]})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_tracking.py -q`

Expected: FAIL for missing interfaces.

- [ ] **Step 3: Implement motion/geometry anchor candidates and validated overrides**

Build independent support from temporally separated frames, connected
components, scene geometry, and subject count. Existing masks are used only to
calculate disagreement after candidates exist. Overrides accept boxes,
positive/negative points, correction frames, and explicit lifecycle ranges.

- [ ] **Step 4: Implement large-model joint propagation and re-seeding**

Wrap the existing SAM2 dependency with one model instance per worker. Joint
instances compete by highest positive logit. Correction frames split the
timeline into segments propagated forward/backward and joined at reviewed
boundaries. Return masks, states, prompts, model metadata, and exact source
frame mapping without writing canonical assets.

- [ ] **Step 5: Verify unit tests and one GPU smoke Case**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_tracking.py -q`

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reference_observations/rebuild_cases.py --dataset datasets/releases/14.0.0/dataset.json --case-id circular_r1_silver02cm_img_0370 --candidate-root .local/reference_observation_curation/smoke --dry-run`

Expected: tests PASS; smoke writes only below `.local/`, records the large model and produces a valid candidate.

- [ ] **Step 6: Commit anchor/tracking layer**

```bash
git add src/physbench/reference_observations/curation configs/reference_observations tests/test_reference_observation_curation_tracking.py
git commit -m "feat: rebuild curated observation candidates"
```

### Task 5: Candidate Bundles, Atomic Installation, and Hash Closure

**Files:**
- Create: `src/physbench/reference_observations/curation/bundle.py`
- Create: `src/physbench/reference_observations/curation/install.py`
- Create: `tests/test_reference_observation_curation_install.py`
- Modify: `scripts/build_dataset_distribution.py`

**Interfaces:**
- Consumes: accepted masks/states, timeline, Case metadata, canonical target paths, current asset lock.
- Produces: `build_candidate_bundle`, `validate_candidate_bundle`, `install_candidate_bundle`, and a scoped `refresh_locked_files(paths)` helper.

- [ ] **Step 1: Write failing tests for complete-bundle validation and atomicity**

```python
def test_candidate_rejects_trajectory_not_derived_from_masks(tmp_path: Path) -> None:
    candidate = write_synthetic_candidate(tmp_path, corrupt_area=True)
    with self.assertRaisesRegex(ValueError, "area_pixels"):
        validate_candidate_bundle(candidate)

def test_failed_install_leaves_every_canonical_byte_unchanged(tmp_path: Path) -> None:
    canonical = write_synthetic_canonical_bundle(tmp_path / "canonical")
    before = tree_sha256(canonical)
    candidate = write_synthetic_candidate(tmp_path / "candidate", missing_review=True)
    with self.assertRaises(ValueError):
        install_candidate_bundle(candidate, canonical)
    self.assertEqual(before, tree_sha256(canonical))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_install.py -q`

Expected: FAIL for missing bundle/install behavior.

- [ ] **Step 3: Build bundles inside-out and validate exact reductions**

Write mask tubes, trajectories, timeline, quality, visualization files,
review, and manifests in dependency order. Recompute centroids, bboxes, and
areas from masks; compare every result before acceptance.

- [ ] **Step 4: Implement scoped atomic installation and asset-lock refresh**

Acquire a per-Case filesystem lock, preflight every target, write/fsync sibling
temporaries, replace the complete target set, validate the installed Case, and
refresh only affected lock entries while preserving unrelated current entries.

- [ ] **Step 5: Verify GREEN and distribution regression tests**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_install.py tests/test_dataset_distribution.py tests/test_dataset_contract_v6.py -q`

Expected: PASS.

- [ ] **Step 6: Commit bundle/install layer**

```bash
git add src/physbench/reference_observations/curation scripts/build_dataset_distribution.py tests/test_reference_observation_curation_install.py
git commit -m "feat: atomically install curated observations"
```

### Task 6: Full Audit and Eight-GPU Orchestration

**Files:**
- Create: `scripts/reference_observations/audit_v14.py`
- Create: `scripts/reference_observations/rebuild_cases.py`
- Create: `tests/test_reference_observation_curation_cli.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: all earlier curation interfaces and audit config.
- Produces: resumable per-Case diagnostics, candidates, review queues, accepted installations, and final JSONL/Markdown reports.

- [ ] **Step 1: Write failing CLI tests for deterministic sharding and safe defaults**

```python
def test_gpu_shards_cover_each_case_exactly_once() -> None:
    shards = shard_case_ids(["c", "a", "b", "d"], shard_count=3)
    self.assertEqual(["a", "b", "c", "d"], sorted(x for s in shards for x in s))
    self.assertEqual(4, len(set(x for s in shards for x in s)))

def test_rebuild_requires_explicit_review_acceptance_before_install() -> None:
    result = run_cli("rebuild_cases.py", "--case-id", "case", "--install")
    self.assertNotEqual(0, result.returncode)
    self.assertIn("accepted review decision", result.stderr)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_cli.py -q`

Expected: FAIL because scripts/interfaces do not exist.

- [ ] **Step 3: Implement resumable orchestration**

The audit script writes one diagnostic record per Case. The rebuild script
supports `--shard-index/--shard-count`, one GPU selected by
`CUDA_VISIBLE_DEVICES`, candidate-only default, and installation only for a
ledger row with an accepted decision and matching candidate digest.

- [ ] **Step 4: Verify GREEN and launch the read-only full structural pass**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_cli.py -q`

Run: `.venv/bin/python scripts/reference_observations/audit_v14.py --dataset datasets/releases/14.0.0/dataset.json --output .local/reference_observation_curation/full/diagnostics.jsonl --render-root .local/reference_observation_curation/full/review --no-install`

Expected: 916 unique diagnostic rows and no canonical asset changes.

- [ ] **Step 5: Commit orchestration**

```bash
git add scripts/reference_observations tests/test_reference_observation_curation_cli.py Makefile
git commit -m "feat: orchestrate full v14 observation audit"
```

### Task 7: Per-Case Visual Review and In-Place Repairs

**Files:**
- Create/update: `docs/audits/v14_full_reference_observation_audit_20260818.jsonl`
- Create/update: `docs/audits/v14_full_reference_observation_audit_20260818.md`
- Create as needed: `configs/reference_observations/overrides/<case_id>.json`
- Create only if required: `scripts/reference_observations/repairs/<case_id>.py`
- Modify as defects require: `datasets/assets/<scene>/<case>/...`
- Modify: `datasets/releases/14.0.0/assets.lock.json`

**Interfaces:**
- Consumes: diagnostics, anchor sheets, contact sheets, dense-event sheets, overlay videos, GPU candidates.
- Produces: 916 final visual decisions and corrected canonical assets.

- [ ] **Step 1: Review numbering and anchor evidence for every Case**

Process scene-by-scene. Record the actual image/sheet inspected and decision;
do not copy existing `approved` status. Start with all 342 multi-object Cases
and the two known contradictions, then single-object scenes.

- [ ] **Step 2: Review every full tube**

Inspect every contact sheet. Inspect full overlay videos or dense-event sheets
for all warnings, lifecycle transitions, contacts, model disagreements, and
diagnostic flags. Record exact failing indices and root-cause hypothesis.

- [ ] **Step 3: Generate candidates on eight GPU shards**

Run eight processes with shard indices 0 through 7 and distinct
`CUDA_VISIBLE_DEVICES`. Re-run only Cases needing repair or independent model
comparison; never install from worker processes.

- [ ] **Step 4: Apply the smallest evidence-backed repair per Case**

Use verified anchor replacement, large-model propagation, correction frames,
lifecycle fixes, declarative overrides, then specific scripts in that order.
For numbering errors, remap the entire Case semantic binding rather than only
renaming masks.

- [ ] **Step 5: Accept and atomically install each repaired candidate**

Re-review candidate anchor, dense failure region, full contact sheet, and
overlay video. Write the accepted ledger decision and install only when its
digest matches the reviewed candidate.

- [ ] **Step 6: Regenerate summary and verify audit completeness**

Run: `.venv/bin/python scripts/reference_observations/audit_v14.py --dataset datasets/releases/14.0.0/dataset.json --ledger docs/audits/v14_full_reference_observation_audit_20260818.jsonl --require-complete --verify-installed`

Expected: exactly 916 reviewed Cases, no duplicate/missing rows, no unreviewed
errors, and every installed digest equal to the ledger.

- [ ] **Step 7: Commit reviewed assets and audit records in scene-sized commits**

Stage explicit affected paths, audit rows, overrides/scripts, and the asset
lock. Never use `git add datasets/assets` without first listing the exact
repaired Cases.

### Task 8: Completion Verification

**Files:**
- Modify if evidence demands: release validation tests and audit summary only.

**Interfaces:**
- Consumes: final working tree, ledger, Dataset, tests, release scripts.
- Produces: fresh requirement-by-requirement evidence for completion.

- [ ] **Step 1: Verify the curation tests**

Run: `.venv/bin/python -m pytest tests/test_reference_observation_curation_identity.py tests/test_reference_observation_curation_quality.py tests/test_reference_observation_curation_review.py tests/test_reference_observation_curation_tracking.py tests/test_reference_observation_curation_install.py tests/test_reference_observation_curation_cli.py -q`

Expected: all PASS.

- [ ] **Step 2: Verify the complete project suite**

Run: `make test`

Expected: exit 0 with zero failed tests.

- [ ] **Step 3: Verify Dataset bytes and hashes**

Run: `.venv/bin/physbench validate-dataset --dataset datasets/releases/14.0.0/dataset.json --check-assets`

Run: `.venv/bin/python scripts/release_audit.py`

Expected: exit 0; no missing, escaping, mismatched, or stale assets.

- [ ] **Step 4: Run the final full semantic audit gate**

Run: `.venv/bin/python scripts/reference_observations/audit_v14.py --dataset datasets/releases/14.0.0/dataset.json --ledger docs/audits/v14_full_reference_observation_audit_20260818.jsonl --require-complete --verify-installed --fail-on-unjustified-unresolved`

Expected: 916/916 anchor decisions and 916/916 tube decisions accepted; no
unexplained unresolved samples, ordering failures, stale evidence, or digest
mismatches.

- [ ] **Step 5: Audit the objective line by line and update the Markdown report**

Confirm the report gives authoritative evidence for numbering, anchor
identity, every-Case full-tube review, each repair, script location, absence of
redundant Dataset versions, and all verification commands/results.

- [ ] **Step 6: Commit final verification records**

```bash
git add docs/audits/v14_full_reference_observation_audit_20260818.*
git commit -m "data: complete v14 observation curation"
```
