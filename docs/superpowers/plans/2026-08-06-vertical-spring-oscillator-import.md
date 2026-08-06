# Vertical Spring Oscillator Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the vertical spring oscillator batch as reviewed canonical Cases and publish them through the sole active Dataset `13.0.0` release.

**Architecture:** A focused import module parses the source workbook, maps Trials to archive members, detects the first post-cycle release-side turning frame, materializes canonical media and Case metadata, and emits review artifacts. A separate release builder promotes only reviewed Cases into Dataset `13.0.0`, preserving the single-current-release policy.

**Tech Stack:** Python 3.10, `openpyxl`, NumPy, OpenCV, FFmpeg/FFprobe, JSON/JSONL, `unittest`/`pytest`, Git.

## Global Constraints

- Scene ID is exactly `vertical_spring_oscillator`.
- Canonical frame zero is the first return to the release-side turning point after one complete cycle.
- Retain every source frame and timestamp from the selected start through EOF; do not crop, scale, sample, or time-warp.
- Normalize source display rotation and encode canonical media as H.264 MP4 at the displayed 1080x1920 resolution.
- `x_0` is the absolute source displacement in metres and every `physics.json` scalar is finite and non-negative.
- Caption direction is `below` for positive source displacement and `above` for negative source displacement.
- The sole object mask includes only visible steel-ball pixels.
- Detector output is a candidate only; every accepted Case requires recorded visual review.
- Missing, ambiguous, duplicate, occluded, or unreliable Trials are excluded with explicit provenance instead of guessed values.
- Dataset `12.0.0` is not edited; Dataset `13.0.0` becomes the sole active runtime release.
- Do not push to a remote and do not create asset hashes or a hash lock.
- Delete `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md` only after implementation and verification finish.

---

### Task 1: Intake parser and Trial/source contract

**Files:**
- Create: `src/physbench/datasets/vertical_spring_import.py`
- Create: `tests/test_vertical_spring_import.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `TrialAnnotation`, `SourceMember`, `load_trial_annotations(workbook: Path) -> list[TrialAnnotation]`, `inventory_archive(archive: Path) -> list[SourceMember]`, and `map_trials_to_sources(...) -> IntakeResult`.
- Consumes: the three workbook sheets and ZIP central directory without extracting videos.

- [ ] **Step 1: Add the optional import dependency group**

Add this exact group to `pyproject.toml`:

```toml
dataset-import = [
  "numpy>=1.26",
  "opencv-python-headless>=4.10",
  "openpyxl>=3.1",
]
```

- [ ] **Step 2: Write failing workbook and mapping tests**

Create an in-memory XLSX/ZIP fixture in `tests/test_vertical_spring_import.py` and assert the signed displacement contract, missing-video exclusion, and identical-name ambiguity:

```python
def test_load_trials_preserves_signed_displacement(tmp_path: Path) -> None:
    workbook = write_workbook_fixture(tmp_path / "spring.xlsx", [
        ("T001", "S01", "IMG_1518.MOV", 30.0),
        ("T002", "S01", "IMG_1519.MOV", -40.0),
    ])
    trials = load_trial_annotations(workbook)
    assert [(item.trial_id, item.displacement_mm, item.direction) for item in trials] == [
        ("T001", 30.0, "below"),
        ("T002", -40.0, "above"),
    ]

def test_mapping_excludes_missing_required_video() -> None:
    result = map_trials_to_sources(
        [TrialAnnotation("T001", "S01", "IMG_1652.MOV", 30.0, 2)],
        [],
    )
    assert result.accepted == ()
    assert result.exclusions[0].reason == "workbook_video_missing_from_archive"
```

- [ ] **Step 3: Run the tests and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: FAIL because `vertical_spring_import` and its types do not exist.

- [ ] **Step 4: Implement typed workbook and ZIP intake**

Implement frozen dataclasses and strict parsing. Required public shapes:

```python
@dataclass(frozen=True)
class TrialAnnotation:
    trial_id: str
    spring_id: str
    video_name: str
    displacement_mm: float
    workbook_row: int

    @property
    def direction(self) -> Literal["above", "below"]:
        return "below" if self.displacement_mm > 0 else "above"

    @property
    def displacement_m(self) -> float:
        return abs(self.displacement_mm) / 1000.0

@dataclass(frozen=True)
class SourceMember:
    member: str
    basename: str
    size: int
    crc32: int
```

Reject zero displacement, non-finite values, empty Trial/video fields, unsupported spring IDs, and duplicate non-identical mappings. Ignore rows with only a Trial ID and record their status in the intake report.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: workbook and source-mapping tests PASS.

- [ ] **Step 6: Commit the intake contract**

```bash
git add pyproject.toml src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): add vertical spring intake contract"
```

### Task 2: Ball trajectory, turning-frame detector, and mask geometry

**Files:**
- Modify: `src/physbench/datasets/vertical_spring_import.py`
- Modify: `tests/test_vertical_spring_import.py`

**Interfaces:**
- Consumes: portrait BGR frames with presentation timestamps and a `TrialAnnotation.direction`.
- Produces: `BallDetection`, `TurningFrameCandidate`, `detect_ball(frame)`, `detect_release_return(track, direction, theoretical_period_s=0.789924128416829)`, and `ball_mask(shape, detection)`.

- [ ] **Step 1: Write failing synthetic trajectory tests**

Use a damped cosine where positive image-y is downward:

```python
def test_return_detector_selects_first_lower_return_for_below() -> None:
    times = np.arange(0.0, 2.0, 1.0 / 240.0)
    y = 600.0 + 120.0 * np.exp(-0.08 * times) * np.cos(2 * np.pi * times / 0.8)
    candidate = detect_release_return(track(times, y), "below", 0.8)
    assert abs(candidate.time_s - 0.8) <= 2 / 240

def test_return_detector_selects_first_upper_return_for_above() -> None:
    times = np.arange(0.0, 2.0, 1.0 / 240.0)
    y = 600.0 - 120.0 * np.exp(-0.08 * times) * np.cos(2 * np.pi * times / 0.8)
    candidate = detect_release_return(track(times, y), "above", 0.8)
    assert abs(candidate.time_s - 0.8) <= 2 / 240
```

Also test irregular timestamps, missing detections, insufficient extrema, and a filled-circle mask that is binary and excludes pixels outside the detected ball.

- [ ] **Step 2: Run detector tests and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import.TrajectoryTests -v`  
Expected: FAIL because detector functions do not exist.

- [ ] **Step 3: Implement candidate detection**

Implement these rules:

```python
TURN_PERIOD_RANGE = (0.60, 1.35)
MIN_TRACK_COVERAGE = 0.90
MAX_INTERPOLATION_GAP = 8
```

- Detect the steel ball with Hough circles in a predicted vertical ROI, score candidates by circular edge support, radius continuity, and distance from the previous centre.
- Decode analysis frames at one-quarter displayed resolution but retain exact source frame indices and timestamps.
- Linearly interpolate only gaps up to eight decoded frames for the analysis track.
- Median-filter five samples and convolve with a symmetric nine-sample kernel only for extrema detection.
- Require an opposite-side extremum followed by a same-side extremum at `0.60T` to `1.35T` after motion onset.
- Refine the selected source index by searching unsmoothed detections within eight frames and choosing maximum image-y for `below` or minimum image-y for `above`.
- Return diagnostics including coverage, observed period, displacement span, selected frame, selected PTS, and rejection reason.
- Build the full-resolution mask from a full-resolution circle refit at the selected frame. Fill the interior disk and exclude the upper suspension connector by clipping the mask to the fitted circular silhouette.

- [ ] **Step 4: Run detector and mask tests**

Run: `python3 -m unittest tests.test_vertical_spring_import.TrajectoryTests -v`  
Expected: all detector and mask tests PASS.

- [ ] **Step 5: Commit detector behavior**

```bash
git add src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): detect vertical spring return frames"
```

### Task 3: Exact media trimming and Case serialization

**Files:**
- Modify: `src/physbench/datasets/vertical_spring_import.py`
- Create: `scripts/import_vertical_spring_oscillator.py`
- Modify: `tests/test_vertical_spring_import.py`

**Interfaces:**
- Consumes: an accepted Trial/source mapping plus `TurningFrameCandidate` and optional reviewed frame/circle overrides.
- Produces: `materialize_case(...) -> CaseDraft`, canonical media, first frame, masks, `physics.json`, `caption.json`, mask manifest, audit record, review sheet, and exclusion record.

- [ ] **Step 1: Write failing exact-media and serializer tests**

Generate a 12-frame synthetic variable-timestamp video and assert that selecting frame 4 yields eight canonical frames, canonical frame zero matches the recorded source frame after rotation, and physics values/caption symbols follow the contract:

```python
def test_physics_uses_positive_displacement_and_all_symbols() -> None:
    physics = build_physics("spring_t001",  -40.0)
    assert physics["physics"]["objects"]["object_1"]["initial_displacement"]["value"] == 0.04
    assert all(quantity["value"] >= 0 for quantity in scalar_quantities(physics))
    caption = build_caption("spring_t001", "above")["caption"]
    for symbol in ("m", "r", "x_0", "k", "L_0", "g"):
        assert symbol in caption
```

- [ ] **Step 2: Run media tests and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import.MediaTests -v`  
Expected: FAIL because serializers/materialization are not implemented.

- [ ] **Step 3: Implement FFmpeg materialization and CLI phases**

The CLI exposes exact phases:

```text
inventory --archive PATH --output-dir PATH
analyze --archive PATH --workbook PATH --output-dir PATH
materialize --archive PATH --analysis PATH --review PATH --repo-root PATH
review-sheets --audit PATH --output-dir PATH
```

Use a temporary extracted MOV per active Case, then run an FFmpeg filter trim by decoded frame index with presentation timestamps shifted to zero. Encode with `libx264`, `-pix_fmt yuv420p`, `-fps_mode passthrough`, rotation normalized by FFmpeg autorotation, no scale/crop filter, and audio omitted. Verify output width/height, frame count, timestamps, and EOF retention with FFprobe.

Serialize the fixed SI physics contract and the approved above/below Caption. Use directory names of the form:

```text
spring_m515p6g_r25mm_x{magnitude_mm}mm_{above|below}_img{source_number}
```

Use stable Case IDs of the form:

```text
vertical_spring_s01_x{magnitude_mm}mm_{above|below}_img_{source_number}
```

Write review candidates to `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/review_candidates.jsonl`. A candidate is materialized only when the review record has `status: "approved"`; overrides may specify `source_start_frame`, `ball_center_xy`, and `ball_radius_px`, each with an explanation.

- [ ] **Step 4: Run all import unit tests**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: all tests PASS.

- [ ] **Step 5: Commit the media importer**

```bash
git add scripts/import_vertical_spring_oscillator.py src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): materialize vertical spring cases"
```

### Task 4: Inventory, pilot, review, and batch materialization

**Files:**
- Create ignored media: `datasets/provenance/source_archives/20260806_vertical_spring_oscillator/vertical_spring_oscillator.zip`
- Create: `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/source_workbook.xlsx`
- Create: `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/normalized_annotations.json`
- Create: `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/review_candidates.jsonl`
- Create: `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/review_decisions.jsonl`
- Create: `datasets/provenance/imports/vertical_spring_oscillator_20260806_import_audit.jsonl`
- Create: `datasets/provenance/imports/vertical_spring_oscillator_20260806_exclusions.json`
- Create: `datasets/provenance/imports/vertical_spring_oscillator_20260806_summary.json`
- Create ignored assets: `datasets/assets/vertical_spring_oscillator/*`

**Interfaces:**
- Consumes: the source ZIP and Task 3 CLI.
- Produces: reviewed Case assets and complete provenance used by the Dataset 13 builder.

- [ ] **Step 1: Install local import dependencies and run intake**

Install `python3-numpy`, `python3-opencv`, and `python3-openpyxl` from the system package manager, then run:

```bash
python3 scripts/import_vertical_spring_oscillator.py inventory \
  --archive /root/Steven/竖直弹簧振子.zip \
  --output-dir datasets/provenance/source_docs/20260806_vertical_spring_oscillator
```

Expected: 225 MOV members are inventoried, the workbook is extracted unchanged, empty T103 and T265-T271 are non-candidate rows, and missing archive videos are exclusions.

- [ ] **Step 2: Prove the duplicate decision**

Compare `IMG_1705.MOV` and `IMG_1705(1).MOV` by CRC/size and decoded frame signatures. Record one canonical member and one duplicate exclusion if identical; otherwise exclude the ambiguous mapping.

- [ ] **Step 3: Analyze and select a stratified pilot**

Run analysis for all mapped Trials, then select at least six pilot records: three `above`, three `below`, and at least three distinct displacement magnitudes. Generate start-frame triptychs showing eight frames before, selected frame, and eight frames after, plus first-frame mask overlays.

- [ ] **Step 4: Visually approve or correct the pilot**

For each pilot, verify the same-side post-cycle turning event, absent hand/tool, complete ball, exact circle mask, and full-tail policy. Record `approved`, corrected override, or exclusion; do not change the shared detector for a single outlier that a documented override resolves.

- [ ] **Step 5: Materialize and validate the pilot**

Run `materialize` only for approved pilot records. Assert each Case has the six required files, frame-zero equality, full-tail frame count, mask dimensions, symbol coverage, and non-negative physics values.

- [ ] **Step 6: Review every remaining candidate and materialize the batch**

Generate paginated contact sheets containing the start triptych and mask overlay for every candidate. Inspect every item, write one review decision per candidate, and materialize only approved items. Re-run review sheets from canonical outputs and resolve any mismatch before continuing.

- [ ] **Step 7: Verify the import summary**

Check that:

```text
candidate_count = approved_count + excluded_after_analysis_count
mapped_count = candidate_count + excluded_before_analysis_count
accepted Case IDs = import-audit Case IDs = asset-directory Case IDs
```

Commit only tracked Case metadata and provenance text/doc files; ignored videos, frames, masks, and the 13 GB source archive remain local assets.

- [ ] **Step 8: Commit the reviewed Scene assets**

```bash
git add datasets/assets/vertical_spring_oscillator \
  datasets/provenance/source_docs/20260806_vertical_spring_oscillator \
  datasets/provenance/imports/vertical_spring_oscillator_20260806_*.json*
git commit -m "data: import vertical spring oscillator scene"
```

### Task 5: Build the sole active Dataset 13 release

**Files:**
- Create: `scripts/build_dataset_v13.py`
- Create: `scripts/validate_dataset_v13.py`
- Create: `datasets/releases/13.0.0/dataset.json`
- Create: `datasets/releases/13.0.0/cases.jsonl`
- Create: `datasets/releases/13.0.0/scenes/vertical_spring_oscillator.json`
- Create: `datasets/releases/13.0.0/views/view_a.json`
- Create: `datasets/releases/13.0.0/views/view_b.json`
- Create: `datasets/provenance/releases/13.0.0/cases.jsonl`
- Create: `datasets/provenance/releases/13.0.0/migration.json`
- Create: `datasets/provenance/releases/13.0.0/validation.json`
- Modify: `src/physbench/data_layout.py`
- Modify: `tests/test_current_dataset.py`
- Remove from active tree: `datasets/releases/12.0.0/`, `datasets/provenance/releases/12.0.0/`

**Interfaces:**
- Consumes: Dataset 12 Case/index content plus reviewed vertical-spring import audits.
- Produces: `V13_DATASET`, `LATEST_DATASET`, one seven-Scene runtime descriptor, deterministic train/ID-test views, and independent validator output.

- [ ] **Step 1: Write failing Dataset 13 tests**

Update `tests/test_current_dataset.py` to assert:

```python
assert LATEST_DATASET == V13_DATASET
assert dataset.descriptor["release"] == "13.0.0"
assert set(dataset.scene_configs) == {
    "collision_1d", "inclined_plane_slide", "parabolic_motion", "pendulum",
    "push_bottle", "uniform_circular_motion", "vertical_spring_oscillator",
}
spring = [case for case in dataset.cases if case["scene_id"] == "vertical_spring_oscillator"]
assert len(spring) == import_summary["accepted_count"]
assert 1 <= len(dataset.views["view_a"]["scenes"]["vertical_spring_oscillator"]["test"]) <= 20
```

Also assert every spring Case has exactly one object mask, positive `x_0`, matching Caption direction, and the approved post-cycle alignment provenance.

- [ ] **Step 2: Run Dataset tests and verify failure**

Run: `python3 -m unittest tests.test_current_dataset -v`  
Expected: FAIL because Dataset 13 does not exist.

- [ ] **Step 3: Implement deterministic Dataset 13 builder**

Copy existing Case/index semantics without copying Case-local Caption or physics bodies into `cases.jsonl`. Append spring Cases from the reviewed audit. Split spring Trials by duplicate/near-duplicate group, stratified over direction and displacement magnitude, with at most 20 ID-test Cases and no group spanning train/test. Rebuild view case-set digests from sorted Case IDs, add five complete direct-evaluation groups for the Scene, and write a Scene catalog entry with:

```json
{
  "scene_id": "vertical_spring_oscillator",
  "display_name": "竖直弹簧振子",
  "schema_version": "2.0",
  "structured_physics_parameters": [
    "objects.object_1.initial_displacement",
    "objects.object_1.mass",
    "objects.object_1.radius",
    "environment.gravity_acceleration",
    "environment.natural_spring_length",
    "environment.spring_stiffness"
  ]
}
```

Preserve all prior Case identities and memberships. Remove Dataset 12 runtime/provenance directories only after Dataset 13 validates.

- [ ] **Step 4: Generalize the independent validator**

Create `scripts/validate_dataset_v13.py` from the current v12 validator, replace hard-coded counts with `799 + accepted_count`, require exactly seven Scene configs, validate spring provenance/alignment/masks/Caption/physics, and reject any active release directory other than `13.0.0`.

- [ ] **Step 5: Build and run focused validation**

Run:

```bash
python3 scripts/build_dataset_v13.py
python3 scripts/validate_dataset_v13.py
python3 -m unittest tests.test_current_dataset -v
```

Expected: all commands PASS and only `datasets/releases/13.0.0` is active.

- [ ] **Step 6: Commit Dataset 13**

```bash
git add scripts/build_dataset_v13.py scripts/validate_dataset_v13.py \
  src/physbench/data_layout.py tests/test_current_dataset.py \
  datasets/releases datasets/provenance/releases
git commit -m "data: publish Dataset 13 with vertical spring scene"
```

### Task 6: Documentation, full verification, and requested design-doc removal

**Files:**
- Modify: `README.md`
- Modify: `datasets/README.md`
- Modify: `datasets/DATASET_OVERVIEW.md`
- Modify: `datasets/releases/README.md`
- Modify: task descriptors that name Dataset 12 but remain compatible with the selected evaluator Scenes
- Modify: tests that intentionally assert the single current Dataset release
- Delete: `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md`

**Interfaces:**
- Consumes: the validated Dataset 13 release and final import summary.
- Produces: accurate public documentation and a clean, verified repository state.

- [ ] **Step 1: Update documentation and task release references**

Replace Dataset 12 runtime references with Dataset 13, document the seventh Scene's Case count and physics fields, retain the distinction between Dataset availability and evaluator support, and do not claim a dedicated spring evaluator unless one was implemented and tested.

- [ ] **Step 2: Run targeted tests**

Run:

```bash
python3 -m unittest tests.test_vertical_spring_import tests.test_current_dataset -v
python3 scripts/validate_dataset_v13.py
```

Expected: PASS.

- [ ] **Step 3: Run the repository gate**

Run: `make test`  
Expected: PASS with no Dataset 12 path/count assertion failures.

- [ ] **Step 4: Re-run final media/provenance audit**

Recount Case directories, audit rows, review decisions, train/test IDs, and spring provenance rows. Re-probe every canonical video and revalidate frame zero and masks. Compare results to `vertical_spring_oscillator_20260806_summary.json`; update the summary only from measured results.

- [ ] **Step 5: Remove the approved design document**

Delete only `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md` after all verification above passes. Do not delete this implementation plan or the unrelated untracked `2026-08-06-physical-response-loss.md` file.

- [ ] **Step 6: Commit final documentation and cleanup**

```bash
git add README.md datasets docs/superpowers/specs \
  tasks src tests scripts
git commit -m "docs: finalize vertical spring Dataset 13 import"
```

- [ ] **Step 7: Inspect final state**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: only pre-existing unrelated user files may remain untracked; all vertical-spring implementation work is committed and no remote was changed.
