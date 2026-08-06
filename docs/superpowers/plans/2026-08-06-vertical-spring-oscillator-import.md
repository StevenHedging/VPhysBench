# Vertical Spring Oscillator Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the vertical-spring-oscillator source batch as reviewed canonical Cases and publish them through the sole active local Dataset `13.0.0`.

**Architecture:** A reusable import module separates workbook/source normalization, one-dimensional ball tracking, exact-frame media materialization, mask creation, and Case serialization. A thin CLI runs intake, pilot, batch, review, and release-build stages; deterministic text assets are tracked while multi-gigabyte source and canonical media remain under existing Git-ignore rules. Dataset `13.0.0` is generated from frozen `12.0.0` metadata plus accepted new Cases, then validated as the only runtime release.

**Tech Stack:** Python 3.10, standard library, OpenPyXL, NumPy, OpenCV, FFmpeg/FFprobe, `unittest`, JSON/JSONL.

## Global Constraints

- Use `scene_id = "vertical_spring_oscillator"`.
- Canonical frame zero is the first post-cycle return to the release-side turning point: lower for positive source displacement and upper for negative source displacement.
- Keep every source frame from the chosen exact frame through EOF; do not crop, scale, sample, duplicate, or time-warp frames.
- Normalize display rotation to portrait, retain source presentation timestamps, and encode H.264 MP4.
- Store non-negative SI values only: `m = 0.5156 kg`, `r = 0.025 m`, `x_0 = abs(source_mm)/1000`, `k = 32.6213467096774 N/m`, `L_0 = 0.068 m`, and `g = 9.80665 m/s^2`.
- Preserve signed source displacement only in provenance; use it to select Caption word `above` or `below`.
- Mask only the visible steel-ball disk; exclude ring, connector, spring, ruler, rig, background, shadows, hands, and tools.
- Do not guess missing values or interpret workbook value `42.55`.
- Every accepted Case requires recorded full visual review; detector confidence alone cannot accept a Case.
- Do not edit Dataset `12.0.0` in place. Publish `13.0.0` as the sole active runtime release and preserve `12.0.0` in Git history.
- Do not generate asset hashes or an asset lock, and do not push to a remote.
- Preserve the unrelated untracked file `docs/superpowers/plans/2026-08-06-physical-response-loss.md`.
- Delete `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md` only after implementation and verification complete, as requested by the user.

---

## File Structure

- Create `src/physbench/datasets/vertical_spring_import.py`: typed intake, tracking, mask, serialization, release-row, and audit helpers.
- Create `scripts/import_vertical_spring_oscillator.py`: staged CLI and FFmpeg orchestration.
- Create `tests/test_vertical_spring_import.py`: synthetic workbook, trajectory, mask, media-tail, and serializer tests.
- Modify `pyproject.toml`: add a `dataset-import` optional dependency group.
- Create generated `datasets/assets/vertical_spring_oscillator/<case>/...`: Case-local text and ignored media/masks.
- Create generated `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/*`: original workbook, normalized annotations, review decisions, and reproducibility inputs.
- Create generated `datasets/provenance/imports/vertical_spring_oscillator_20260806_{import_audit.jsonl,exclusions.json,summary.json}`.
- Create `scripts/build_dataset_v13.py`: deterministic current-release builder.
- Rename/replace `scripts/validate_dataset_v12.py` with `scripts/validate_dataset_v13.py`: independent Dataset 13 validator.
- Replace `datasets/releases/12.0.0/` and `datasets/provenance/releases/12.0.0/` on the active filesystem with generated `13.0.0/` equivalents.
- Modify `src/physbench/data_layout.py`, `tests/test_current_dataset.py`, `datasets/releases/README.md`, `datasets/DATASET_OVERVIEW.md`, `datasets/README.md`, and root `README.md` for Dataset 13 and the seventh Scene.
- Modify official Task JSON files that name Dataset `12.0.0` so existing evaluated Scene selections resolve against Dataset `13.0.0`; do not claim a spring evaluator in this data-import change.
- Delete the approved design spec after all checks pass.

---

### Task 1: Intake and Formal Annotation Contracts

**Files:**
- Create: `src/physbench/datasets/vertical_spring_import.py`
- Create: `tests/test_vertical_spring_import.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `TrialRecord`, `SourceMember`, `ImportDecision`, `load_trial_records(workbook: Path) -> list[TrialRecord]`, `normalize_source_members(names: Iterable[str]) -> dict[str, list[SourceMember]]`, `build_physics(trial: TrialRecord, case_id: str) -> dict[str, Any]`, and `build_caption(trial: TrialRecord, case_id: str) -> dict[str, Any]`.
- Consumes: workbook sheets `实验设置`, `弹簧设置`, and `Trial记录` plus ZIP member names.

- [ ] **Step 1: Add optional import dependencies**

Add this exact group to `pyproject.toml`:

```toml
dataset-import = [
  "numpy>=1.26",
  "opencv-python-headless>=4.10",
  "openpyxl>=3.1",
]
```

- [ ] **Step 2: Install local test dependencies**

Run:

```bash
apt-get update
apt-get install -y python3-numpy python3-opencv python3-openpyxl
```

Expected: `python3` can import `numpy`, `cv2`, and `openpyxl`. These are host dependencies, not repository files.

- [ ] **Step 3: Write failing workbook and physics tests**

Create test fixtures in a temporary directory with OpenPyXL and assert the signed displacement is retained only by `TrialRecord` while formal physics is non-negative:

```python
def test_physics_uses_absolute_displacement_and_caption_keeps_direction(self):
    below = TrialRecord("T001", "S01", "IMG_1518.MOV", 40.0, 12)
    above = TrialRecord("T002", "S01", "IMG_1519.MOV", -40.0, 13)
    physics = build_physics(above, "spring_t002_img1519")
    quantity = physics["physics"]["objects"]["object_1"]["initial_displacement"]
    self.assertEqual({"value": 0.04, "unit": "m", "symbol": "x_0"}, quantity)
    self.assertIn("below equilibrium", build_caption(below, "a")["caption"])
    self.assertIn("above equilibrium", build_caption(above, "b")["caption"])
    self.assertNotIn("42.55", json.dumps(physics))
```

- [ ] **Step 4: Run the tests and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: FAIL because `vertical_spring_import` does not exist.

- [ ] **Step 5: Implement typed intake and serializers**

Implement frozen dataclasses and explicit field validation. `load_trial_records` must use `data_only=False` and preserve raw cell values in an adjacent normalized provenance record while rejecting rows without Trial ID, video stem, spring `S01`, or non-zero signed displacement. `build_physics` must emit only `case_id`, `scene_id`, and grouped `physics`; `build_caption` must emit only `case_id`, `scene_id`, and `caption`.

Use the stable field names and symbols from Global Constraints. Define `source_stem` by stripping directories, the optional `(1)` duplicate suffix, and a case-insensitive `.MOV` extension without altering the original member name.

- [ ] **Step 6: Add mapping and duplicate tests**

```python
def test_member_mapping_retains_duplicate_candidates(self):
    mapping = normalize_source_members([
        "root/IMG_1705.MOV", "root/IMG_1705(1).MOV", "root/IMG_1706.MOV"
    ])
    self.assertEqual(2, len(mapping["IMG_1705"]))
    self.assertEqual("root/IMG_1705.MOV", mapping["IMG_1705"][0].member_name)
```

- [ ] **Step 7: Run focused tests**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: all Task 1 tests PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add pyproject.toml src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): define vertical spring import contract"
```

### Task 2: Trajectory Detector and Ball Mask

**Files:**
- Modify: `src/physbench/datasets/vertical_spring_import.py`
- Modify: `tests/test_vertical_spring_import.py`

**Interfaces:**
- Consumes: `Sequence[TrackedPoint]`, signed displacement, timestamps, and a portrait BGR first frame.
- Produces: `track_ball(frames: Iterable[np.ndarray], times_s: Sequence[float]) -> list[TrackedPoint]`, `select_post_cycle_turning_frame(points: Sequence[TrackedPoint], release_side: Literal["above", "below"]) -> TurningCandidate`, and `ball_disk_mask(frame: np.ndarray, circle: BallCircle) -> np.ndarray`.

- [ ] **Step 1: Write failing synthetic trajectory tests**

```python
def test_selects_first_same_side_turn_after_opposite_turn(self):
    times = np.arange(0.0, 2.0, 1 / 240)
    y = 500.0 + 80.0 * np.cos(2 * np.pi * times / 0.8)
    points = [TrackedPoint(i, float(t), 300.0, float(v), 45.0, 1.0)
              for i, (t, v) in enumerate(zip(times, y))]
    candidate = select_post_cycle_turning_frame(points, "below")
    self.assertLess(abs(candidate.time_s - 0.8), 1 / 120)

def test_above_release_selects_upper_turn(self):
    times = np.arange(0.0, 2.0, 1 / 240)
    y = 500.0 - 80.0 * np.cos(2 * np.pi * times / 0.8)
    points = [TrackedPoint(i, float(t), 300.0, float(v), 45.0, 1.0)
              for i, (t, v) in enumerate(zip(times, y))]
    candidate = select_post_cycle_turning_frame(points, "above")
    self.assertLess(abs(candidate.time_s - 0.8), 1 / 120)
```

- [ ] **Step 2: Run trajectory tests and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import.VerticalSpringTrajectoryTests -v`
Expected: FAIL because detector types/functions are absent.

- [ ] **Step 3: Implement robust one-dimensional turning detection**

Implement median filtering followed by a symmetric moving-average analysis signal; retain original frame indices and timestamps. Require, in order: initial release-side support, one opposite-side extremum, then one same-side extremum within `0.55 <= elapsed_s <= 1.20`. Reject gaps longer than eight decoded frames, motion amplitude below 12 analysis pixels, fewer than 80% confident circle observations, or an extremum within eight frames of a track boundary. Snap the candidate to the unsmoothed tracked point with the most extreme `y` inside the local window.

- [ ] **Step 4: Write failing circle and mask tests**

Generate a 480x270 synthetic portrait frame with a 25-pixel disk and assert centre/radius recovery and binary disk-mask shape. Assert that a rectangular connector above the disk is not included.

- [ ] **Step 5: Implement tracking and mask creation**

Use grayscale/blurred `cv2.HoughCircles` inside a vertically broad central ROI, with radius bounds derived from frame width (`0.045w` to `0.10w`). Seed subsequent frames from the previous circle and fall back to normalized template matching inside a bounded search window. `ball_disk_mask` must fill only the fitted circular silhouette; output dtype `uint8` with values `{0, 255}`.

- [ ] **Step 6: Run Task 2 tests**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): detect vertical spring turning frames"
```

### Task 3: Exact-Frame Media and Case Materialization

**Files:**
- Create: `scripts/import_vertical_spring_oscillator.py`
- Modify: `src/physbench/datasets/vertical_spring_import.py`
- Modify: `tests/test_vertical_spring_import.py`

**Interfaces:**
- Consumes: source ZIP, Trial mapping, `TurningCandidate`, optional reviewed overrides JSON, and repository root.
- Produces: `probe_member`, `extract_member`, `materialize_reference`, `decode_first_frame`, `write_mask_bundle`, `write_case`, staged CLI commands `intake`, `analyze`, `pilot`, `batch`, and `review-report`.

- [ ] **Step 1: Write failing exact-tail media test**

Create a 20-frame synthetic VFR-or-CFR fixture with a distinct frame index burned into pixel colour, select source frame 7, and assert canonical output has 13 frames, frame zero matches source frame 7 after decode tolerance, dimensions are unchanged, codec is H.264, and final PTS duration matches the retained source interval within one source timebase tick.

- [ ] **Step 2: Run media test and verify failure**

Run: `python3 -m unittest tests.test_vertical_spring_import.VerticalSpringMediaTests -v`
Expected: FAIL because the CLI/materializer is absent.

- [ ] **Step 3: Implement FFmpeg orchestration**

Use argument lists with `subprocess.run(check=True)`; never compose shell strings. Decode analysis frames after autorotation and downscale only the analysis stream. For canonical media use a frame-select/PTS filter beginning at the exact selected decoded frame, `-fps_mode passthrough`, `libx264`, `-pix_fmt yuv420p`, and `-movflags +faststart`. Do not pass `-r`, `fps`, `scale`, or an end time. Record source and canonical FFprobe JSON and the exact filter in the import audit.

- [ ] **Step 4: Implement Case and mask writers**

Write canonical `first_frame.png` from the final MP4 frame zero. Write `01.png`, compressed `01.npz` key `mask`, and `manifest.json` with `schema_version`, `case_id`, `scene_id`, first-frame path, and an `object_1` entry pointing to both masks and `objects.object_1` physics fields. Use atomic temporary sibling files followed by `Path.replace` for text and small mask assets.

- [ ] **Step 5: Implement staged CLI and idempotence**

Require explicit `--archive`, `--repo-root`, and `--work-root`. `intake` may copy the source archive and workbook; `analyze` writes candidate JSONL only; `pilot` and `batch` refuse to overwrite an existing Case whose audit identity differs; `review-report` builds contact sheets with the first frame, trajectory plot, selected-frame neighbourhood, and mask overlay.

- [ ] **Step 6: Run Task 3 tests**

Run: `python3 -m unittest tests.test_vertical_spring_import -v`  
Expected: all tests PASS, including an FFmpeg skip only when FFmpeg is unavailable.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/import_vertical_spring_oscillator.py src/physbench/datasets/vertical_spring_import.py tests/test_vertical_spring_import.py
git commit -m "feat(dataset): materialize vertical spring cases"
```

### Task 4: Intake, Pilot, Review, and Full Batch

**Files:**
- Generate: `datasets/assets/vertical_spring_oscillator/*`
- Generate: `datasets/provenance/source_docs/20260806_vertical_spring_oscillator/*`
- Generate: `datasets/provenance/imports/vertical_spring_oscillator_20260806_*`

**Interfaces:**
- Consumes: the Task 3 CLI and `/root/Steven/竖直弹簧振子.zip`.
- Produces: reviewed accepted Cases, exact exclusions, normalized annotations, and import summary.

- [ ] **Step 1: Record import dependency versions**

Record `python3`, NumPy, OpenCV, OpenPyXL, FFmpeg, and FFprobe versions in the import summary. Do not add environment directories to Git.

- [ ] **Step 2: Run intake and exact duplicate comparison**

Run `intake`, then compare `IMG_1705.MOV` and `IMG_1705(1).MOV` by FFprobe metadata and decoded-frame equality. Record the deterministic selected member or exclusion; confirm `IMG_1652`–`IMG_1654` absence and all empty Trial rows in exclusions.

- [ ] **Step 3: Run analysis for every uniquely mapped Trial**

Generate candidate JSONL and a failure list. Confirm every candidate records source row, signed displacement, direction, frame index, timestamp, cycle extrema, observation coverage, and circle geometry.

- [ ] **Step 4: Materialize a stratified pilot**

Choose at least six candidates covering positive and negative displacement, low/medium/high magnitude, and distinct capture backgrounds. Run `pilot` without changing detector thresholds per Case.

- [ ] **Step 5: Review every pilot and correct shared rules**

Inspect each pilot's contact sheet and mask overlay. Record `accepted`, `corrected_start_frame`, `corrected_circle`, or an exact exclusion reason in `review_decisions.json`. If a shared detector change is needed, add a failing regression test before changing code, rerun all import tests, and recommit.

- [ ] **Step 6: Materialize the full batch**

Run `batch` using reviewed detector settings and any explicit per-Case review corrections. Keep the output process resumable and poll it at intervals short enough to report progress.

- [ ] **Step 7: Perform full visual review**

Generate paginated review sheets and inspect every accepted Case for correct post-cycle side, hand/tool absence, complete ball, untrimmed tail, direction Caption, and steel-ball-only mask. Move any failure to exclusions and remove its partial Case with the CLI's exact-case cleanup action.

- [ ] **Step 8: Validate generated Case assets**

Run the CLI validation stage. Expected: zero unreviewed Cases, zero identity/path mismatches, zero negative quantities, zero Caption symbol gaps, zero frame-zero mismatches, zero invalid masks, and accepted plus excluded source decisions equal the eligible Trial count.

- [ ] **Step 9: Commit tracked import records and Case text**

```bash
git add datasets/assets/vertical_spring_oscillator datasets/provenance/source_docs/20260806_vertical_spring_oscillator datasets/provenance/imports/vertical_spring_oscillator_20260806_*
git commit -m "data: import vertical spring oscillator cases"
```

Confirm ignored media/archive files remain present locally but are not staged.

### Task 5: Build and Validate Dataset 13.0.0

**Files:**
- Create: `scripts/build_dataset_v13.py`
- Create: `scripts/validate_dataset_v13.py`
- Modify: `src/physbench/data_layout.py`
- Modify: `tests/test_current_dataset.py`
- Replace: `datasets/releases/12.0.0/*` with `datasets/releases/13.0.0/*`
- Replace: `datasets/provenance/releases/12.0.0/*` with `datasets/provenance/releases/13.0.0/*`
- Modify: `tasks/official/*.json` entries that name release `12.0.0`

**Interfaces:**
- Consumes: frozen Dataset 12 records from Git, accepted Case assets/import audits, and review decisions.
- Produces: deterministic Dataset 13 descriptor, case/provenance indexes, seven Scene configs, View A train/ID split, View B complete groups, and `V13_DATASET`/`LATEST_DATASET` constants.

- [ ] **Step 1: Write failing Dataset 13 tests**

Update `tests/test_current_dataset.py` to assert release `13.0.0`, seven Scene configs, old 799 Case IDs retained, spring Case count equals the accepted import summary, test count at most 20, every spring Case appears exactly once in train/test and complete groups, all spring test annotations are ID, and `LATEST_DATASET == V13_DATASET`.

- [ ] **Step 2: Run current Dataset tests and verify failure**

Run: `python3 -m unittest tests.test_current_dataset -v`
Expected: FAIL because Dataset 13 does not exist.

- [ ] **Step 3: Implement deterministic Dataset 13 builder**

Copy existing Case rows and Scene configs semantically, append spring rows in stable `case_id` order, and create the spring Scene config with structured paths for `mass`, `radius`, `initial_displacement`, `spring_stiffness`, `natural_spring_length`, and `gravity_acceleration`. Split spring Cases by a deterministic near-duplicate group key derived from direction, displacement magnitude, capture background/session, and recorded duplicate group; select whole groups for an ID test set no larger than 20. Recompute case-set digests already required by View schemas; do not hash assets.

- [ ] **Step 4: Build the release and switch the runtime constant**

Run `scripts/build_dataset_v13.py`, update `src/physbench/data_layout.py` to define `V13_RELEASE_ROOT`, `V13_DATASET`, and `LATEST_DATASET = V13_DATASET`, and remove active `12.0.0` runtime/provenance directories only after `13.0.0` is complete.

- [ ] **Step 5: Implement the independent validator**

Adapt the structural checks from v12 to calculate expected counts from the import summary and enforce exactly one active release directory named `13.0.0`. Validate every Case-local physics/Caption/mask link, provenance coverage, Scene count, View coverage, absence of asset locks/hashes, and preservation of all 799 prior Case identities.

- [ ] **Step 6: Update official Task Dataset references**

Rename task IDs/version text from v12 to v13 where they describe the Dataset snapshot, point their descriptors at `datasets/releases/13.0.0/dataset.json`, and retain their existing evaluated Scene selections. The new spring Scene remains Dataset-accessible without falsely declaring an evaluator that this import does not implement.

- [ ] **Step 7: Run release-focused tests**

Run:

```bash
python3 scripts/validate_dataset_v13.py
python3 -m unittest tests.test_current_dataset tests.test_flat_dataset_layout tests.test_dataset_contract_v5 -v
```

Expected: all PASS.

- [ ] **Step 8: Commit Dataset 13**

```bash
git add datasets/releases datasets/provenance/releases scripts/build_dataset_v13.py scripts/validate_dataset_v13.py src/physbench/data_layout.py tests/test_current_dataset.py tasks/official
git commit -m "data: publish vertical spring Dataset v13"
```

### Task 6: Documentation, Full Verification, and Requested Spec Removal

**Files:**
- Modify: `README.md`
- Modify: `datasets/README.md`
- Modify: `datasets/DATASET_OVERVIEW.md`
- Modify: `datasets/releases/README.md`
- Delete: `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md`

**Interfaces:**
- Consumes: verified import summary and Dataset 13 outputs.
- Produces: accurate public documentation and a clean, tested repository state.

- [ ] **Step 1: Write documentation assertions first**

Extend current Dataset tests to require that public docs name Dataset `13.0.0`, list `vertical_spring_oscillator`, and report Case counts equal to generated summary plus the prior 799.

- [ ] **Step 2: Run assertions and verify failure**

Run: `python3 -m unittest tests.test_current_dataset -v`
Expected: FAIL on stale v12/six-Scene documentation.

- [ ] **Step 3: Update public documentation**

Document the seventh Scene, its formal physics fields, accepted/excluded counts, Dataset 13 single-release rule, ignored-media download expectation, and the fact that this import publishes data but does not claim a dedicated spring evaluator.

- [ ] **Step 4: Run targeted validation**

Run:

```bash
python3 scripts/validate_dataset_v13.py
python3 -m unittest tests.test_vertical_spring_import tests.test_current_dataset -v
git diff --check
```

Expected: all PASS and no whitespace errors.

- [ ] **Step 5: Use the verification-before-completion skill and run the full suite**

Run: `make test`  
Expected: PASS. If unavailable dependencies cause documented skips, confirm none of the Dataset/import tests are skipped.

- [ ] **Step 6: Delete the approved design document**

Delete only `docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md` after all validation passes. Do not delete this implementation plan or the unrelated untracked plan.

- [ ] **Step 7: Commit documentation and requested deletion**

```bash
git add README.md datasets/README.md datasets/DATASET_OVERVIEW.md datasets/releases/README.md tests/test_current_dataset.py docs/superpowers/specs/2026-08-06-vertical-spring-oscillator-import-design.md
git commit -m "docs: publish Dataset v13 import status"
```

- [ ] **Step 8: Final repository audit**

Run `git status --short`, verify only the pre-existing unrelated untracked physical-response-loss plan remains, and report accepted/excluded counts, Dataset 13 Case count, test evidence, local ignored media locations, commits, and spec deletion.
