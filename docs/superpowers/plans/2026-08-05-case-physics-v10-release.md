# Case-local Physics and Minimal Dataset 10.0.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind unchanged structured physics to every active asset Case through a locked `physics.json`, remove 32 verified legacy collision directories, and publish a minimal Dataset 10.0.0 as the official default.

**Architecture:** Dataset 9.0.0 remains the immutable base. A V10 builder derives one self-identifying physics document per active Case, adds `assets.physics_annotation`, creates a seven-component runtime release, and writes build/validation evidence under provenance. The generic loader validates file-to-Case identity and exact physics equality whenever the new asset role is present, while old releases retain their existing behavior.

**Tech Stack:** Python 3.10, `unittest`, JSON/JSONL, SHA-256 asset locks, POSIX atomic rename, Git.

## Global Constraints

- Work in the current checkout as authorized; do not create a media-copying worktree.
- Preserve all 799 ordered 9.0.0 Case identities and every inline `physics` object exactly.
- Do not rewrite video, image, mask, NPZ, source archive, text, Scene, View, temporal, appearance, alignment, or provenance facts.
- Do not modify historical releases 1.0.0 through 9.0.0.
- Admit only Case-local `physics.json` below the ignored asset tree; do not track media.
- Delete only the exact 32 enumerated legacy `collision_r2_*` directories after a complete read-only preflight.
- Keep 10.0.0 limited to `README.md`, `dataset.json`, `release.json`, `cases.jsonl`, `assets.lock.json`, `scenes/`, and `views/`.
- Store V10 migration and validation evidence in `datasets/provenance/releases/10.0.0/`.
- Make `physics_video_six_scene_v10` the official default and migrate both official Tasks without changing their selections, seeds, or evaluation protocol.

---

### Task 1: Enforce the Case-local physics asset contract in the loader

**Files:**
- Modify: `tests/test_dataset_maintenance_scripts.py`
- Modify: `src/physbench/datasets/loader.py`

**Interfaces:**
- Consumes: a Case with optional `assets.physics_annotation` and the resolved Dataset `asset_root`.
- Produces: `_validate_case_physics_annotation(case: dict[str, Any], asset_root: Path) -> None`, called by `load_dataset` for every Case after resolving `asset_root`.

- [ ] **Step 1: Extend the test Dataset writer with a Case-local physics document**

Add an optional `physics_document` argument to `AssetLockBuilderSafetyTests._write_dataset`. When supplied, write `assets/physics.json` and add `"physics_annotation": "physics.json"` to the Case assets. The valid literal document is:

```python
{
    "schema_version": "1.0",
    "case_id": "pendulum_case_1",
    "scene_id": "pendulum",
    "physics": {
        "gravity": {
            "value": 9.8,
            "unit": "m/s^2",
            "annotated": True,
        }
    },
}
```

- [ ] **Step 2: Write failing behavior tests**

Add tests proving that:

```python
def test_matching_case_local_physics_loads_and_is_locked(self): ...
def test_case_local_physics_mismatch_is_rejected_without_asset_checks(self): ...
def test_missing_case_local_physics_is_rejected_without_asset_checks(self): ...
def test_case_local_physics_identity_and_schema_are_exact(self): ...
```

The mismatch test changes only `physics.gravity.value` in the file to `9.81` and expects `ValueError` containing `differs from inline case.physics`. The missing-file test deletes `assets/physics.json` and expects `FileNotFoundError`. Identity/schema subtests independently alter `schema_version`, `case_id`, `scene_id`, and add an unknown top-level key.

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_dataset_maintenance_scripts.AssetLockBuilderSafetyTests -v
```

Expected: the semantic mismatch and missing-file tests fail because the current loader treats the role as an opaque optional asset when asset checks are disabled.

- [ ] **Step 4: Implement exact document validation**

In `src/physbench/datasets/loader.py`, add:

```python
def _validate_case_physics_annotation(
    case: dict[str, Any],
    asset_root: Path,
) -> None:
    relative = case["assets"].get("physics_annotation")
    if relative is None:
        return
    path = (asset_root / relative).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ValueError(
            f"case {case['case_id']} physics annotation escapes asset_root"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"case {case['case_id']} missing assets.physics_annotation: {path}"
        )
    document = load_json(path)
    expected_fields = {"schema_version", "case_id", "scene_id", "physics"}
    if set(document) != expected_fields:
        raise ValueError(
            f"case {case['case_id']} physics annotation fields must be "
            f"{sorted(expected_fields)}"
        )
    if document["schema_version"] != "1.0":
        raise ValueError(
            f"case {case['case_id']} physics annotation schema must be 1.0"
        )
    if document["case_id"] != case["case_id"]:
        raise ValueError(f"case {case['case_id']} physics annotation Case mismatch")
    if document["scene_id"] != case["scene_id"]:
        raise ValueError(f"case {case['case_id']} physics annotation Scene mismatch")
    if document["physics"] != case["physics"]:
        raise ValueError(
            f"case {case['case_id']} physics annotation differs from inline "
            "case.physics"
        )
```

Resolve `asset_root` before loading the asset lock, call this helper for every validated Case, and leave Cases without the role unchanged.

- [ ] **Step 5: Run loader and maintenance tests and verify GREEN**

Run the Step 3 command. Expected: all tests pass.

- [ ] **Step 6: Commit the loader contract**

```bash
git add src/physbench/datasets/loader.py tests/test_dataset_maintenance_scripts.py
git commit -m "feat(dataset): validate case-local physics assets"
```

### Task 2: Build deterministic V10 Cases and preflight legacy cleanup

**Files:**
- Create: `scripts/build_dataset_v10.py`
- Create: `tests/test_case_local_physics_v10.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `datasets/releases/9.0.0`, the current asset tree, and releases 1.0.0–9.0.0 for reference auditing.
- Produces: `PhysicsWrite`, `LegacyRemoval`, `prepare_v10_cases(...)`, `preflight_legacy_cleanup(...)`, `materialize_physics_documents(...)`, and `build_release(...)`.

- [ ] **Step 1: Add failing tests for one-to-one Case directory derivation**

Create `tests/test_case_local_physics_v10.py`. Load the builder module and assert:

```python
def test_v9_has_799_unique_case_directories():
    cases = builder.read_jsonl(builder.BASE_RELEASE_ROOT / "cases.jsonl")
    writes, output_cases = builder.prepare_v10_cases(cases, builder.DATASETS_ROOT)
    self.assertEqual(799, len(writes))
    self.assertEqual(799, len({item.relative_path for item in writes}))
    self.assertTrue(all(
        case["assets"]["physics_annotation"].endswith("/physics.json")
        for case in output_cases
    ))
```

Add a temporary-fixture test where `first_frame` and `reference_video` point to different Case directories and require `ValueError` containing `does not resolve to one Case directory`.

- [ ] **Step 2: Add failing tests for deterministic physics bytes and conflict refusal**

Assert that a `PhysicsWrite.payload` decodes to exactly the four-field document from the design, ends in one newline, and is byte-stable over two preparations. In a temporary asset tree, write conflicting bytes at the target and require `materialize_physics_documents` to abort before writing any missing document.

- [ ] **Step 3: Add failing tests for the exact 32-directory cleanup preflight**

On the real read-only tree, assert that `preflight_legacy_cleanup` returns 32 unique `LegacyRemoval` records, each named `collision_r2_*`, each containing one `source/first_frame_source.png`, and each mapping to an active V9 Case whose current directory differs. Add a temporary fixture with an extra file and require preflight to fail before deletion.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_case_local_physics_v10 -v
```

Expected: import failure because `scripts/build_dataset_v10.py` does not exist.

- [ ] **Step 5: Implement the V10 builder's pure preparation layer**

Implement frozen dataclasses:

```python
@dataclass(frozen=True)
class PhysicsWrite:
    case_id: str
    relative_path: str
    absolute_path: Path
    payload: bytes

@dataclass(frozen=True)
class LegacyRemoval:
    case_id: str
    relative_directory: str
    absolute_directory: Path
    relative_file: str
    size_bytes: int
    sha256: str
```

`prepare_v10_cases` derives the Case directory independently from every non-null `first_frame`, `reference_video`, and `physics_reference_video`, requires a single `assets/<scene>/<case>` parent, serializes the self-identifying document with `ensure_ascii=False`, `indent=2`, `sort_keys=True`, and a final newline, and deep-copies the Case before adding the asset role.

`materialize_physics_documents` first verifies all existing candidates: byte-identical files are accepted, any conflict aborts, and only then are missing files written through a temporary sibling plus `Path.replace`.

`preflight_legacy_cleanup` enumerates immediate `collision_r2_*` directories, excludes all current V9 Case directories, requires exactly 32 candidates, validates exact directory content, proves the remaining PNG path is absent from all 1.0.0–9.0.0 Case asset values, maps the directory name to an active Case ID, and records size/SHA-256.

- [ ] **Step 6: Update ignore rules without exposing media**

Change `.gitignore` so Git descends through scene and Case directories but admits only `physics.json` at the Case root. Add a test invoking `git check-ignore` to prove a generated `physics.json` is not ignored while representative `canonical/reference.mp4`, `canonical/masks/01.npz`, and `source/reference.mov` remain ignored.

- [ ] **Step 7: Run the new tests and verify GREEN**

Run the Step 4 command. Expected: all tests pass and the real tree remains unchanged.

- [ ] **Step 8: Commit builder preparation and safety checks**

```bash
git add .gitignore scripts/build_dataset_v10.py tests/test_case_local_physics_v10.py
git commit -m "feat(dataset): prepare case-local physics release"
```

### Task 3: Publish and independently validate the minimal V10 release

**Files:**
- Modify: `scripts/build_dataset_v10.py`
- Create: `scripts/validate_dataset_v10.py`
- Modify: `tests/test_case_local_physics_v10.py`
- Create: `datasets/assets/<scene>/<case>/physics.json` for exactly 799 active Cases
- Create: `datasets/releases/10.0.0/README.md`
- Create: `datasets/releases/10.0.0/dataset.json`
- Create: `datasets/releases/10.0.0/release.json`
- Create: `datasets/releases/10.0.0/cases.jsonl`
- Create: `datasets/releases/10.0.0/assets.lock.json`
- Create: `datasets/releases/10.0.0/scenes/*.json`
- Create: `datasets/releases/10.0.0/views/*.json`
- Create: `datasets/provenance/releases/10.0.0/migration.json`
- Create: `datasets/provenance/releases/10.0.0/validation.json`
- Delete: the exact 32 preflighted `datasets/assets/collision_1d/collision_r2_*` legacy directories

**Interfaces:**
- Consumes: Task 2 preparation records and `scripts.build_dataset_asset_lock.rebuild_asset_lock`.
- Produces: `physics_video_six_scene_v10`, 6,038 locked assets, two provenance evidence documents, and a standalone validator.

- [ ] **Step 1: Add failing release-shape and invariance tests**

Add tests requiring:

```python
EXPECTED_RELEASE_ENTRIES = {
    "README.md", "dataset.json", "release.json", "cases.jsonl",
    "assets.lock.json", "scenes", "views",
}
```

The V10 release must have exactly those entries; 799 Cases; 799 unique physics paths; 6,038 locked assets; all V9 locked path records unchanged; and each V10 Case must equal its V9 counterpart after removing only `assets.physics_annotation`. Compute the compact sorted-key SHA-256 of ordered `(case_id, physics)` pairs and require equality between V9 and V10.

- [ ] **Step 2: Run the release tests and verify RED**

Run the Task 2 test command. Expected: failure because 10.0.0 has not been published.

- [ ] **Step 3: Implement staged release construction**

Extend `build_release` to:

1. preflight all Case writes and legacy removals;
2. materialize byte-stable physics documents;
3. create a temporary sibling below `datasets/releases`;
4. copy only `scenes/` and `views/` from 9.0.0;
5. write `cases.jsonl`, a descriptor without `mask_annotations`, and a README explaining every retained entry;
6. invoke `rebuild_asset_lock` on the staged descriptor;
7. require 6,038 lock entries and a successful `check_asset_hashes=True` load;
8. write `migration.json` with base/output identities, physics fingerprints, counts, runtime entries, and every `LegacyRemoval` record;
9. rename the staged directory to `datasets/releases/10.0.0` only if the target is absent;
10. delete only the enumerated legacy directories;
11. invoke the standalone validator to write `validation.json`.

The builder refuses an existing 10.0.0 directory unless `--check` is used. `--check` performs all preflight and candidate comparisons without writing or deleting.

- [ ] **Step 4: Implement independent V10 validation**

`scripts/validate_dataset_v10.py` must expose:

```python
def validate_v10(
    dataset_path: Path = V10_DATASET,
    *,
    write_report: bool = False,
) -> dict[str, Any]: ...
```

It checks every acceptance criterion without calling builder preparation helpers, loads V9 and V10 independently, verifies all 6,038 SHA-256 values, checks the exact release tree, compares V9/V10 Case facts and locked paths, proves 799 documents equal inline physics, and confirms every migration-recorded legacy directory is absent. When requested, it writes the stable result to `datasets/provenance/releases/10.0.0/validation.json`.

- [ ] **Step 5: Materialize V10 and perform the authorized cleanup**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/build_dataset_v10.py
```

Expected: `cases=799 physics=799 locked_assets=6038 removed_legacy_dirs=32`.

- [ ] **Step 6: Run independent validation and release tests**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/validate_dataset_v10.py --write-report
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_case_local_physics_v10 -v
```

Expected: validation and all tests pass.

- [ ] **Step 7: Stage only metadata and verify no media additions**

Run:

```bash
git add scripts/build_dataset_v10.py scripts/validate_dataset_v10.py \
  tests/test_case_local_physics_v10.py datasets/releases/10.0.0 \
  datasets/provenance/releases/10.0.0
git add datasets/assets/*/*/physics.json
git diff --cached --name-only --diff-filter=A -z | \
  xargs -0 -r du -b | awk '$1 > 10485760 {print; bad=1} END {exit bad}'
```

Expected: no staged addition exceeds 10 MiB and no media extension is staged.

- [ ] **Step 8: Commit the V10 release**

```bash
git commit -m "data: publish case-local physics release v10"
```

After deletion, report that the 32 ignored PNG remnants are no longer directly recoverable from Git; their path, size, and SHA-256 remain in migration evidence.

### Task 4: Make V10 the official Dataset and Task default

**Files:**
- Modify: `src/physbench/data_layout.py`
- Modify: `tasks/official/five_scene_direct_eval.json`
- Modify: `tasks/official/five_scene_finetune_eval.json`
- Modify: `tests/test_six_scene_dataset_v8.py`
- Modify: `tests/test_task_builder.py`
- Modify: `tests/test_five_scene_dataset_v7.py`
- Modify: current-default assertions found by `rg "LATEST_DATASET|V8_DATASET" tests`

**Interfaces:**
- Consumes: the validated 10.0.0 descriptor and Dataset ID.
- Produces: `V10_RELEASE_ROOT`, `V10_DATASET`, `LATEST_DATASET = V10_DATASET`, and official V10 Task identities.

- [ ] **Step 1: Write failing current-default and Task tests**

Add assertions that:

```python
self.assertEqual(V10_DATASET, LATEST_DATASET)
self.assertEqual("physics_video_six_scene_v10", dataset.dataset_id)
self.assertEqual("10.0.0", dataset.descriptor["release"])
self.assertEqual("five_scene_direct_eval_v10", direct_task.task_id)
self.assertEqual("five_scene_finetune_eval_v10", finetune_task.task_id)
```

Retain the existing plan counts: 582 finetune train Cases, 76 finetune jobs, and 658 direct jobs.

- [ ] **Step 2: Run affected tests and verify RED**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_six_scene_dataset_v8 tests.test_task_builder \
  tests.test_five_scene_dataset_v7 -v
```

Expected: current-default and Task identity assertions fail on V8.

- [ ] **Step 3: Switch runtime constants and official Task identities**

Add V10 constants in `data_layout.py`, set `LATEST_DATASET = V10_DATASET`, change both Task IDs from `_v8` to `_v10`, and change both Task `dataset_id` values to `physics_video_six_scene_v10`. Do not change any other Task field.

- [ ] **Step 4: Align current-default tests while preserving historical V8 coverage**

Load V10 for tests of the current release and official Tasks. Keep explicit V8 tests for its historical Case/asset counts, but replace assertions that V8 is current with `self.assertNotEqual(V8_DATASET, LATEST_DATASET)`.

- [ ] **Step 5: Run affected and maintenance tests and verify GREEN**

Run the Step 2 command plus `tests.test_dataset_maintenance_scripts`. Expected: all pass.

- [ ] **Step 6: Commit the official-default migration**

```bash
git add src/physbench/data_layout.py tasks/official tests
git commit -m "feat(dataset): make v10 the official default"
```

### Task 5: Synchronize all current documentation and verify the repository

**Files:**
- Modify: `README.md`
- Modify: `Makefile`
- Modify: `datasets/README.md`
- Modify: `datasets/assets/README.md`
- Modify: `datasets/releases/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DATASET.md`
- Modify: `docs/DATASET_INGESTION.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/BASELINE_INTEGRATION.md`
- Modify: active Baseline READMEs and current operational docs found by repository search

**Interfaces:**
- Consumes: final V10 identity, digest, asset counts, paths, and provenance files.
- Produces: consistent user-facing instructions for Case-local physics and the slim release boundary.

- [ ] **Step 1: Update current documentation**

Document:

- `datasets/releases/10.0.0/dataset.json` as the official entry;
- 799 Cases and 6,038 locked assets;
- `assets/<scene>/<case>/physics.json` and `assets.physics_annotation`;
- inline `case.physics` as the compatibility runtime API with mandatory equality;
- the seven-entry minimal release layout;
- build/validation evidence under `datasets/provenance/releases/10.0.0`;
- historical V8/V9 status and the reason `masks.jsonl` is absent from V10;
- the removal and non-recoverability of the 32 legacy PNG-only directories.

Do not rewrite frozen runs, results, historical provenance records, experiment reports, or historical release metadata.

- [ ] **Step 2: Search for stale current-version references**

Run:

```bash
rg -n "datasets/releases/8\.0\.0|physics_video_six_scene_v8|_eval_v8|LATEST_DATASET = V8" \
  README.md Makefile datasets docs src scripts tasks baselines tests \
  --glob '!datasets/releases/1.0.0/**' --glob '!datasets/releases/2.0.0/**' \
  --glob '!datasets/releases/3.0.0/**' --glob '!datasets/releases/4.0.0/**' \
  --glob '!datasets/releases/5.0.0/**' --glob '!datasets/releases/5.1.0/**' \
  --glob '!datasets/releases/6.0.0/**' --glob '!datasets/releases/7.0.0/**' \
  --glob '!datasets/releases/8.0.0/**' --glob '!datasets/releases/9.0.0/**' \
  --glob '!docs/superpowers/**'
```

Classify every remaining result as an explicit historical reference or fix it.

- [ ] **Step 3: Run release, Dataset, Task, loader, and syntax verification**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/validate_dataset_v10.py
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_case_local_physics_v10 \
  tests.test_dataset_maintenance_scripts \
  tests.test_six_scene_dataset_v8 \
  tests.test_task_builder \
  tests.test_five_scene_dataset_v7 -v
git diff --check
git diff --name-only --diff-filter=ACMR HEAD -- '*.py' | \
  xargs -r /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m py_compile
```

Expected: all targeted checks pass.

- [ ] **Step 4: Run the full unit suite and compare the known baseline**

Run:

```bash
PYTHONPATH=src:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest discover -s tests -v
```

The pre-change baseline is 522 tests with 21 failures, 64 errors, and 12 skips. New V10 tests increase the total test count; there must be no new failure/error category caused by this work.

- [ ] **Step 5: Verify Git scope and commit documentation**

Require a clean diff check, no untracked non-ignored files, no staged media, exactly 799 tracked `physics.json` additions, and an absent set of the 32 recorded legacy directories. Then commit:

```bash
git add README.md Makefile datasets/README.md datasets/assets/README.md \
  datasets/releases/README.md docs baselines
git commit -m "docs(dataset): document case-local physics v10"
```

- [ ] **Step 6: Final post-commit verification**

Run `git status --short --branch`, `scripts/validate_dataset_v10.py`, and the targeted suite once more. Report commit IDs, final Dataset digest, physics fingerprint, locked asset count, removed-directory evidence path, and the unchanged known full-suite failure baseline.
