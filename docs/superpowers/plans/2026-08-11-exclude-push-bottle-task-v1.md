# Exclude Push Bottle from Task v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sole official finetune Task v1 select six training scenes and five evaluation scenes while retaining every `push_bottle` Dataset case and asset.

**Architecture:** Keep the Dataset and planner unchanged. Express the exclusion only in the benchmark-owned official Task selector, rename that Task so its identity describes the canonical plan, then update release metadata, consumers, and public documentation to the same identity and literal counts.

**Tech Stack:** JSON Task declarations, Python `unittest`, PhysBench canonical planner, Markdown release documentation, Make release gates.

## Global Constraints

- The official finetune Task ID and filename are `six_scene_train_five_scene_eval_v1`.
- Finetune training is exactly six scenes and 679 View A train cases.
- Finetune evaluation remains exactly five scored scenes and 76 View A ID test cases.
- Direct evaluation remains exactly five scenes and 658 cases.
- All 127 train and 14 test `push_bottle` cases remain in Dataset 13.0.0 and Distribution v1.
- Do not modify Dataset descriptors, assets, distribution shards, Hub binding, schema, planner, evaluator, or baseline capabilities.
- Keep `schema_version: "1.0"` and `scene_default_v1`; do not create Task v2.

---

### Task 1: Correct the official Task v1 selector and identity

**Files:**
- Rename: `tasks/official/seven_scene_train_five_scene_eval_v1.json` to `tasks/official/six_scene_train_five_scene_eval_v1.json`
- Modify: `tests/test_current_dataset.py`
- Modify: `tests/test_release_v1_contract.py`
- Modify: `tests/test_architecture_v4.py`
- Modify: `tests/test_managed_baselines.py`
- Modify: `tests/test_task_builder.py`
- Modify: `tests/test_task_runtime_contracts.py`
- Modify: `tests/test_release_documentation.py`
- Modify: `RELEASE_MANIFEST.json`
- Modify: `README.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/BENCHMARK_PROTOCOL.md`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: Dataset `physics_video_seven_scene_v13`, View A split, `load_task(path)`, and `plan_atomic_task(task, dataset)`.
- Produces: official Task ID `six_scene_train_five_scene_eval_v1` whose canonical plan contains six training scenes, 679 train case IDs, five evaluation scenes, and 76 jobs.

- [ ] **Step 1: Write the failing canonical-plan regression**

In `tests/test_current_dataset.py`, keep loading the existing finetune file for the initial RED run, rename the test to `test_official_v1_tasks_exclude_push_bottle`, and replace the old training assertions with independently derived literals:

```python
self.assertEqual(6, len(finetune["training_scene_ids"]))
self.assertEqual(679, len(finetune["train_case_ids"]))
self.assertNotIn("push_bottle", finetune["training_scene_ids"])
self.assertEqual(
    679,
    sum(
        1
        for case in self.dataset.cases
        if case["case_id"] in set(finetune["train_case_ids"])
    ),
)
self.assertNotIn(
    "push_bottle",
    {
        case["scene_id"]
        for case in self.dataset.cases
        if case["case_id"] in set(finetune["train_case_ids"])
    },
)
self.assertNotIn("push_bottle", finetune["scene_ids"])
self.assertNotIn(
    "push_bottle",
    {job["scene_id"] for job in finetune["jobs"]},
)
```

Keep the existing Dataset split fixture and add explicit preservation assertions to `test_view_a_is_train_plus_id_test_only`:

```python
self.assertEqual(127, len(self.view["scenes"]["push_bottle"]["train"]))
self.assertEqual(14, len(self.view["scenes"]["push_bottle"]["test"]))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_current_dataset.CurrentDatasetTests.test_official_v1_tasks_exclude_push_bottle
```

Expected: FAIL because the current canonical plan has seven training scenes, 806 training cases, and includes `push_bottle`.

- [ ] **Step 3: Apply the minimal official declaration change**

Rename the Task file, set:

```json
"task_id": "six_scene_train_five_scene_eval_v1"
```

and make `selection.training_scene_ids` exactly:

```json
[
  "pendulum",
  "collision_1d",
  "inclined_plane_slide",
  "uniform_circular_motion",
  "parabolic_motion",
  "vertical_spring_oscillator"
]
```

Do not change `evaluation_scene_ids`, seeds, reporting, dataset identity, schema version, or evaluation protocol.

- [ ] **Step 4: Update contract consumers and literal expectations**

Change every active Task path and expected ID from `seven_scene_train_five_scene_eval_v1` to `six_scene_train_five_scene_eval_v1`. In `tests/test_release_v1_contract.py`, define training scenes independently:

```python
TRAINING_SCENES = SCORED_SCENES | {"vertical_spring_oscillator"}
```

Then assert `TRAINING_SCENES`, 679 train cases, unchanged five scored evaluation scenes, 76 jobs, and absence of `push_bottle` from both selected training cases and evaluation jobs. Keep `ALL_SCENES` only where it describes Dataset contents, or remove it if unused.

Update `tests/test_release_documentation.py` so `official_tasks` expects the renamed Task path. Do not edit tests that validate generic Dataset support for `push_bottle` or generic baseline capability fixtures.

- [ ] **Step 5: Update release metadata and human documentation**

In `RELEASE_MANIFEST.json`, replace the official Task path/key and set `training_cases` to `679`. Retain `push_bottle` under `preview_scenes`, because that field describes retained unscored Dataset content.

Update `README.md`, `docs/TASKS.md`, and `docs/BENCHMARK_PROTOCOL.md` to say six training scenes/679 cases and explain that `push_bottle` remains available in the Dataset but is selected by no official Task. Update `docs/ARCHITECTURE.md` from seven-scene to six-scene training. Do not rewrite the historical release-readiness design record.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_current_dataset \
  tests.test_release_v1_contract \
  tests.test_release_documentation \
  tests.test_architecture_v4 \
  tests.test_managed_baselines \
  tests.test_task_builder \
  tests.test_task_runtime_contracts
```

Expected: PASS with the six-scene canonical plan and all existing runtime contracts intact.

- [ ] **Step 7: Run pre-commit verification**

Run:

```bash
make test-evaluation
make smoke-interface
git diff --check
```

Expected: every command exits 0. The release archive gate is intentionally
deferred until after the commit because it audits an archive exported from
`HEAD`, not the uncommitted working tree.

- [ ] **Step 8: Verify scope and commit**

Run `git diff --stat`, `git diff -- tasks/official datasets RELEASE_MANIFEST.json`, and `git status --short`. Confirm that no file below `datasets/releases`, `datasets/assets`, or `datasets/distribution` changed. Commit only the Task declaration, references, tests, release metadata, and documentation:

```bash
git add README.md RELEASE_MANIFEST.json docs/ARCHITECTURE.md \
  docs/BENCHMARK_PROTOCOL.md docs/TASKS.md \
  docs/superpowers/plans/2026-08-11-exclude-push-bottle-task-v1.md \
  scripts/release_audit.py tasks/official tests
git commit -m "refactor: exclude push bottle from Task v1"
```

- [ ] **Step 9: Verify the committed release, push, and inspect hosted checks**

Run from the new `HEAD`:

```bash
make test-interface
make test-evaluation
make smoke-interface
make release-check
make release-archive-check
git diff --check
```

Push `2026-08-11`, then verify that the exact pushed SHA has a completed successful `clean-release` workflow, with both `interface-and-release` and `evaluation-contract` jobs completed successfully.
