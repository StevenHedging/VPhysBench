# Six-Scene Evaluation and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sole public Task v1 surface evaluate six scenes and publish a coherent user-documentation path for Baseline integration, DataAdapter behavior, V2V limitations, and Run result interpretation.

**Architecture:** Rename both official Task files and IDs so their names match their six-scene semantics, then let the existing planner and evaluator registry expand spring Cases through the unchanged `scene_default_v1` protocol. Treat public documentation as part of the release contract: focused tests lock Task counts, manifest identity, navigation, V2V status, and removal of model-specific residue.

**Tech Stack:** JSON Task/manifest contracts, Python `unittest`, existing Dataset/Task planner, Markdown public manuals, Make release checks.

## Global Constraints

- Official Task inventory contains exactly `six_scene_direct_eval_v1.json` and `six_scene_train_six_scene_eval_v1.json`.
- Direct evaluation selects exactly six scored scenes and `775` jobs, including `117` vertical-spring jobs.
- Finetune evaluation selects exactly six training scenes, `679` training Cases, six scored scenes, and `96` ID-test jobs including `20` vertical-spring jobs.
- `push_bottle` data and assets remain unchanged and are selected by neither official Task.
- Both Tasks keep `schema_version: "1.0"`, inference seed `42`, and protocol `scene_default_v1`.
- Managed V2V remains a reserved interface only; official Dataset 13.0.0 and official Tasks do not support it because Cases have no independently authorized cropped conditioning-prefix video.
- Reference/evaluator/source videos must never be documented as V2V conditioning substitutes.
- No Dataset, Hub binding, evaluator implementation, CSTI implementation/configuration, model Baseline, checkpoint, or runtime output changes.

---

### Task 1: Publish the six-scene Task v1 contract

**Files:**
- Rename: `tasks/official/five_scene_direct_eval_v1.json` to `tasks/official/six_scene_direct_eval_v1.json`
- Rename: `tasks/official/six_scene_train_five_scene_eval_v1.json` to `tasks/official/six_scene_train_six_scene_eval_v1.json`
- Modify: `RELEASE_MANIFEST.json`
- Modify: `scripts/release_audit.py`
- Modify: `Makefile`
- Modify: `tests/test_current_dataset.py`
- Modify: `tests/test_release_v1_contract.py`
- Modify: `tests/test_release_documentation.py`
- Modify: `tests/test_task_runtime_contracts.py`
- Modify: `tests/test_managed_baselines.py`
- Modify: `tests/test_architecture_v4.py`
- Modify: `tests/test_baseline_bundle_v5.py`
- Modify: `tests/test_task_builder.py`

**Interfaces:**
- Consumes: Dataset 13.0.0 View A scene partitions and existing `vertical_spring_oscillator_v1` protocol resolver.
- Produces: two official Task v1 identities whose canonical plans contain the exact six-scene counts in Global Constraints.

- [ ] **Step 1: Write the failing Task and release assertions**

Change active test constants to the two new filenames and assert literal Task IDs, exact official inventory, six scored scenes, direct job counts by scene (including `vertical_spring_oscillator: 117`), finetune `96` ID jobs (including `vertical_spring_oscillator: 20`), and complete `push_bottle` exclusion. Update release-documentation expectations to require the new manifest paths/counts, six scored scenes, and one preview scene.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH \
PYTHONPATH=src:tests:. python -m unittest \
  tests.test_current_dataset \
  tests.test_release_v1_contract \
  tests.test_release_documentation -v
```

Expected: failures because the new Task files do not exist and the release manifest still declares the five-scene identities/counts.

- [ ] **Step 3: Rename and minimally update the Task contracts**

Rename the files, replace their `task_id` values, append `vertical_spring_oscillator` to both evaluation selectors, and leave training selectors/seeds/protocol/reporting unchanged. Update all active test/runtime fixture paths, `Makefile`, and `scripts/release_audit.py` to the new filenames.

Update `RELEASE_MANIFEST.json` to:

```json
{
  "official_tasks": [
    "tasks/official/six_scene_direct_eval_v1.json",
    "tasks/official/six_scene_train_six_scene_eval_v1.json"
  ],
  "task_counts": {
    "six_scene_direct_eval_v1": {
      "training_cases": 0,
      "evaluation_jobs": 775
    },
    "six_scene_train_six_scene_eval_v1": {
      "training_cases": 679,
      "evaluation_jobs": 96
    }
  }
}
```

Retain the other manifest fields, move `vertical_spring_oscillator` into the six-item `scored_scenes`, and leave only `push_bottle` in `preview_scenes`.

- [ ] **Step 4: Run focused and dependent Task tests**

Run the Step 2 command plus:

```bash
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH \
PYTHONPATH=src:tests:. python -m unittest \
  tests.test_task_runtime_contracts \
  tests.test_managed_baselines \
  tests.test_architecture_v4 \
  tests.test_baseline_bundle_v5 \
  tests.test_task_builder -v
```

Expected: all pass; planner-derived counts equal the literal contract.

- [ ] **Step 5: Commit Task 1**

```bash
git add tasks/official RELEASE_MANIFEST.json scripts/release_audit.py Makefile tests
git commit -m "feat: publish six-scene Task v1"
```

---

### Task 2: Synchronize public Benchmark and Baseline documentation

**Files:**
- Modify: `tests/test_release_documentation.py`
- Modify: `README.md`
- Modify: `baselines/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/GETTING_STARTED.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/BENCHMARK_PROTOCOL.md`
- Modify: `docs/EVALUATION.md`
- Modify: `docs/CUSTOM_BASELINE_QUICKSTART.md`
- Modify: `docs/SUBMISSION_QUICKSTART.md`
- Modify: `docs/BASELINE_INTEGRATION.md`
- Modify: `docs/DATA_ADAPTER.md`

**Interfaces:**
- Consumes: Task identities/counts from Task 1 and the current Dataset asset schema.
- Produces: one discoverable public narrative that states six scored scenes and the exact V2V limitation.

- [ ] **Step 1: Add failing documentation-contract assertions**

Extend `tests/test_release_documentation.py` so real public manuals must:

- contain “Six scored scenes” and “One Dataset-only unsupported scene”;
- link from README to `BASELINE_INTEGRATION.md`, `DATA_ADAPTER.md`, and `RUN_LAYOUT.md` with result-reading language;
- state that official managed V2V is unsupported because Dataset Cases lack cropped conditioning-prefix videos;
- state that reference/source/evaluator videos cannot substitute for V2V conditioning;
- contain none of `WAN`, `Cosmos`, or `free_fall` in `DATA_ADAPTER.md`.

- [ ] **Step 2: Run documentation tests and confirm RED**

Run:

```bash
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH \
PYTHONPATH=src:tests:. python -m unittest tests.test_release_documentation -v
```

Expected: failures on the old five-scene headings, missing navigation/V2V statements, and stale DataAdapter model sections.

- [ ] **Step 3: Update Task/protocol wording and command paths**

Replace active public references to the old Task filenames/IDs and five-scene counts. Document six scored scenes, spring membership, `775` direct jobs, `679` training Cases, and `96` finetune evaluation jobs. Keep `push_bottle` as the sole Dataset-only unsupported scene and do not describe it as scored.

- [ ] **Step 4: Publish the V2V limitation and navigation**

Add a support-status block near the start of `CUSTOM_BASELINE_QUICKSTART.md` and a matching detailed section in `BASELINE_INTEGRATION.md`. State that `baseline init --backend managed-v2v` is for reserved interface development only on this Dataset release. Link the quickstart to both detailed contracts. Add README navigation entries for Baseline Integration, DataAdapter, and Run result interpretation.

- [ ] **Step 5: Remove model-specific DataAdapter residue**

Delete the “当前 WAN 与 Cosmos 配置” material and all `free_fall` references. Preserve and promote the generic “统一 I2V 媒体契约” section as a model-independent top-level section.

- [ ] **Step 6: Run documentation tests and verify GREEN**

Run the Step 2 command and scan active public manuals (excluding historical `docs/superpowers`) for old Task filenames, five-scene wording, WAN/Cosmos, and `free_fall`.

- [ ] **Step 7: Commit Task 2**

```bash
git add README.md baselines/README.md docs tests/test_release_documentation.py
git commit -m "docs: explain six-scene baseline workflow"
```

---

### Task 3: Make Run results self-explanatory and verify the release

**Files:**
- Modify: `tests/test_release_documentation.py`
- Modify: `docs/RUN_LAYOUT.md`
- Modify: `run/README.md`

**Interfaces:**
- Consumes: existing AtomicRun output filenames and Evaluation v1 status/aggregation semantics.
- Produces: a canonical user-facing result-reading guide reachable from README.

- [ ] **Step 1: Add failing result-guide assertions**

Require `RUN_LAYOUT.md` to name `evaluation/case_results.jsonl`, `evaluation/task_result.json`, `coverage`, `status_counts`, expert score, CSTI, degradation reasons, and the rule that partial `observed_mean_score` is not an official result.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH \
PYTHONPATH=src:tests:. python -m unittest \
  tests.test_release_documentation.ReleaseDocumentationTests.test_run_layout_explains_official_result_fields -v
```

Expected: failure because `RUN_LAYOUT.md` currently names only directories.

- [ ] **Step 3: Consolidate Run documentation**

Expand `RUN_LAYOUT.md` with the canonical output tree and a concise “read results” sequence. Explain status meanings and official-score eligibility without duplicating evaluator internals. Reduce `run/README.md` to the directory-local rules plus a link to the canonical guide.

- [ ] **Step 4: Run focused and full release verification**

Run:

```bash
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH \
PYTEST_ADDOPTS='-p no:cacheprovider' make test-interface

PATH=/root/Steven/.venvs/wan22-pair-text/bin:$PATH \
PYTEST_ADDOPTS='-p no:cacheprovider' make test-evaluation

PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH make release-check
PATH=/root/Steven/.venvs/vphysbench-ablation/bin:$PATH make release-archive-check

/root/Steven/.venvs/vphysbench-ablation/bin/python - <<'PY'
import json
from pathlib import Path
for path in [*Path('tasks/official').glob('*.json'), Path('RELEASE_MANIFEST.json')]:
    json.loads(path.read_text(encoding='utf-8'))
print('json=ok')
PY

git diff --check
git status --short
```

Expected: interface and evaluation suites pass; release audits report `ok`; JSON parsing and diff checks exit zero; only intentional commits are present.

- [ ] **Step 5: Commit Task 3**

```bash
git add docs/RUN_LAYOUT.md run/README.md tests/test_release_documentation.py
git commit -m "docs: explain benchmark result interpretation"
```
