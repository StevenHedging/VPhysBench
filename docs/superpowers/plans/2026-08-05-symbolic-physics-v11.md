# Symbolic Physics and Directional Prompt Dataset 11.0.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish Dataset 11.0.0 with non-negative symbolic structured physics, deterministic English directional prompts, corrected independent/audit-only roles, and unchanged media and View membership.

**Architecture:** Dataset 10.0.0 remains immutable. A V11 builder performs a fixed metadata-only migration, writes one `physics.v11.json` per active Case, stages the same minimal seven-entry release shape, and records independent migration/validation evidence. Loader schema dispatch preserves V3/V4 behavior while enforcing the V5 four-field quantity and V2 physics-document contracts; Baselines consume only corrected independent quantities and evaluators retain complete audit physics.

**Tech Stack:** Python 3.10, `unittest`, JSON Schema 2020-12, JSON/JSONL, SHA-256 locks, POSIX atomic rename, Git.

## Global Constraints

- Use `datasets/releases/10.0.0` as the sole base; do not modify any historical Release or existing `physics.json`.
- Preserve all 799 Case IDs, Scene IDs, ordered Case membership, Views, media roles, media bytes, masks, appearance, temporal, alignment, and source provenance facts.
- Add exactly 799 tracked `physics.v11.json` files; never stage video, image, mask, archive, workbook, or source media.
- Use the exact symbol, flag, prompt, direction, and count contracts in `docs/superpowers/specs/2026-08-05-symbolic-physics-v11-design.md`.
- Keep stable parameter keys. Convert only the 494 documented negative collision scalar occurrences to absolute magnitudes.
- V11 quantity values must be finite and non-negative; prompts contain symbols but no numeric physical values or acquisition/appearance hints.
- V11 runtime Release entries are exactly `README.md`, `dataset.json`, `release.json`, `cases.jsonl`, `assets.lock.json`, `scenes/`, and `views/`.
- Store V11 evidence only in `datasets/provenance/releases/11.0.0/`.
- Preserve the official Task selections, Views, Scene lists, seeds, reporting, evaluation protocol, 582 train Cases, 76 finetune jobs, and 658 direct jobs.

---

### Task 1: Add the schema-5 symbolic quantity and physics-document contract

**Files:**
- Create: `schemas/v5/dataset.schema.json`
- Create: `schemas/v5/case.schema.json`
- Modify: `src/physbench/datasets/loader.py`
- Modify: `tests/test_dataset_contract_v4.py`
- Modify: `tests/test_dataset_maintenance_scripts.py`

**Interfaces:**
- Consumes: Dataset/Case `schema_version` and optional `assets.physics_annotation`.
- Produces: schema-dispatched quantity validation and `_validate_case_physics_annotation(case, asset_root)` accepting V10 document schema 1.0 or V11 document schema 2.0 according to the containing Case schema.

- [ ] **Step 1: Write failing schema and Loader tests**

Add tests that construct a schema-5 Case with:

```python
{"value": 0.2, "unit": "kg", "annotated": True, "symbol": "m"}
```

and prove it loads with a matching schema-2 physics document. Add independent subtests rejecting a missing/empty/non-string symbol, an unknown fifth quantity field, a negative value, V11 document schema 1.0, and a schema-4 Case carrying `symbol`. Retain a passing historical schema-4/V10 fixture.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_dataset_contract_v4 tests.test_dataset_maintenance_scripts -v
```

Expected: schema 5.0 is rejected and the Loader still requires three quantity fields/document schema 1.0.

- [ ] **Step 3: Add exact V5 JSON Schemas**

Copy the V4 Dataset and Case structural contracts, set descriptor/Case version to `5.0`, and make the quantity definition require exactly `value`, `unit`, `annotated`, and non-empty `symbol`; add `minimum: 0` to `value`.

- [ ] **Step 4: Implement Loader schema dispatch**

Extend accepted descriptor and Case schemas to `5.0`. Use:

```python
quantity_fields = (
    {"value", "unit", "annotated", "symbol"}
    if schema_version == "5.0"
    else {"value", "unit", "annotated"}
)
```

Require finite `value >= 0`, non-empty `symbol`, and no `ood` for schema 5.0. Require physics-document schema `2.0` for a schema-5 Case and `1.0` for earlier Cases. Preserve all historical paths and behavior.

- [ ] **Step 5: Run tests and verify GREEN**

Run Step 2 plus `tests.test_case_local_physics_v10`. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add schemas/v5/dataset.schema.json schemas/v5/case.schema.json \
  src/physbench/datasets/loader.py tests/test_dataset_contract_v4.py \
  tests/test_dataset_maintenance_scripts.py
git commit -m "feat(dataset): validate symbolic physics schema v5"
```

### Task 2: Implement the pure V11 physics and prompt migration

**Files:**
- Create: `scripts/build_dataset_v11.py`
- Create: `tests/test_symbolic_physics_v11.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: ordered V10 Cases/Scenes and the V11 design's fixed tables.
- Produces: `SYMBOLS`, `AUDIT_ONLY`, `PhysicsWrite`, `migrate_case(case)`, `migrate_scene(scene, migrated_cases)`, `prepare_v11(cases, scenes, datasets_root)`, and `materialize_physics_documents(writes)`.

- [ ] **Step 1: Write failing symbol-table completeness tests**

Load all V10 Cases and assert that the union of `(scene_id, parameter)` equals the exact keys of `SYMBOLS`, that all mappings are non-empty, and that applying the table makes symbols unique within every Case.

- [ ] **Step 2: Write failing signed-value and flag tests**

Assert V10 contains exactly 494 negative occurrences in the two documented collision fields. After `migrate_case`, require zero negatives, unchanged zeros/positives, exactly 715 true-to-false transitions across all Cases, no false-to-true transition, and deep equality for every value/unit not explicitly migrated.

- [ ] **Step 3: Write failing prompt tests for every Scene branch**

Use real V10 Cases to cover two-ball single, two-ball opposed, three-ball single, pendulum with/without mass, circular one/two object, and the other four fixed Scene forms. Assert exact literal prompts from the design, `language="en"`, `annotation_source="symbolic_physics_prompt_v1"`, all annotated symbols appear as tokens, and the forbidden acquisition/appearance vocabulary and numeric-value patterns are absent.

- [ ] **Step 4: Write failing direction contradiction tests**

Mutate real collision fixtures so a single-incident striker sign or opposed direction conflicts with `appearance` and require `ValueError` containing `signed velocity contradicts audited direction`. Prove the builder does not infer direction for circular motion.

- [ ] **Step 5: Write failing document and scene-classification tests**

Require deterministic `physics.v11.json` bytes with document schema 2.0 and one final newline. Require Scene structured/non-conditionable lists to equal the migrated true/false parameter universes. Reuse conflict refusal tests to prove no partial writes.

- [ ] **Step 6: Run tests and verify RED**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_symbolic_physics_v11 -v
```

Expected: import failure because the V11 builder does not exist.

- [ ] **Step 7: Implement fixed migration tables and pure functions**

Implement the exact symbol table and audit-only set from the design. Deep-copy each Case, set schema 5.0, assign every symbol, normalize only validated collision negatives with `abs`, correct flags, generate the deterministic Scene prompt, and point `assets.physics_annotation` to the Case-root `physics.v11.json`.

`migrate_scene` deep-copies schema-2 Scene content and replaces only `structured_physics_parameters` and `non_conditionable_physics_parameters` with sorted true/false universes found in migrated Cases.

- [ ] **Step 8: Update ignore rules**

Admit `datasets/assets/*/*/physics.v11.json` while keeping all media ignored. Add `git check-ignore --no-index` assertions for V10/V11 physics files and representative media.

- [ ] **Step 9: Run tests and verify GREEN**

Run Step 6. Expected: all tests pass with no filesystem mutation.

- [ ] **Step 10: Commit**

```bash
git add .gitignore scripts/build_dataset_v11.py tests/test_symbolic_physics_v11.py
git commit -m "feat(dataset): prepare symbolic physics migration"
```

### Task 3: Publish and independently validate Dataset 11.0.0

**Files:**
- Modify: `scripts/build_dataset_v11.py`
- Create: `scripts/validate_dataset_v11.py`
- Modify: `tests/test_symbolic_physics_v11.py`
- Create: `datasets/assets/<scene>/<case>/physics.v11.json` for 799 Cases
- Create: `datasets/releases/11.0.0/{README.md,dataset.json,release.json,cases.jsonl,assets.lock.json,scenes/,views/}`
- Create: `datasets/provenance/releases/11.0.0/{migration.json,validation.json}`

**Interfaces:**
- Consumes: Task 2 candidates and `scripts.build_dataset_asset_lock.rebuild_asset_lock`.
- Produces: `build_release(check: bool=False)` and independent `validate_v11(dataset_path=V11_DATASET, write_report=False)`.

- [ ] **Step 1: Write failing Release invariance tests**

Require the exact seven-entry Release tree, 799 schema-5 Cases, 799 unique V11 physics paths, 6,038 locked assets, unchanged 5,239 V10 non-physics lock records, byte-identical Views, and unchanged Case facts after removing the explicitly allowed fields (`schema_version`, `text`, `physics`, and `assets.physics_annotation`).

- [ ] **Step 2: Write failing evidence/count tests**

Require migration evidence to record 494 value changes, 715 flag changes, 799 prompt changes, zero media changes, the symbol-table digest, unchanged View digests, and old/new Dataset/asset digests.

- [ ] **Step 3: Run Release tests and verify RED**

Run Task 2's test command. Expected: V11 Release files are absent.

- [ ] **Step 4: Implement staged V11 construction**

Preflight all candidates, materialize conflict-free documents atomically, create a temporary sibling under `datasets/releases`, copy Views unchanged, write migrated Scenes/Cases and a schema-5 descriptor, rebuild the asset lock, require 6,038 entries, and perform a complete `check_asset_hashes=True` load before atomic publication.

Write migration evidence after staged validation and before final independent validation. `--check` performs all comparisons without writing. Refuse an existing V11 target unless `--check` is used.

- [ ] **Step 5: Implement the independent validator**

Without calling builder migration helpers, verify the exact symbol table, flags, prompts, values, direction phrases, schemas, document equality, View equality, allowed Case differences, all 6,038 SHA-256 values, minimal tree, evidence counts, and a full Loader hash check. Write a stable validation report only with `--write-report`.

- [ ] **Step 6: Publish V11**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/build_dataset_v11.py
```

Expected: `cases=799 physics=799 locked_assets=6038 negatives_normalized=494 flags_demoted=715 prompts=799`.

- [ ] **Step 7: Run independent validation and tests**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  scripts/validate_dataset_v11.py --write-report
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_symbolic_physics_v11 -v
```

- [ ] **Step 8: Stage only metadata and commit**

Require exactly 799 added `physics.v11.json`, no added file over 10 MiB, and no staged media extension. Then commit:

```bash
git add scripts/build_dataset_v11.py scripts/validate_dataset_v11.py \
  tests/test_symbolic_physics_v11.py datasets/releases/11.0.0 \
  datasets/provenance/releases/11.0.0 datasets/assets/*/*/physics.v11.json
git commit -m "data: publish symbolic physics release v11"
```

### Task 4: Align Baseline and evaluator consumers with V11 semantics

**Files:**
- Modify: `src/physbench/evaluation/common/entities/manifest.py`
- Create: `src/physbench/baseline_plugins/resources/six_scene_physics_clauses_v2.json`
- Create: `baselines/wan22_quantity_embedding/quantity_registry_v2.json`
- Modify: `baselines/causal_forcing_pp_2step_i2v/physics.baseline.json`
- Modify: `baselines/wan22_quantity_embedding/baseline.json`
- Modify: affected tests in `tests/test_entity_manifest.py`, `tests/test_causal_forcing_autoregressive_baseline.py`, `tests/test_wan22_quantity_embedding.py`, and `tests/test_conditioning_contracts.py`

**Interfaces:**
- Consumes: schema-5 quantities with symbol and corrected `annotated` flags.
- Produces: magnitude-aware collision alias validation and Baseline parameter registries containing only V11 independent fields.

- [ ] **Step 1: Write failing compatibility tests**

Prove a V11 collision alias is allowed to be `annotated=false`, compares equal by magnitude to the indexed numbered ball, and retains its symbol. Prove historical signed schema-4 fixtures still use exact signed equality. Assert standard physics text and quantity embedding select exactly the V11 annotated field set for every current Case.

- [ ] **Step 2: Run affected tests and verify RED**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python -m unittest \
  tests.test_entity_manifest tests.test_causal_forcing_autoregressive_baseline \
  tests.test_wan22_quantity_embedding tests.test_conditioning_contracts -v
```

- [ ] **Step 3: Implement schema-aware alias comparison**

For Case schema 5.0 compare `abs(alias.value)` and `abs(numbered.value)` plus unit, allow the alias's false annotation, and keep the numbered quantity annotated. Preserve the existing exact signed/flag comparison for schemas 3.0/4.0.

- [ ] **Step 4: Add versioned Baseline resource registries**

Create `six_scene_physics_clauses_v2.json` and
`quantity_registry_v2.json`; do not redefine the semantics of either existing
v1 resource in place. Give both new resources explicit v2 identities, remove
audit-only parameters from their numeric-injection selections, add missing
independent parameters such as optional pendulum mass and push-bottle force
summaries where the resource supports the Scene, and preserve
values/units/symbols in adaptation audit payloads. Point the two active
Baseline manifests at the new resource names/identities so their bundle
digests change transparently. Do not change Baseline IDs or Task families
solely for this Dataset migration.

- [ ] **Step 5: Run affected tests and verify GREEN**

Run Step 2. Expected: all current V11 assertions and historical fixtures pass.

- [ ] **Step 6: Commit**

```bash
git add src/physbench/evaluation/common/entities/manifest.py \
  src/physbench/baseline_plugins/resources/six_scene_physics_clauses_v2.json \
  baselines/causal_forcing_pp_2step_i2v/physics.baseline.json \
  baselines/wan22_quantity_embedding/quantity_registry_v2.json \
  baselines/wan22_quantity_embedding/baseline.json tests
git commit -m "feat(dataset): align consumers with symbolic physics"
```

### Task 5: Make V11 the official default and synchronize documentation

**Files:**
- Modify: `src/physbench/data_layout.py`
- Modify: `tasks/official/five_scene_direct_eval.json`
- Modify: `tasks/official/five_scene_finetune_eval.json`
- Modify: current-default Dataset/Task/Baseline tests
- Modify: `README.md`, `Makefile`, `datasets/{README.md,assets/README.md,releases/README.md,DATASET_OVERVIEW.md}`, current `docs/*.md`, and active Baseline READMEs

**Interfaces:**
- Consumes: validated V11 Dataset and aligned consumers.
- Produces: `V11_RELEASE_ROOT`, `V11_DATASET`, `LATEST_DATASET=V11_DATASET`, official `_v11` Tasks, and current documentation.

- [ ] **Step 1: Write failing default and Task tests**

Require V11 identity/release/schema/counts, official Task IDs `five_scene_direct_eval_v11` and `five_scene_finetune_eval_v11`, unchanged 582/76/658 plan counts, and historical V10 still loadable.

- [ ] **Step 2: Run tests and verify RED**

Run current Dataset, Task builder, flat layout, maintenance, and Causal Forcing suites. Expected: current default remains V10.

- [ ] **Step 3: Switch constants and Task identities**

Add V11 constants, switch `LATEST_DATASET`, and change only official Task/Dataset identities. Update tests that compile the current official Task to use V11; retain explicit historical V8/V10 coverage.

- [ ] **Step 4: Synchronize documentation**

Document schema 5.0, `symbol`, non-negative magnitude semantics, English symbolic prompts, independent/audit-only roles, `physics.v11.json`, V10 immutability, V11 slim Release/evidence boundary, and future-ingestion requirements. Do not rewrite frozen historical reports or Releases.

- [ ] **Step 5: Search stale current references**

Search V10 paths/IDs outside frozen V10, provenance, experiments, and superpowers records. Classify remaining occurrences as explicit history or fix them.

- [ ] **Step 6: Run current-default tests and commit**

```bash
git add src/physbench/data_layout.py tasks/official tests README.md Makefile \
  datasets/README.md datasets/assets/README.md datasets/releases/README.md \
  datasets/DATASET_OVERVIEW.md docs baselines
git commit -m "feat(dataset): make symbolic physics v11 official"
```

### Task 6: Final verification and baseline comparison

**Files:**
- Verify only; fix scoped defects if discovered.

**Interfaces:**
- Consumes: final committed V11 repository.
- Produces: fresh validation evidence and completion report.

- [ ] **Step 1: Run both historical and current full-hash validators**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/validate_dataset_v10.py
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python scripts/validate_dataset_v11.py
```

- [ ] **Step 2: Run the targeted release gate**

Run V10/V11, Loader/schema, Dataset/Task, Baseline adaptation, entity manifest, and maintenance suites. Require zero failures.

- [ ] **Step 3: Verify repository scope**

Require 799 tracked `physics.json`, 799 tracked `physics.v11.json`, no changed media since the design commit, exact V11 release/evidence trees, clean `git diff --check`, and successful `py_compile` for changed Python files.

- [ ] **Step 4: Run the full unit suite**

```bash
PYTHONPATH=src:tests:. /mnt/nvme1/NicoCache/envs/physics_wan/bin/python \
  -m unittest discover -s tests -v
```

Compare with the current known baseline of 536 tests, 21 failures, 64 errors,
and 12 skips. New V11 tests increase the total; no new failure/error category
or count is allowed.

- [ ] **Step 5: Final post-commit verification**

Run `git status --short --branch`, V11 validation, and the targeted gate again.
Report commit IDs, V11 Dataset and asset digests, symbol-table digest, 494/715
migration counts, 6,038 locked assets, evidence paths, and unchanged V10
validation status.
