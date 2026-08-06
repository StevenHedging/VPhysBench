# Grouped Physics Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make current physics documents contain only formal parameters, grouped by object and environment for five scenes, while deferring push-bottle classification.

**Architecture:** The on-disk grouped document is the sole truth. A small scene-aware helper validates and projects grouped quantities to established flat semantic names for existing evaluator and Baseline consumers. Push bottle is an explicit temporary flat-scene exception, not a generic fallback.

**Tech Stack:** Python 3.10+, JSON/JSONL, `unittest`, existing PhysBench Loader and Baseline runtime.

## Global Constraints

- Delete every quantity whose current `annotated` value is false.
- Delete `annotated` from every retained quantity, including push bottle.
- Remove inclined-plane `kinetic_friction_coefficient` even though it is currently true.
- Do not classify or otherwise redesign push-bottle quantities in this change.
- Do not create a new release or another physics annotation file.
- Do not calculate or add hashes.
- Do not modify video, image, mask, NPZ, XLSX, or archive bytes.

---

### Task 1: Lock the new Dataset contract with failing tests

**Files:**
- Modify: `tests/test_dataset_contract_v5.py`
- Modify: `tests/test_single_current_physics_v12.py`
- Modify: `tests/test_symbolic_consumers.py`

**Interfaces:**
- Consumes: current V12 Loader and physics documents.
- Produces: assertions for grouped physics, three-field quantities, 3,959-leaf coverage, and the push-bottle exception.

- [ ] Add tests asserting grouped scene keys, object IDs, exact scene mappings, no `annotated`, no removed quantities, and flat push-bottle leaves.
- [ ] Add a test asserting the Loader rejects current quantities containing `annotated`.
- [ ] Run the three modules and verify failures are caused by the old flat/four-field contract.

### Task 2: Add the grouped-physics helper and Loader validation

**Files:**
- Create: `src/physbench/datasets/physics.py`
- Modify: `src/physbench/datasets/loader.py`
- Modify: `schemas/v5/case.schema.json`
- Test: `tests/test_dataset_contract_v5.py`

**Interfaces:**
- Produces: `flat_physics_quantities(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]` and `iter_physics_quantities(case)`.
- Mapping: grouped canonical paths project to existing scene semantic names; push-bottle flat names pass through.

- [ ] Implement strict grouped/flat validation and deterministic semantic-name projection.
- [ ] Change V5 quantity validation to exact `{value, unit, symbol}` leaves.
- [ ] Change the V5 schema to grouped physics plus the explicit push-bottle flat alternative.
- [ ] Run `tests.test_dataset_contract_v5` and verify it passes.

### Task 3: Migrate all 799 physics documents and related Case metadata

**Files:**
- Modify: `datasets/assets/*/*/physics.json`
- Modify: `datasets/releases/12.0.0/scenes/*.json`
- Modify: `datasets/assets/*/*/canonical/masks/manifest.json` where present
- Modify: `datasets/provenance/releases/12.0.0/validation.json`

**Interfaces:**
- Consumes: fixed classification table in the design spec.
- Produces: 3,959 retained three-field leaves and canonical mask physics paths.

- [ ] Run an in-memory preflight that rejects unknown scene keys and verifies expected current counts.
- [ ] Rewrite the 799 existing files in place; do not add a second annotation file.
- [ ] Update scene parameter lists to canonical grouped paths and remove non-conditionable parameter lists.
- [ ] Update mask-manifest object IDs and `physics_keys`; leave mask assets untouched.
- [ ] Run the Dataset contract tests and verify all migrated records pass.

### Task 4: Update current runtime and Baseline consumers

**Files:**
- Modify: `src/physbench/baseline_runtime/compiler.py`
- Modify: `src/physbench/baseline_runtime/adapter.py`
- Modify: `src/physbench/baseline_api/input_policy.py`
- Modify: `src/physbench/baseline_runtime/scaffold.py`
- Modify: `schemas/v5/baseline.schema.json`
- Modify: current `baselines/*/*.json`, adapters, and READMEs that select `annotated=true`
- Test: `tests/test_task_builder.py`, `tests/test_conditioning_contracts.py`, `tests/test_baseline_bundle_v5.py`, `tests/test_wan22_quantity_embedding.py`, `tests/test_causal_forcing_autoregressive_baseline.py`

**Interfaces:**
- Consumes: `flat_physics_quantities`.
- Produces: Baseline-visible flat formal-quantity maps with no annotated filtering.

- [ ] Update tests to require `case.physics` and no `annotated` member, then verify the expected failures.
- [ ] Route compiler and adapters through the central projection helper.
- [ ] Change current input-policy literals from `case.physics[annotated=true]` to `case.physics`.
- [ ] Run the five Baseline/runtime test modules and verify they pass.

### Task 5: Update evaluator consumers and captions

**Files:**
- Modify: `src/physbench/evaluation/common/entities/manifest.py`
- Modify: direct scene evaluators that index `case.physics`
- Modify: inclined-plane `caption.json` files
- Test: evaluator/entity test modules affected by the flat projection.

**Interfaces:**
- Consumes: `flat_physics_quantities` and three-field current leaves.
- Produces: unchanged evaluator semantics for retained parameters.

- [ ] Add or update tests proving current entity manifests accept three-field formal quantities.
- [ ] Route current evaluator lookup through the flat projection helper while preserving legacy fixture handling.
- [ ] Remove `mu_k` from inclined-plane captions and keep all retained symbols present.
- [ ] Run entity and affected scene-evaluator tests.

### Task 6: Synchronize documentation and perform full verification

**Files:**
- Modify: `README.md`, `datasets/*.md`, `datasets/assets/README.md`, `datasets/releases/README.md`, `docs/DATASET.md`, `docs/DATASET_INGESTION.md`, `docs/ARCHITECTURE.md`, `docs/BASELINE_INTEGRATION.md`
- Modify: `scripts/validate_dataset_v12.py`

**Interfaces:**
- Produces: one documented current contract and a non-hash V12 validator.

- [ ] Update documentation to describe grouped formal parameters and the temporary push-bottle exception.
- [ ] Update V12 validation for 3,959 leaves, exact classifications, captions, and masks.
- [ ] Run the V12 validator, current Dataset/Task/Baseline/evaluator suites, JSON parsing, Python compilation, and `git diff --check`.
- [ ] Confirm version-control status contains no media, image, or NPZ changes.
- [ ] Commit the completed migration.

