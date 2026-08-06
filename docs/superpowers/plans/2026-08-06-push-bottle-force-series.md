# Push-Bottle Force-Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace push-bottle force summaries with complete Case-local force time series attached to `object_1`.

**Architecture:** Extend the current physics leaf contract with one explicit time-series leaf shape, while retaining the existing scalar shape. Migrate 141 Cases from the already normalized XLSX records, then make scalar-only Baseline consumers skip series leaves without inventing summaries.

**Tech Stack:** Python 3.10+, JSON/JSONL, XLSX-derived normalized provenance, `unittest`.

## Global Constraints

- Preserve source sample order and explicit timestamps; do not interpolate, smooth, deduplicate, or reconstruct time.
- Convert signed source force to formal magnitude with `max(force_n, 0.0)`; retain signed values only in existing provenance.
- Remove formal mean and peak force quantities.
- Do not create a sidecar signal file or a second physics annotation file.
- Do not calculate hashes or modify media bytes.
- Preserve stable Case IDs and View membership.

---

### Task 1: Lock the time-series quantity contract

**Files:**
- Modify: `tests/test_dataset_contract_v5.py`
- Modify: `tests/test_single_current_physics_v12.py`
- Modify: `schemas/v5/case.schema.json`
- Modify: `src/physbench/datasets/physics.py`
- Modify: `src/physbench/datasets/loader.py`

**Interfaces:**
- Produces: scalar and time-series leaf validation through `iter_physics_quantities(case)`.

- [ ] Add failing tests for a valid `applied_force` series and malformed samples, fields, units, negative values, and empty series.
- [ ] Run the two Dataset modules and confirm failure under the scalar-only contract.
- [ ] Add exact time-series schema and Loader validation while keeping scalar validation unchanged.
- [ ] Add `push_bottle` to the grouped Scene contract and stable projection mapping.
- [ ] Run the two Dataset modules and confirm the contract passes.

### Task 2: Migrate the 141 Case annotations and names

**Files:**
- Modify: `datasets/assets/push_bottle/*/physics.json`
- Modify: `datasets/assets/push_bottle/*/caption.json`
- Modify: `datasets/releases/12.0.0/cases.jsonl`
- Modify: `datasets/releases/12.0.0/scenes/push_bottle.json`
- Modify: push-bottle mask manifests and relevant provenance reports

**Interfaces:**
- Consumes: `normalized_annotations.json.records[].force_annotation.force_samples`.
- Produces: 141 grouped annotations with `objects.object_1.applied_force`.

- [ ] Preflight exact Case-to-workbook/sheet matching, sample counts, field sets, and unique target directory names.
- [ ] Write all transformed JSON in one mechanical migration only after every preflight check passes.
- [ ] Rename push-bottle Case directories to remove `fmax` and `fmean`, then update all path references without changing Case IDs.
- [ ] Update mask object IDs and physics paths without touching PNG or NPZ files.
- [ ] Verify all 141 Cases against their source records and run Dataset tests.

### Task 3: Align scalar-only consumers

**Files:**
- Modify: `src/physbench/baseline_runtime/adapter.py`
- Modify: `baselines/wan22_quantity_embedding/adapter.py`
- Modify: active structured-text and quantity registries
- Modify: related Baseline tests

**Interfaces:**
- Consumes: projected scalar and time-series quantities.
- Produces: current scalar-only adapters that use mass/height and explicitly omit `applied_force`.

- [ ] Add failing tests proving scalar adapters do not summarize or serialize the series.
- [ ] Make scalar adapters require a scalar `value` before rendering a quantity.
- [ ] Remove mean/peak registry clauses and retain mass/height clauses.
- [ ] Run Task, Baseline, symbolic-consumer, and quantity-embedding tests.

### Task 4: Synchronize validation and documentation

**Files:**
- Modify: `scripts/validate_dataset_v12.py`
- Modify: current Dataset, ingestion, architecture, and Baseline documentation
- Modify: `datasets/provenance/releases/12.0.0/validation.json`

**Interfaces:**
- Produces: a source-grounded, hash-free V12 validation report.

- [ ] Validate the 141 series against normalized provenance, including sample order, time, clamped magnitude, and sample count.
- [ ] Update total leaf and scalar/series coverage counts.
- [ ] Document the series shape, negative-drift rule, caption, naming, and scalar-Baseline behavior.
- [ ] Run the V12 validator, all related unit tests, JSON parsing, Python compilation, and `git diff --check`.
- [ ] Confirm version-control status contains no media or mask binary changes, then commit.
