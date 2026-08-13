# Six-scene V15 Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the 2026-08-12 six-scene task and vertical-spring evaluator as the backward-compatible latest V15 benchmark on main.

**Architecture:** Preserve V1-V14 and all baseline code, add the release evaluator as a new scene plugin, and compose V15 from unchanged V14 scene definitions plus the final vertical-spring config. Keep schema-4 task planning and adapt the release tasks to View A/View B rather than importing the release branch's schema reset.

**Tech Stack:** Python 3.11+, JSON Schema, NumPy, OpenCV, SciPy, unittest.

## Global Constraints

- Do not delete or rewrite existing baselines, historical protocols, tasks, or provenance.
- Existing `scene_default_v14` and frozen baseline experiment tasks remain unchanged.
- Register `vertical_spring_oscillator_v1` in addition to every existing evaluator type.
- New official tasks use main's schema version 4.0 and Dataset 13.0.0 views.
- Follow red-green-refactor for new observable behavior.

---

### Task 1: V15 protocol and task contract

**Files:**
- Create: `tests/test_evaluation_protocol_v15.py`
- Create: `tests/test_task_protocol_v15.py`
- Create: `configs/evaluation/protocols/scene_default_v15.json`
- Create: `tasks/official/six_scene_train_six_scene_eval_v15.json`
- Create: `tasks/official/six_scene_direct_eval_v15.json`
- Modify: `schemas/v3/evaluation_protocol.schema.json`

**Interfaces:**
- Consumes: `load_evaluation_protocol(protocol_id)` and `plan_atomic_task(task, dataset)`.
- Produces: protocol ID `scene_default_v15` and two schema-4 task files with literal counts 679/96/775.

- [ ] Write tests asserting V14 remains unchanged, V15 has exactly six scenes, vertical spring resolves, and task planning yields the frozen counts.
- [ ] Run the new tests and verify failures are caused by missing V15 artifacts/evaluator.
- [ ] Add V15 JSON/tasks and the schema definition for the vertical-spring scene.
- [ ] Re-run protocol-only tests; retain expected evaluator-registration failure for Task 2.

### Task 2: Vertical-spring evaluator

**Files:**
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/__init__.py`
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/observation.py`
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/scoring.py`
- Create: `src/physbench/evaluation/scenes/vertical_spring_oscillator/evaluator.py`
- Create: `tests/test_vertical_spring_scoring.py`
- Create: `tests/test_vertical_spring_evaluator.py`
- Modify: `src/physbench/evaluation/registry.py`

**Interfaces:**
- Consumes: robust subject evaluation base, frozen subject records, case manifest, SAM2 observation runtime.
- Produces: lazy-resolved `VerticalSpringOscillatorCaseEvaluator` for type `vertical_spring_oscillator_v1`.

- [ ] Port the release tests and verify they fail because the scene package/type is absent.
- [ ] Port the final scene package and add the evaluator type without removing legacy registrations.
- [ ] Run scoring and evaluator tests to green.

### Task 3: Shared manifest and robust failure contract

**Files:**
- Modify: `src/physbench/evaluation/common/entities/manifest.py`
- Modify: `src/physbench/evaluation/common/frozen_subject.py`
- Modify: `src/physbench/evaluation/common/base.py`
- Modify: `tests/test_entity_manifest.py`
- Modify: `tests/test_frozen_subject_anchor.py`
- Add or modify focused failure-contract tests imported from the release branch.

**Interfaces:**
- Consumes: Dataset 13 vertical-spring case physics and robust-subject evaluator hooks.
- Produces: deterministic spring mass/apparatus entities and origin-aware evaluated-zero/unavailable failures.

- [ ] Port focused release assertions and verify the missing behavior fails.
- [ ] Apply only additive/hardening changes required by those assertions.
- [ ] Run entity, frozen-subject, common-base, and all scene evaluator regressions.

### Task 4: Publish latest official benchmark

**Files:**
- Modify: current official task aliases where they intentionally track latest.
- Modify: `README.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/EVALUATION.md`
- Modify: focused current-dataset/task tests.

**Interfaces:**
- Consumes: V15 protocol/tasks from Tasks 1-3.
- Produces: latest documentation and canonical current task entry points selecting V15, while frozen baseline experiment tasks retain V14.

- [ ] Write/adjust tests for current official task IDs, scene selection, protocol ID, and counts.
- [ ] Verify old expectations fail for the current aliases.
- [ ] Switch current aliases and documentation to V15 without editing baseline implementations.
- [ ] Run task/current-dataset/integrated-baseline tests.

### Task 5: Full verification and commit

**Files:**
- Review all changed files; no new production scope.

**Interfaces:**
- Consumes: completed V15 implementation.
- Produces: a clean, reviewable main-worktree commit.

- [ ] Run JSON/schema checks and `git diff --check`.
- [ ] Run all relevant unit and integration suites with the complete evaluation environment.
- [ ] Confirm no files under `baselines/` changed and V14 hashes are unchanged.
- [ ] Commit only the reviewed benchmark sync.

