# Benchmark `run/` Path Migration Design

## Objective

Make the active Physics Video Benchmark internally consistent after the
filesystem changes that permanently removed the legacy `runs/` directory and
renamed `runs_v2/` to `run/`.

After the migration, `run/` is the only supported top-level run-output root.
New AtomicRun and matrix commands use it by default, tracked documentation and
tests describe it consistently, and Git ignores generated run artifacts while
retaining `run/README.md`.

## Current State and Failure Evidence

The filesystem already has the intended top-level layout:

```text
physics_video_benchmark/
└── run/
```

The repository metadata and active interfaces have not caught up:

- `src/physbench/cli.py` still defaults `atomic-run` and `matrix-run` to
  `runs_v2`.
- Twenty tracked files contain 62 `runs_v2` occurrences.
- Three tracked files contain five legacy `runs/` occurrences.
- `.gitignore` ignores `runs/` and `runs_v2/`, but not `run/`, so Git currently
  sees the 19 GB `run/` tree as untracked.
- Git records `runs_v2/README.md` as deleted and does not yet recognize
  `run/README.md` as its replacement.
- The ignored `baselines/wan22_lora/baseline.local.json` refers to a checkpoint
  that was permanently deleted with `runs/`. This makes two locally enabled
  integrated-baseline tests fail during deployment validation.

## Scope

### Active runtime and repository paths

Change the canonical run root from `runs_v2` to `run` in:

- CLI defaults for `atomic-run` and `matrix-run`;
- root README and current operational documentation;
- Baseline integration and model-specific usage documentation;
- architecture, task, evaluation, and visualization examples;
- tracked experiment reports whose paths refer to artifacts now located below
  `run/`;
- tests and test fixtures;
- `.gitignore`;
- `run/README.md`.

Remove active descriptions of the deleted `runs/` directory. Historical design
or migration prose may describe legacy concepts only when it is explicitly
marked as historical and cannot be mistaken for a live path. The intended end
state is no actionable tracked command or path that uses `runs/` or `runs_v2/`.

### Stale local deployment

Remove the ignored local file
`baselines/wan22_lora/baseline.local.json`. Its declared checkpoint no longer
exists, so keeping it would falsely claim that the deployment is configured.
Preserve `baseline.local.example.json` and the portable Baseline manifests.
Users can create a new local override when a valid checkpoint is available.

Tests that require real local model deployments must skip when the required
local overrides are absent. Portable schema, registry, adapter, planner, and
artifact tests remain mandatory and must not require the deleted checkpoint.

## Non-goals

- Do not create a compatibility symlink named `runs_v2` or `runs`.
- Do not restore the deleted legacy `runs/` contents.
- Do not rewrite frozen JSON, predictions, evaluation outputs, fingerprints,
  hashes, or reports inside generated run directories.
- Do not make old third-party Baseline weights portable.
- Do not solve public Dataset media hosting, licensing, CI, packaging, or other
  GitHub-release blockers in this migration.
- Do not push to GitHub or alter the remote repository.

## Design

### Canonical path constant at the interface boundary

The user-visible canonical spelling is `run`. Both CLI parser defaults use this
same literal. Internal orchestration continues accepting an explicit
`--output-root`, so callers can intentionally choose another location without
adding a compatibility mode for the deleted names.

The existing orchestration data flow remains unchanged:

```text
CLI default or explicit --output-root
    -> atomic/matrix orchestration
    -> run-local jobs, predictions, evaluation, and visualization artifacts
```

Only the default root changes; AtomicRun identity, schemas, artifact layout
below a run directory, and evaluation behavior do not.

### Git tracking policy

The output-root rules become:

```gitignore
run/*
!run/README.md
```

The obsolete `runs/*`, `!runs/README.md`, `runs_v2/*`, and
`!runs_v2/README.md` rules are removed. `run/README.md` stays tracked as the
directory contract, while all generated matrices and AtomicRun directories
remain ignored.

The existing physical README is recorded as the Git rename
`runs_v2/README.md -> run/README.md`; its text is updated to name `run/` as the
only active root.

### Documentation migration

All actionable examples and live artifact layouts use:

```text
run/<run_id>/
run/<matrix_id>.matrix.json
run/<matrix_id>__<baseline_id>/
```

References in experiment reports are updated because their corresponding
artifacts physically moved under `run/`. This changes path labels only; it does
not mutate the artifacts or their sealed content.

The historical Superpowers flatten-layout plan is updated only enough to avoid
advertising deleted top-level directories as current protected locations. Its
core rule remains: frozen generated artifacts are not rewritten.

### Local deployment truthfulness

Deleting the invalid ignored WAN local override restores an honest state:

- the portable WAN Bundle remains discoverable;
- the example documents the fields required to configure it;
- deployment validation no longer relies on a path to deleted content;
- tests guarded by the presence of all real local deployments skip rather than
  fail for an absent optional deployment.

No placeholder checkpoint path is substituted, because that would preserve a
misconfigured deployment under a different spelling.

## Error Handling and Compatibility

- Commands that omit `--output-root` write to `run/`.
- Commands with an explicit output root retain current behavior.
- No automatic fallback probes `runs_v2/` or `runs/`; silently selecting stale
  locations would reintroduce ambiguity.
- Existing frozen artifacts that internally record old absolute paths remain
  historical records. They may require legacy-path interpretation for forensic
  use, but they are outside the active runtime contract.
- A WAN deployment without a new local override is explicitly unconfigured;
  `baseline list` can discover it, while deployment execution requires valid
  local paths.

## Test Strategy

Implementation follows test-driven development.

1. Add or update a parser regression test proving that both `atomic-run` and
   `matrix-run` default `output_root` to `run`.
2. Update artifact-layout tests to construct active output paths under `run/`.
3. Add a repository-layout assertion that `run/README.md` is the retained
   contract and generated children are ignored.
4. Run a tracked-text gate that rejects actionable `runs_v2` and legacy
   `runs/` references.
5. Verify `git status` does not enumerate generated `run/` contents and records
   the README move without staging the 19 GB artifact tree.
6. Run the Dataset 12.0.0 validator to prove the path migration did not alter
   Dataset content.
7. Run targeted CLI, artifact, integrated-Baseline, orchestration, evaluation,
   and visualization tests.
8. Run the repository's complete supported test command.

The migration is complete only when:

- both CLI defaults equal `run`;
- the tracked path scan has no unintended `runs_v2` or `runs/` references;
- `run/` is ignored except for its tracked README;
- no frozen generated artifact was modified;
- the stale WAN local override is absent;
- all mandatory tests pass, with optional real-deployment tests skipped when
  their local deployment is not configured.

## GitHub Publication Assessment After Migration

The implementation handoff will report publication readiness separately from
run-path consistency. This migration alone cannot make the repository ready for
normal community use. The current known publication blockers remain:

- no repository license;
- 2,395 of 3,993 runtime Dataset assets are not Git-tracked, including all 799
  first frames, all 799 reference videos, and 797 mask manifests;
- no public media download/materialization workflow;
- no CI configuration or reproducible dependency lock;
- community-facing installation instructions still contain machine-local
  absolute paths and an external SAM2 checkout.

These findings do not block the `run/` migration. They must be handled in a
separate public-release design before claiming that a GitHub clone is usable.
