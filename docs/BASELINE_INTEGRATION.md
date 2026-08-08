# Baseline integration contract

This document is the detailed reference for user-created baseline bundles.
Start with [CUSTOM_BASELINE_QUICKSTART.md](CUSTOM_BASELINE_QUICKSTART.md).

## Discovery and identity

The registry discovers `baseline.json` descriptors under the requested
baseline root. A clean checkout has none. Each descriptor declares:

- `baseline_id` and `baseline_version`;
- implementation kind and fingerprinted files;
- supported Task families, scenes and generation modes;
- model identity without embedding weights;
- input policy and DataAdapter recipe;
- runner/trainer recipes when applicable.

`baseline.local.json` is merged as deployment-only configuration. Both the
portable bundle and the effective deployment receive independent digests.

## Implementation kinds

### Managed I2V

Use `baseline init <id> --backend managed-i2v`. The standard driver authorizes
the canonical prompt and first frame, materializes the conditioning canvas,
writes a sealed job spec, executes the configured command and validates its
video.

### Managed V2V

Use `--backend managed-v2v`. The standard command receives the authorized input
video instead of a first-frame image. The same run-owned output and identity
rules apply.

### Submission

Use `--backend submission`. The plugin imports a complete JSONL mapping from
canonical jobs to existing videos. It verifies identity and coverage, copies
videos into the run, and records source/destination hashes.

## Input policy

The baseline descriptor, not the Task, declares text and physics use. Physics
may be `ignored`, `optional` or `required`; declared representations must be
handled by the adapter. A managed command never receives reference videos,
ground-truth trajectories or evaluator annotations.

## Standard I2V process boundary

```text
--prompt TEXT
--image PATH
--output PATH
--seed INTEGER
--job-spec PATH
```

The command must return zero and write a decodable video to `--output`.
`--job-spec` is authoritative for width, height, FPS, start time and frame
count. Extra flags belong in `runner.config.extra_args`; secrets do not.

## Validation order

1. Registry validates the descriptor and local override shape.
2. Dependency paths are resolved inside the bundle and fingerprinted.
3. Dataset and Task compile into a sealed BaselineTaskInstance.
4. The driver prepares only contract-authorized inputs.
5. Execution writes under `run/<run_id>/predictions/`.
6. Media validation runs before evaluator dispatch.
7. Prediction SHA-256 and provenance are sealed.

Failures retain logs and explicit statuses. They must not be converted into a
successful prediction or silently assigned an official score.
