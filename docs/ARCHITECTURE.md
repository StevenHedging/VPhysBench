# Architecture

VPhysBench separates five owners:

```text
Dataset → model-agnostic Task → baseline-owned adaptation → AtomicRun → Evaluator
```

- Dataset owns prompts, structured physical quantities, input/reference assets
  and views.
- Task owns family, selection, seeds and evaluation protocol.
- Baseline owns model identity, input policy, adaptation and external runner.
- AtomicRun freezes all identities and owns predictions/logs/artifacts.
- Evaluator alone reads reference assets and emits scene-local measurements.

The clean repository contains no baseline bundle. `baseline_api` validates and
discovers user bundles; `baseline_runtime` compiles TaskInstances and implements
generic managed-I2V, managed-V2V and submission flows. Concrete algorithms stay
under user-created `baselines/<baseline_id>/` directories.

The same Task produces the same canonical plan for every compatible baseline.
Only baseline-owned adaptation and execution may differ. This boundary prevents
Task definitions from encoding model-specific conditioning and prevents model
runners from reading evaluator-only references.

Dataset assets live under `datasets/` after immutable Hub download. Runtime
outputs live only under `run/<run_id>/`. See [RUN_LAYOUT.md](RUN_LAYOUT.md).
