# AtomicRun layout

`run/` is the only runtime-output root. Every execution creates an immutable
`run/<run_id>/` identity boundary:

```text
run/<run_id>/
├── run.json
├── plan.json
├── predictions.jsonl
├── frozen/
├── task_instance/
├── jobs/
├── adaptations/
├── training/
├── predictions/
├── logs/
├── artifacts/
├── evaluation/
└── reevaluations/
```

- `frozen/`: Dataset, Task, baseline and component identities used by the run.
- `task_instance/`: sealed source Cases, canonical plan and inference jobs.
- `jobs/`: model-facing job specifications.
- `adaptations/`: audited baseline-owned input transformations.
- `training/`: training/finetuning records when requested by the Task.
- `predictions/`: run-owned generated or imported videos.
- `run/<run_id>/predictions/`: the complete path of that video directory.
- `predictions.jsonl`: the baseline-to-evaluator output boundary.
- `logs/`: subprocess and failure logs.
- `artifacts/`: prediction hashes and evaluator diagnostic artifacts.
- `evaluation/`: canonical per-case results and Task aggregation.
- `reevaluations/`: coexisting evaluation variants; canonical results are not
  overwritten.

The Dataset is read-only during a run. Generated videos, resized inputs,
features, caches and visualizations must never be written back to `datasets/`.
Run directories are ignored by Git; archive selected runs in external storage
when they must be retained.
