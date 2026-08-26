# AtomicRun layout and result interpretation

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
│   └── prediction_artifacts.json
├── evaluation/
│   ├── manifest.json
│   ├── case_results.jsonl
│   ├── task_result.json
│   └── cases/<job_id>/
└── reevaluations/
```

## Identity and prediction boundary

- `frozen/` records the Dataset, Task, Baseline, protocol and component
  identities used by the run.
- `task_instance/` contains the sealed Cases, canonical plan and inference
  jobs; `jobs/` contains model-facing job specifications.
- `adaptations/` audits Baseline-owned input transformations; `training/`
  records training or finetuning when the Task requires it.
- `run/<run_id>/predictions/` contains the run-owned generated or imported
  videos. `predictions.jsonl` is the only Baseline-to-evaluator output
  boundary, while `artifacts/prediction_artifacts.json` seals video hashes.
- `logs/` retains subprocess and failure logs. `reevaluations/` may hold
  additional evaluation runs without overwriting the canonical one.

## Read the evaluation result

Start with `evaluation/task_result.json`, then use
`evaluation/case_results.jsonl` to investigate individual jobs.

The Task result reports:

- `coverage`: evaluated jobs divided by canonical official jobs;
- `status_counts`: counts of `evaluated`, `unavailable`, `protocol_error` and
  `error` outcomes;
- `score`: the official six-scene macro expert score, present only when every
  required scene has complete coverage;
- `dimensions.expert`: the same expert score identity, and
  `dimensions.csti`: the independent CSTI aggregation;
- `by_scene` and `breakdown`: scene- and partition-level coverage and scores;
- `robustness.degradation_reason_counts`: why prediction-side failures were
  converted to degraded evaluated-zero results;
- `observed_mean_score`: a diagnostic over results that happened to be
  observed. `observed_mean_score` is not official when coverage is incomplete.

Official publication requires `coverage` equal to `1.0`, a non-null `score`,
and matching frozen identities. A partial run, a successful orchestration, or
a non-null diagnostic mean is not a completed benchmark result.

Each row in `evaluation/case_results.jsonl` binds a canonical `job_id` to its
scene, partition, `status`, expert `score`, `reason_code`, quality diagnostics,
metrics (including per-case CSTI where applicable), and provenance. Interpret
the status before the numeric value:

For current-v1 CSTI results, each scored object also records a
`spatial_tolerance` audit with the reference Tube's median area-equivalent
diameter, the resulting half-diameter radius in native-analysis pixels, and
the number of non-empty postcondition reference masks used. This tolerance is
derived only from the reference Tube after excluding the condition frame at
sample index zero; that is also the only initial sample excluded from the
score.

Prediction-side CSTI identity is initialized by SAM 3.1 text tracking and a
single frame-zero Hungarian match against the frozen GT masks. A successful
case record includes `csti_video`, `csti_per_subject`, `subject_count`,
`initial_matching`, `initial_matching_iou`, and
`termination_frame_per_subject`, plus locked-track diagnostics. A failed
initial match remains visible with `status=evaluator_init_failure`, a null
score, `evaluator_init_success=false`, and a structured
`evaluator_init_failure_reason`; it is never converted to zero or silently
removed.

At Task level, `dimensions.csti.score` is the equal-video mean over initialized
cases. The same record separately reports `init_coverage`,
`valid_video_count`, `evaluator_init_failure_video_count`, and grouped failure
reason counts. These fields distinguish model quality from evaluator
initialization coverage.

- `evaluated`: the case contributes a finite score; `quality.degraded=true`
  identifies prediction failures deliberately counted as zero.
- `unavailable`: a required reference asset or reference observation could
  not be authenticated, so the case is not silently scored.
- `protocol_error`: the prediction violates the sealed media or record
  contract.
- `error`: Dataset identity, evaluator dependency, configuration, or internal
  evaluation failed and must be fixed before publishing results.

`evaluation/manifest.json` binds the protocol fingerprint, planned jobs,
prediction-record count and evaluator registry to these two result files.
Per-job audit and visualizations, when enabled, live under
`evaluation/cases/<job_id>/`.

## Storage boundary

The Dataset is read-only during a run. Generated videos, resized inputs,
features, caches and visualizations must never be written back to `datasets/`.
Run directories are ignored by Git; archive selected runs in external storage
when they must be retained.
