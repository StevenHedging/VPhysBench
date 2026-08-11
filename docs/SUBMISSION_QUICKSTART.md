# Submission quickstart

Use a submission bundle when videos have already been generated and the model
does not need to run inside VPhysBench.

## Create the bundle and freeze jobs

```bash
physbench baseline init my_submission --backend submission

physbench task-build \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_v1.json \
  --baseline baselines/my_submission \
  --output submission_task_instance.json
```

Each submission JSONL row must exactly match a canonical inference job:

```json
{
  "job_id": "task__case__seed000042",
  "case_id": "case",
  "seed": 42,
  "video_path": "../external/predictions/case.mp4"
}
```

Coverage must be exact: no missing, duplicate, or extra job IDs. `case_id` and
`seed` must equal the sealed TaskInstance.

Point the local deployment at the JSONL file:

```json
{
  "runtime": {
    "submission_manifest": "../external/submission.jsonl"
  }
}
```

Save it as `baselines/my_submission/baseline.local.json`, then run:

```bash
physbench baseline validate my_submission

physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval_v1.json \
  --baseline baselines/my_submission \
  --run-id imported_submission \
  --output-root run \
  --execute
```

VPhysBench copies every accepted source video into
`run/imported_submission/predictions/`, records SHA-256 provenance, and rejects
external video references during evaluation. Do not modify the manifest after
baseline validation; its digest is part of the deployment identity.
