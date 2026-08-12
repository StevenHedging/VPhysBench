# Six-Scene Evaluation and Documentation Design

## Goal

Publish one internally consistent Task v1 surface with six training scenes and
six scored evaluation scenes, while keeping `push_bottle` data intact but
unselected. Update the public manuals so users can discover the Baseline
integration contract, DataAdapter contract, and result interpretation without
reading source code. State precisely that official V2V evaluation is not yet
supported because Dataset Cases do not contain cropped conditioning-prefix
videos.

## Task v1 identity

The release continues to expose exactly two official Task v1 files. The files
and `task_id` values are renamed because retaining `five_scene` in a six-scene
contract would be permanently misleading:

- `six_scene_direct_eval_v1`
  - family: `direct_eval`
  - training cases: `0`
  - evaluation scenes: pendulum, collision, inclined-plane slide, uniform
    circular motion, parabolic motion, and vertical spring oscillator
  - evaluation jobs: all `775` Cases from those six scenes
- `six_scene_train_six_scene_eval_v1`
  - family: `finetune_eval`
  - training scenes: the same six scenes
  - training cases: `679` View A train Cases
  - evaluation scenes: the same six scenes
  - evaluation jobs: `96` View A ID-test Cases

Both Tasks retain inference seed `42` and Evaluation protocol
`scene_default_v1`. `push_bottle` remains present in Dataset 13.0.0 with all of
its Cases and assets, but is selected by neither Task for training or
evaluation.

The release manifest changes atomically with the Task files: official paths,
Task IDs, counts, six scored scenes, and the sole remaining preview scene
(`push_bottle`) must agree. No legacy five-scene Task remains in the official
directory.

## Evaluation and aggregation boundary

`vertical_spring_oscillator_v1` is already public and tested, so making spring
the sixth scored scene changes only Task selection and aggregation membership;
it does not create another protocol or evaluator version. The canonical plan,
coverage denominator, per-scene breakdown, overall macro aggregation, and CSTI
aggregation must all consume the six-scene job set produced by the planner.

The public description changes from “five scored plus two preview scenes” to
“six scored plus one unsupported Dataset-only scene.” Public text must not call
vertical spring a preview scene after this change.

## V2V support boundary

The generic managed-V2V adapter/driver code may remain as a reserved extension
surface, but the current official Dataset and Tasks do not support a V2V
benchmark. Dataset Cases currently have no independent cropped conditioning
prefix video such as `assets.input_video`.

Public manuals and navigation must therefore state all of the following:

- official managed-I2V and submission evaluation are supported;
- managed-V2V is not currently runnable on the official Dataset/Tasks;
- a reference video, physics reference, source video, or evaluator asset must
  never be substituted for a conditioning prefix;
- V2V becomes supported only after a future Dataset release publishes an
  explicit, independently authorized cropped prefix asset and a Task version
  selects it.

The `baseline init --backend managed-v2v` scaffolding command remains available
for interface development. Documentation must label it as reserved and must
not present it as a currently executable official benchmark path.

## Public documentation structure

The main README navigation adds direct entries for:

- Baseline integration contract;
- DataAdapter and input-policy contract;
- Run layout and result interpretation.

`CUSTOM_BASELINE_QUICKSTART.md` links forward to the detailed Baseline and
DataAdapter references, leads with the supported managed-I2V path, and places
the V2V limitation next to the V2V scaffold command.

`RUN_LAYOUT.md` becomes the canonical result-reading guide. It names the
canonical files and explains at least:

- `predictions.jsonl` as the Baseline/evaluator boundary;
- `evaluation/case_results.jsonl` as per-job status, expert score, CSTI, and
  degradation diagnostics;
- `evaluation/task_result.json` as coverage, status counts, scene breakdowns,
  aggregate expert/CSTI dimensions, and official-score eligibility;
- `coverage == 1` as a prerequisite for an official Task score;
- `observed_mean_score` from a partial run as diagnostic only.

The duplicate `run/README.md` stays a short directory-local entry point and
links to `docs/RUN_LAYOUT.md` rather than maintaining a competing contract.

`DATA_ADAPTER.md` removes the WAN, Cosmos, and `free_fall` material. Its unified
I2V media-contract section remains model-independent and normative.

## Testing and release checks

Contract tests are changed before production Task files so the old five-scene
state fails visibly. They independently assert:

- exactly two official filenames and Task IDs;
- direct Task scene set and `775` jobs, including `117` spring jobs;
- finetune Task six training scenes, `679` training Cases, six evaluation
  scenes, and `96` ID-test jobs including `20` spring jobs;
- `push_bottle` is absent from both Tasks;
- release manifest paths, counts, six scored scenes, and one preview scene;
- public protocol documentation says six scored scenes;
- README exposes Baseline Integration, DataAdapter, and result-reading links;
- public V2V documentation contains the unsupported reason and the prohibition
  against using reference assets;
- public manuals contain no WAN, Cosmos, or `free_fall` residue.

After the focused RED/GREEN cycle, run the interface suite, evaluation contract
suite, release audit, release archive verification, JSON parsing, and
`git diff --check`. Dataset assets and Hub binding are not modified.

## Non-goals

- No new Dataset assets or conditioning-prefix extraction.
- No official V2V Task.
- No new evaluator or Evaluation protocol version.
- No change to spring scoring, CSTI, or scene configuration.
- No integrated model Baseline, checkpoint, or runtime output.
- No compatibility alias for the misleading five-scene Task filenames.
