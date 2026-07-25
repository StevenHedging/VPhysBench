# Scene-aware Evaluation

## 1. Boundary

Evaluation is owned by the Benchmark, not by a Baseline. A Baseline ends at a
frozen `predictions.jsonl` record. The Task evaluator consumes:

```text
CanonicalTaskPlan + frozen cases + predictions + evaluation protocol
```

It uses `plan.jobs` as the primary table, so a missing prediction record still
produces one explicit case result.

The four atomic Task variants use the same evaluation path:

```text
finetune_eval × generic
finetune_eval × physics
direct_eval   × generic
direct_eval   × physics
```

Task family and conditioning are report dimensions; they do not change the
scene evaluator.

## 2. Scene dispatch and statuses

`SceneEvaluatorRegistry` resolves a case evaluator by `scene_id`. The current
protocol implements `pendulum_state_v1`. Collision, free fall, inclined-plane
slide, and uniform circular motion return `unsupported`; they are never assigned
a fabricated zero.

Each planned job produces exactly one status:

- `evaluated`: a score in `[0,1]` was produced;
- `unavailable`: prediction or trustworthy reference data are absent;
- `unsupported`: the scene algorithm has not been implemented;
- `error`: decoding, segmentation, tracking, or quality validation failed.

The official Task score is non-null only with complete coverage. Reports also
show an explicitly partial `observed_mean_score`.

## 3. Pendulum protocol

The evaluator uses Jensen's evaluator as an algorithmic reference, not as a
drop-in script.

### Media normalization

- physical interval: `0..5 s`;
- common sampling timeline: 16 Hz, 81 inclusive time points;
- fixed processing canvas: 480 × 832;
- resize policy: preserve aspect ratio and letterbox;
- no GT frame is ever injected into a prediction.

Videos may have different resolution, FPS, and frame count if they cover the
required physical interval. State trajectories are compared at common physical
timestamps. A prediction that does not cover the interval is unavailable with
`insufficient_duration`.

### Subject extraction and primary score

Reference and prediction are independently segmented by SAM2.1. The evaluator
extracts a pivot, bob center, pendulum length, and angle trajectory from each
mask sequence. The primary case score combines:

- angle-trajectory similarity;
- period similarity;
- amplitude similarity;
- predicted pivot and length consistency.

Scoring in pivot-relative, length-normalized coordinates prevents output
resolution from affecting the physical score.

### OOD1

An appearance-only OOD1 case may use its ID parent as a
`parent_physics_reference`. Before doing so, the evaluator requires:

- an existing `parent_case_id`;
- identical structured physics labels;
- the OOD case's physics reference to equal the parent's reference.

That video is a dynamics reference, not a same-appearance visual GT. If no
trustworthy reference exists, the GT-dependent official score is unavailable.

### Jensen-style IoU diagnostic

In addition to the state score, every successfully evaluated pendulum case
writes:

```text
evaluation/cases/<job_id>/
├── result.json
├── per_frame.csv
└── physical_subject_iou_curve.png
```

The PNG follows Jensen's comparison-curve idea and plots reference-vs-generation
physical-subject mask IoU over all 81 physical timestamps. IoU is diagnostic
rather than the official case score because background or support OOD can alter
pixel space without altering pendulum dynamics. Empty/empty masks receive IoU
zero, and insufficient valid masks fail quality validation instead of producing
a false perfect score.

## 4. Outputs and aggregation

Canonical outputs are:

```text
evaluation/
├── manifest.json
├── case_results.jsonl
├── task_result.json
└── cases/<job_id>/...
```

`case_metrics.jsonl` and `summary.json` are compatibility projections.

For `finetune_eval`, ID and OOD1 partition means are macro-averaged within a
scene. For `direct_eval`, View B groups remain diagnostic and all scene cases
are averaged together. The Task score is a macro mean across selected scenes,
so a scene with more cases cannot dominate the result.

An AtomicRun can be evaluated again after predictions are repaired or copied
into place:

```bash
PYTHONPATH=src python3 -m physbench evaluate --run-dir runs_v2/<run_id>
```

The command detects the frozen v2 TaskInstance and uses the scene-aware
evaluator; legacy v1 run directories continue to use the original metric path.

## 5. Runtime dependencies

Install the Benchmark evaluation extras and Meta's official SAM2 package:

```bash
pip install -e ".[pendulum-evaluation]"
pip install -e /path/to/facebookresearch/sam2
```

The configured model is `facebook/sam2.1-hiera-tiny`. The evaluator loads it
lazily and reuses one predictor instance for all pendulum cases in the Task.
