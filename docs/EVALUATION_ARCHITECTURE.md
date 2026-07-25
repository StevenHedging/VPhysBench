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

`SceneEvaluatorRegistry` resolves a case evaluator by `scene_id`.
`scene_default_v1` implements all five official scenes:

| Scene | Evaluator type | Primary state |
| --- | --- | --- |
| `pendulum` | `pendulum_state_v1` | angle, period, amplitude, structure |
| `free_fall` | `free_fall_state_v1` | vertical path, acceleration, impact |
| `inclined_plane_slide` | `inclined_plane_state_v1` | plane-local path and acceleration |
| `uniform_circular_motion` | `uniform_circular_motion_state_v1` | relative angle, angular velocity, orbit |
| `collision_1d` | `collision_1d_state_v1` | instances, contact, pre/post velocity |

The `unsupported` status remains part of the contract for an unknown scene or
for a protocol that explicitly disables an evaluator; it is never converted to
a fabricated zero.

Each planned job produces exactly one status:

- `evaluated`: a score in `[0,1]` was produced;
- `unavailable`: prediction or trustworthy reference data are absent;
- `unsupported`: the scene algorithm has not been implemented;
- `error`: decoding, segmentation, tracking, or quality validation failed.

The official Task score is non-null only with complete coverage. Reports also
show an explicitly partial `observed_mean_score`.

## 3. Shared observation contract

The shared layer owns media probing, timestamp sampling, aspect-preserving
letterbox transforms, SAM2 propagation, mask quality checks, centroid/instance
tracking, small-N assignment, axis/circle fitting, geometry rectification, CSV
tables, and curves. Scene modules own prompt construction, state extraction,
quality policy, and scoring.

Except for the fixed five-second pendulum protocol, a scene uses a
case-reference-bounded interval:

```text
duration = min(reference last-frame time, protocol maximum)
```

Reference and prediction are sampled at the same physical timestamps. A
prediction that does not cover the interval is `unavailable`; no frame is
injected, repeated, or borrowed from GT.

Mask IoU is an observation diagnostic. The primary score always comes from a
scene-local physical state so that output resolution, harmless appearance
changes, and viewpoint normalization do not become the definition of physics.

Reference selection follows:

```text
same-case real reference
→ physics-identical parent reference
→ unavailable
```

The contract also reserves `physics_model` and `reference_free` for future
no-GT protocols. A GT-dependent IoU curve is never fabricated in those modes.

## 4. Pendulum protocol

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

## 5. Other scene protocols

### Free fall

The falling ball is independently segmented in reference and generation. The
evaluator compares normalized downward displacement, quadratic acceleration,
time to 95% travel, monotonic downward progress, and horizontal drift. Its
timeline is short and case-specific because the real captures last only a
fraction of a second. Initial height calibrates a reference-only acceleration
diagnostic in SI units.

### Inclined-plane slide

The block mask centroids define an independently fitted plane axis. Scoring uses
along-plane displacement, pre-arrival quadratic acceleration, descent time,
cross-track drift, monotonic progress, and block orientation stability.
`initial_velocity` is fitted rather than trusted from the unannotated label.
The evaluator reports both original-image IoU and an axis/scale-rectified IoU.

### Uniform circular motion

The green disk is localized first, then non-green object components are tracked
inside it. One- and two-object cases use continuity assignment and are ordered
by fitted orbit radius. Initial angle is deliberately eliminated by comparing
`theta(t) - theta(0)`. The state score combines angular trajectory, angular
velocity, circularity, uniformity, and multi-object radius configuration.
Center/radius-rectified IoU is reported beside original-image IoU.

### One-dimensional collision

Frame-zero color components identify the entering striker and adjacent target
pair; one shared SAM2 state propagates three instance masks. A fitted track axis
produces scalar trajectories, contact time, and pre/post velocity windows.
Mass annotations are used for a momentum residual. Because the dataset has no
trusted restitution label, effective restitution is estimated from reference
pre/post velocities and never guessed from material names. Diagnostics include
union IoU and identity-preserving instance IoU.

## 6. Outputs and aggregation

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

## 7. Runtime dependencies

Install the Benchmark evaluation extras and Meta's official SAM2 package:

```bash
pip install -e ".[scene-evaluation]"
pip install -e /path/to/facebookresearch/sam2
```

The configured model is `facebook/sam2.1-hiera-tiny`. The evaluator loads it
lazily and reuses one predictor instance per scene evaluator across the Task.
