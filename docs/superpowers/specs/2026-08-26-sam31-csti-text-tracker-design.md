# SAM 3.1 Text-Prompted CSTI Tracking Design

## Objective

Make CSTI identity initialization and prediction Tube extraction robust across
all six scored VPhysBench scenes by using the official SAM 3.1 multiplex video
predictor with text prompts. Keep the published `exact_full_tube_edt` CSTI
implementation and its mathematical inputs unchanged.

The generated video's first frame is the supplied I2V condition frame. SAM 3.1
detects and tracks physical-subject candidates from text only. Frozen GT masks
are used only after detection to bind SAM object IDs to benchmark entity IDs on
frame zero.

## Existing Boundaries

The current evaluator has three relevant layers:

1. Scene evaluators decode and align reference and prediction videos, load the
   frozen V14 observations, and produce `CSTIInput` objects.
2. `physbench.evaluation.common.csti.metric` validates aligned binary Tubes,
   evaluates the existing cumulative soft Tube IoU, and macro-averages the
   legal GT entities.
3. `task_evaluator` aggregates per-case CSTI records into the dataset result.

The new observer sits between layers 1 and 2. It replaces only the prediction
masks and matched track IDs in the CSTI input. Existing scene-specific expert
observers and physics scores remain unchanged.

## Architecture

### SAM 3.1 backend adapter

`physbench.evaluation.common.masks.sam31_text` owns model loading, shared
predictor reuse, aligned-frame serialization, official session calls, and
output normalization. It lazily imports the pinned official SAM 3 source and
builds the predictor with `build_sam3_multiplex_video_predictor`.

One process reuses one predictor per immutable runtime configuration. Inference
on a predictor is serialized because official predictor sessions share model
state and are not documented as thread-safe. Every text concept uses its own
session because the installed SAM 3.1 `add_prompt` implementation resets the
semantic state when a new text prompt is added. A session performs:

1. `start_session` on the aligned frame directory;
2. `add_prompt` with only `text` at `frame_index=0`;
3. `propagate_in_video` with `propagation_direction="forward"`;
4. `close_session` in a `finally` block.

The adapter records `out_obj_ids`, `out_binary_masks`, `out_boxes_xywh`,
`out_probs`, prompt identity, and frame index. Object IDs are namespaced by
prompt group because backend IDs are session-local. No GT mask, box, point, or
trajectory is passed to SAM.

### Prompt and expected-count mapping

Each scene declares `csti_observer.prompt_groups`. A group contains a text
prompt and the compatible manifest `entity_classes`. Its expected instance
count is derived from the immutable entity manifest for the current case. Every
manifest class must belong to exactly one group. This supports variable body
counts in collision without case-specific configuration.

The initial candidate count is checked independently per prompt group. Extra
candidates are retained as diagnostics but never enter CSTI.

### Frame-zero identity binding

`physbench.evaluation.common.csti.observation` is a pure NumPy/SciPy layer. For
each prompt group it computes the binary-mask IoU matrix between compatible GT
frame-zero masks and SAM frame-zero candidates. Hungarian assignment maximizes
the total IoU.

Initialization succeeds only when:

- every expected entity has a compatible candidate;
- the assignment is complete and one-to-one;
- every assigned IoU meets `initial_match_iou_threshold`; and
- the best complete assignment is separated from the best alternative
  complete assignment by at least `initial_match_ambiguity_margin` when the
  latter is feasible above the IoU threshold.

The resulting `entity_id -> namespaced SAM object ID` mapping is immutable for
the entire video. Later frames never run Hungarian matching, identity recovery,
or re-identification. Extra candidates cannot replace a locked identity.

Candidate shortage, incomplete assignment, low IoU, or configured ambiguity is
an `evaluator_init_failure`. It produces a visible CSTI record with a null
score, not a zero. Missing SAM dependencies, checkpoint mismatch, invalid
backend tensor shapes, and unexpected implementation exceptions remain
evaluator errors because they indicate a broken installation or implementation.

### Locked Tube lifecycle

For each locked object ID, the observer emits a binary mask or an empty mask at
every common timeline sample. `is_valid_track_observation` validates object-ID
presence, two-dimensional shape, finite/binary content, configured minimum
pixel area, and optional confidence. It does not introduce motion, area-change,
or drift gates unless an existing documented threshold is explicitly wired in.

The role state is `ACTIVE`, `TERMINATED`, or `VIDEO_END`:

- one or two consecutive invalid observations remain empty but provisional;
- a later valid observation before the configured patience clears the counter;
- `termination_patience` consecutive invalid observations confirm permanent
  termination, defaulting to three;
- the termination frame is backdated to the first invalid observation and all
  masks from that frame to video end are empty;
- after confirmation no observation can restore the role.

If video end arrives with fewer than `termination_patience` pending invalid
frames, those frames stay empty, no termination frame is reported, and the
final state is `VIDEO_END`. This is conservative because unsupported pixels are
never invented, while the evaluator does not claim a permanent disappearance
without the required evidence.

The timeline already comes from VPhysBench's physical-overlap sampler, so its
length is the shorter physical duration of GT and generated video. The observer
must preserve that entire common timeline, including empty masks after
termination.

### CSTI adapter and immutable metric

On successful initialization, the observer creates a new `CSTIInput` by
copying the reference capability, times, frame shape, entity order, role IDs,
and reference masks from the existing aligned input and replacing only:

- `prediction_masks`; and
- `matched_prediction_track_ids`.

`common/csti/metric.py` is not modified. A regression fixture evaluates the
same Tubes through the pre-change and post-change path and requires identical
per-subject and video scores. Prediction Tube and GT Tube remain equal in
length, spatial shape, binary semantics, and time index.

The metric result is augmented, outside the mathematical implementation, with
the requested audit fields: video score alias, per-subject scores, subject
count, initialization status/reason, initial mapping and IoUs, termination
frames, and observer diagnostics.

### Failure integration and aggregation

The CSTI observer runs after aligned media are available and independently of
scene expert identity success. When a scene prediction observer degrades but
the SAM CSTI observer succeeds, the expert score remains degraded while CSTI
uses the SAM Tubes. Reference failures remain `unavailable`.

Dataset CSTI averages all `evaluator_init_success` video scores with equal
video weight. `evaluator_init_failure` videos are excluded from that mean and
reported separately. The result includes:

- `init_coverage`;
- `valid_video_count`;
- `evaluator_init_failure_video_count`; and
- failure-reason counts.

`not_applicable` videos remain outside the CSTI-applicable denominator. Other
prediction failures continue to follow the existing fail-closed finite-zero
policy and are not reclassified as evaluator initialization failures.

## Configuration

Every scored scene receives a `csti_observer` block containing:

- backend ID, checkpoint environment variable/path, expected digest, device,
  precision, compilation and frame-loading choices;
- prompt groups and manifest-class mappings;
- output mask/probability threshold;
- initial matching IoU and ambiguity thresholds;
- `termination_patience`, default three;
- minimum mask pixels and minimum observation confidence; and
- debug Tube/overlay output flag.

No absolute checkpoint path, Dataset path, or GPU index is published. The
checkpoint defaults to the existing environment-variable mechanism and is
verified by SHA-256.

## Output Contract

Successful case-level CSTI records include:

- `status="evaluated"`, `score`, and `csti_video`;
- `csti_per_subject` and existing `objects` detail;
- `subject_count`;
- `evaluator_init_success=true` and null failure reason;
- `initial_matching` and `initial_matching_iou`;
- `termination_frame_per_subject`; and
- SAM prompt, candidate, confidence, box, and session diagnostics.

Initialization failures include the same audit surface with
`status="evaluator_init_failure"`, `score=null`,
`evaluator_init_success=false`, and a structured failure reason. They remain
visible in case JSON, logs, and dataset summaries.

## Verification

CPU tests use synthetic masks and a fake official-request predictor. They cover
all thirteen required matching, lifecycle, metric-regression, macro-average,
aggregation, and validation cases. Adapter tests prove that prompts contain no
GT geometric fields, propagation is forward-only, sessions close on success
and exceptions, outputs retain boxes/confidence, and a predictor is reused.

A real-model smoke test uses `/public/model/sam3.1/sam3.1_multiplex.pt` only
when the shared SAM 3.1 environment and a GPU are available. Representative
VPhysBench cases from all six scenes are then replayed to measure initialization
coverage and inspect debug overlays. Repository evaluator tests, protocol
schema validation, type/import checks, and the release preflight remain gates.

## Rejected Approaches

### Patch each scene observer independently

This duplicates matching and termination semantics six times and leaves CSTI
coverage dependent on unrelated expert-scoring heuristics.

### Reuse the previous SAM 3.1 role-recovery branch unchanged

That branch supplies GT-derived mask/box/point prompts to SAM and performs
post-initialization recovery and reassociation. Both conflict with this
protocol's text-only initialization and locked identity semantics.

### Treat every failed detection as model error and score zero

The generated first frame is the exact condition frame. Failing to initialize
its declared legal subjects is therefore an evaluator coverage failure, which
must be reported rather than silently penalizing the model.

