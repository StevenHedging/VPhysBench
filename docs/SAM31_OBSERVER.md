# SAM3.1 prediction observer

The public `scene_default_v1` protocol uses observer revision 2 for prediction
segmentation. The protocol entrypoint is unchanged, but its configuration and
therefore its protocol and evaluator fingerprints differ from results produced
with the earlier observer settings. Results should retain those fingerprints in
their provenance rather than being compared as if the observer were unchanged.

## Frame-zero candidates

The observer sends each declared semantic text prompt on frame zero. It never
sends a ground-truth mask, box, point, crop, case identifier, or match result to
SAM. Ground truth is used only after prediction candidates exist, to bind their
frame-zero masks to the declared benchmark entities.

`initial_detection.score_threshold` is the detector proposal gate.
`initial_detection.new_object_threshold` is the new-object birth gate and must
be at least the detector gate. Both are finite floating-point values in
`[0, 1]`. Revision 2 sets both to `0.2`. These overrides exist only during the
initial text request; the native backend values are restored before propagation
and after failures. Omitting `initial_detection` preserves the backend gates.

`output_probability_threshold` is a separate argument supplied to the backend
for prompt and propagation requests. The pinned multiplex path accepts this
argument but does not guarantee that it filters exported probabilities, so it
must not be interpreted as an enforced confidence floor. The IoU threshold used
for identity acceptance is separate again.

## Identity and temporal output

Revision 2 uses `threshold_feasible_v2` matching. It chooses a complete,
one-to-one frame-zero assignment only among candidate/entity edges that meet the
configured IoU threshold. Shortage, infeasibility, or ambiguity produces an
explicit initialization failure; it does not trigger another prompt. Once
accepted, the backend object ID is fixed for the common physical timeline and
is never rebound from later ground truth.

The instance compatibility policy records every newly born object as a
conditioning input when the pinned tracker would otherwise classify its birth
as a non-conditioning correction. The wrapper leaves the native state update,
memory encoding, frame history, and index maintenance in place and restores the
backend flag after each call.

Revision 2 also sets `masklet_confirmation_enable` to `false` for the whole
prompt-group session. Discovery confirmation can hide a nonempty mask until the
same discovery is confirmed across frames, which conflicts with a benchmark
that already fixed identity on frame zero. Disabling that output filter exposes
the backend's real mask for the fixed ID. This confirmation setting itself does
not synthesize or interpolate masks, disable removal, expose suppressed
unmatched objects, rebind IDs, or change CSTI termination. The native setting
is restored after session completion or failure. Omitting the option preserves
native behavior.

The public `discovery_pruning_policy` is `fixed_initial_ids_v1`. During the
whole prompt-group session it sets the backend hotstart delay to zero and limits
unmatched suppression to the now-empty hotstart interval. This prevents later
text-detector misses from pruning an object whose identity was already fixed on
frame zero. Both native settings are restored after success or failure. The
legacy `backend_native_v1` policy makes no override and remains segmenter policy
revision 1 when no other revised adapter policy is configured.

Fixed-initial pruning does not force presence logits, synthesize masks, change
occlusion handling or the object budget, admit late backend IDs into the locked
candidate set, or use ground truth to select an output. Native empty masks and
identity drift remain observable limitations; persistent false detections are
also possible and must be reported rather than interpreted as successful
tracking merely because a mask is nonempty.

## Provenance and failures

Segmenter provenance distinguishes the configured frame-zero gates from the
restored native propagation gates. Its `segmenter_policy_revision` describes
only those adapter-level policies. It also records the requested, effective and
native discovery-confirmation setting, the requested export-probability policy,
the birth-conditioning policy, and requested/native/effective discovery-pruning
settings. Top-level observer provenance records the authoritative observer
revision and initial matching policy.

An explicitly configured policy requires the corresponding pinned backend
attributes. Missing attributes are evaluator interface errors rather than a
silent fallback. Initialization shortage and ambiguity remain null CSTI
outcomes, while runtime and interface faults remain evaluator errors.

The public `0.2` gates were selected by comparing fixed coherent detector/birth
floors against the previous native settings and by checking independently
selected controls. Lowering a proposal gate can add false or ambiguous
candidates and is not evidence of better segmentation by itself. The revised
policy improves initial candidate availability but does not guarantee complete
tracking when the native tracker loses or removes an object. It does not add
multi-scale retries or any case-conditioned recovery path.
