# Generic SAM3.1 Observer Robustness

## Scope and authority

Improve benchmark-owned prediction segmentation, initial identity binding and
tracking-state safety. The user authorized autonomous implementation, validation,
main-worktree integration and publication to the dated release branch. Research
videos, reports, local paths, weights and model-specific generation code remain
outside this repository. No case identifier may select evaluator behavior.

## Boundaries

- Segmentation consumes predicted RGB/BGR frames and declared semantic text only.
- GT masks participate only in the existing post-detection frame-zero identity
  matching. They never prompt SAM, select a detection retry, or recover a track.
- Identity is fixed on frame zero for the entire common physical timeline.
- Initial shortage/ambiguity remains an explicit null; runtime faults remain
  evaluator errors, not fabricated scores or silent empty successful tubes.
- The exact CSTI mathematical implementation and historical results are unchanged.
- Shared third-party sources and environments are immutable. Compatibility fixes
  wrap the loaded predictor instance and are recorded in observer provenance.

## A. Tracking birth invariant

The pinned multiplex implementation classifies a newly born object's input as a
non-conditioning correction when its frame was already tracked for another
object. Deleting the earlier object's last conditioning frame can consequently
erase all surviving input history while retaining IDs.

Install an idempotent wrapper on the loaded underlying tracker's bound
`add_new_masks`. For calls with previously unknown IDs and `reconditioning=False`,
temporarily set `add_all_frames_to_correct_as_cond=True` and restore it in a
`finally` block. Keep the native method, frame history, memory encoder and index
maintenance intact. Ordinary existing-object updates are unchanged. Serialize
calls using the existing predictor lock. Missing required pinned interfaces must
fail clearly when the fix is requested; test predictor factories remain usable.

Separately, skip video propagation for a prompt group with no frame-zero IDs.
Keep session cleanup and let the existing matching layer report initialization
failure. This guard is not evidence that the underlying state bug was repaired;
that repair requires a raw-predictor replay traversing the original failing path.

## B. Feasible identity matching

Add an explicit matching policy. Preserve legacy maximum-total-IoU behavior for
legacy configurations; the new policy maximizes total IoU only over assignments
whose every edge reaches the configured threshold. Use the same constrained
solver for the best assignment and for alternative assignments used in ambiguity
testing. No feasible complete matching means initialization failure. Hall
conflicts, rectangular matrices and tiny numerical differences must not turn
into crashes or duplicate identity use.

Retain the complete initial IoU matrix and candidate IDs in diagnostics for both
success and failure. Do not alter future-frame identity or termination semantics.

## C. Explicit initial candidate policy

Distinguish detector proposal score, new-object score and the requested export
probability argument. The pinned multiplex backend accepts the latter argument
but does not apply it as an output probability filter on this path; provenance
and documentation must not imply that it guarantees an exported score floor.
Expose validated, provenance-recorded frame-zero detector
and birth gates, applied only while executing the initial text prompt and
restored before normal video propagation, including after exceptions. Keep
export confidence and IoU acceptance separate. No GT-driven retries, extra
case-conditioned prompts, arbitrary spatial crops or method-dependent branches.

Choose the public frame-zero gate using a documented, fixed-policy calibration
comparison: retain the old gates as control; compare coherent detector/birth
floors 0.3, 0.2 and 0.1. Inspect candidate precision as well as complete matching
and validate on independently selected settings. A lower gate is not by itself
proof of better segmentation. If these gates cannot recover correct masks,
document the remaining limitation before proposing a separate image-only
multi-scale subsystem. Do not introduce unvalidated multi-scale tracking in this
change. Preserve old configurations and make the public observer revision and
resolved candidate/matching policies explicit in configuration/provenance.

## D. Separate discovery confirmation from fixed-ID visibility

Full-video inspection found nonempty internal masks hidden solely because the
backend had not yet confirmed a discovery through consecutive detections. This
can exceed the benchmark's existing invalid-frame patience even though tracking
continues with the same ID. Paired native probes found disabling this output
confirmation filter preserves the underlying masks, initial identities, removal
decisions and unmatched-object suppression while exposing those real masks.

Expose an optional strictly boolean `masklet_confirmation_enable` segmenter
setting. Legacy omission leaves the backend default intact; the revised public
observer explicitly disables it for its fixed-frame-zero identity contract.
Apply it to the predictor instance for the entire prompt-group session under the
existing lock, restoring the original value even on failure. Record requested,
effective and native values. Missing required backend support must fail clearly.
Do not disable removal or unmatched suppression, synthesize masks, rebind IDs,
or alter benchmark termination semantics. This does not guarantee complete
tracking when the native tracker actually loses or removes an object.

## Validation and publication

1. Individually inspect all previously failed initializations and the runtime
   error: exact input hashes, frame-zero candidates, full IoU matrices and visual
   overlays. Keep every sample, including unrecovered ones.
2. Verify native state regression with real vendor state machinery and a raw
   single-video GPU replay that bypasses the new empty-initialization guard.
3. Run all-video initial coverage comparison and full-video smoke/regression on
   recovered failures and normal controls; broaden full-video validation before
   reporting aggregate metrics. No old score is overwritten.
4. Run CPU observer/adapter, integration, metric-invariance, public interface and
   release checks. Review individual changes and the final branch independently.
5. Publish only benchmark source, tests, portable protocol and documentation.
   No baseline bundles, diagnostics or generated assets enter the release tree.
