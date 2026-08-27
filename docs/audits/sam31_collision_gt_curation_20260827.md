# SAM 3.1 collision GT curation audit (2026-08-27)

## Scope and identity contract

All 330 `collision_1d` Cases in Dataset 14.0.0 were rebuilt: 264 two-ball
Cases and 66 three-ball Cases.  The curation run preserved every Case's 24 Hz
timeline and original sample count.  Existing anchors and tubes were not used
as SAM prompts or candidate geometry.

Frame-zero subjects are numbered by visual matrix order: rows from top to
bottom and instances within a row from left to right.  This binds
`physics.objects.object_N`, caption symbols `m_N/r_N/v_N`, evaluator ID
`ball_N`, mask ID `NN`, and frozen observation entity `object_N` to the same
physical subject.

## Frozen implementation

- Benchmark curation revision: `cb68f92`
- Configuration fingerprint:
  `7cc93dc79d9ae621e754eaa8719c95c845b4e0ebce477bf60e745d49920ae656`
- SAM source revision: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- SAM checkpoint SHA-256:
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`
- Candidate review-ledger SHA-256:
  `b48f0f872b91c2f2663a9f583bb59c59762ce2c4fc55fe8c502e4c3ee737beb4`

The final tracker uses text-only frame-zero discovery, one independent
text-plus-positive-box session per selected subject, negative boxes for the
other same-class subjects, forward stateful propagation, targeted internal
and trailing gap recovery, collision-order canonicalization, and temporal
connected-component selection.  Masks remain independent and may overlap at
contact.

## QA and publication

- Candidate bundles generated: 330
- Candidate bundles passing digest/reduction validation: 330
- Empty visible masks: 0
- Unresolved or occluded runs: 0
- Material fragmented-mask events: 0
- Remaining priority diagnostics: one genuine high-speed centroid jump in
  `collision_r2_small_steel_small_steel_small_steel_v10861`
- Installed candidate files: 5,874
- Install failures: 0
- Published asset-lock digest:
  `3efbd1d2327f465efdceb9fe05b7bcbbbc700fc9092077b448d287d1d9611cd8`
- Hugging Face Dataset revision:
  `9d0ba0467b31fdd02248cd7697ef6b459c7c06bd`
- Full Dataset asset and SHA-256 validation: passed (916 Cases, seven scenes)

The pre-install canonical bytes and their digests are retained in the shared
run provenance so the publication is recoverable.

## Re-evaluation of the 330 existing predictions

| Result | Before | After |
|---|---:|---:|
| Valid CSTI videos | 317 | 329 |
| Initialization failures | 13 | 1 |
| Initialization coverage | 0.960606 | 0.996970 |
| CSTI mean over valid videos | 0.316880 | 0.319352 |

Twelve of the thirteen earlier initialization failures are now evaluated.
Across the 317 videos valid in both runs, mean CSTI changed by `+0.006047`
(median `+0.002506`; 244 increased and 73 decreased).

The remaining failure is
`collision_r2_small_steel_medium_steel_large_steel_v01985`.  Its first small
ball is only partially visible at the left image boundary.  The frozen GT mask
is correct, but the online evaluator's three text candidates contain no mask
overlapping that ball, so its matching IoU is zero.  This is an evaluator
frame-zero discovery miss rather than a GT segmentation defect.

GT curation and online evaluation use the same pinned SAM 3.1 checkpoint.
This can correlate category and boundary errors and should be treated as a
shared-model bias.  Reference and generated videos are still tracked
independently, and the modest common-case score change indicates that the main
measured effect here is improved initialization coverage rather than a large
uniform score inflation.
