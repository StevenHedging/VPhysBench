# SAM 3.1 Collision GT Resegmentation Design

## Objective

Rebuild the first-frame anchors and frozen reference-observation tubes for all
330 `collision_1d` Cases with the pinned SAM 3.1 checkpoint.  The new assets
must bind `object_1..object_N`, `ball_1..ball_N`, and mask IDs `01..NN` to the
same physical subjects as `physics.json` and `caption.json`.

## Dataset facts

- The release contains 264 two-ball Cases and 66 three-ball Cases.
- 298 Cases are 832x480 and 32 R2 Cases are 1920x1080.
- Every collision timeline is 24 Hz and contains 12--196 samples.
- Physics object keys are contiguous and every caption contains the matching
  `m_N`, `r_N`, and `v_N` symbols.
- Existing anchors and tubes are evidence only.  They are never prompts or
  candidate geometry for the new segmentation.

## Architecture

### Candidate discovery

SAM 3.1 first receives the scene prompt `small round object` on frame zero.
When the exact physical count cannot be selected confidently, the curation
pipeline tries the configured prompt ensemble (`ball`, material-specific ball
phrases) and border crops.  Crop detections are mapped back to full-frame
coordinates and used only to create SAM 3.1 geometric prompts; the persisted
pixels always come from SAM 3.1.

Candidates from multiple sessions are deduplicated by first-frame mask IoU and
centroid distance.  A global selector chooses exactly the count declared by
`physics.objects`, using confidence, compactness, size consistency with the
declared radii, the collision-lane row, and temporal persistence.  A shortage
or ambiguous optimum is a curation failure, never a guessed annotation.

### Identity binding

Selected first-frame masks are grouped into visual rows with a tolerance based
on median mask height.  Rows are ordered by median center y from top to bottom;
masks inside a row are ordered by center x from left to right.  The resulting
row-major sequence binds to `object_1..object_N`.  For collision this normally
reduces to strict left-to-right ordering.

The binding produces the following immutable tuple for each subject:

```text
physics object_N <-> evaluator ball_N <-> mask NN <-> SAM backend object ID
```

The mapping is fixed at frame zero.  Later frames cannot reassign identities.

### Stateful tube generation

The official multiplex video predictor is reused per worker.  After text-only
discovery, each selected subject receives its own stateful session.  Frame zero
is initialized with the same text plus one positive tight box; the other
selected subjects are supplied as negative boxes.  This isolates same-class
balls without making their persisted masks mutually exclusive.  Each session
then propagates forward over the existing 24 Hz reference timeline.

Short internal gaps are repaired by a frame-local SAM 3.1 box request derived
from the subject's adjacent valid observations.  An unexplained trailing drop
is reinitialized once and propagated forward; a verified boundary exit remains
empty and is labelled `OUT_OF_FRAME`.  All repairs remain inside the locked
semantic subject session.

SAM backend IDs can swap at ball contact even when their masks remain visually
correct.  Collision identities therefore use the physical non-penetration
invariant: at every frame, visible material components are assigned to the
fixed left-to-right slots in an order-preserving manner, with short missing
sets assigned by predicted centroids.  Finally, if one subject mask contains
two disconnected ball-sized components, temporal continuity retains exactly
the component belonging to that subject.  Pixel overlap between touching
subjects is still allowed.

### Quality gates

Every candidate Case must satisfy:

1. exact agreement among physics objects, caption symbols, anchors, tubes,
   trajectories, review records, and manifest entities;
2. exact row-major frame-zero numbering;
3. anchor mask equality with tube sample zero;
4. binary, shape-aligned, nonempty frame-zero masks;
5. no unexplained internal disappearance, identity-order inversion, or nearly
   duplicate full-subject tubes;
6. normal contact overlap remains allowed and is never resolved with exclusive
   ownership;
7. lifecycle states contain no unexplained `UNRESOLVED` samples;
8. every file digest and asset-lock record matches the installed bytes.

The pipeline writes contact sheets, trajectory images, overlay videos, and a
machine-readable ledger.  Automated gates can reject candidates but cannot use
the old GT to approve semantic correctness.  All flagged Cases receive dense
visual review before installation.

### Staging and publication

Candidates are written beneath a removable shared `.local` run directory.
Canonical Dataset assets are not touched during inference or review.  Accepted
candidate bundles are validated and installed atomically with the existing
curation installer, after which dependent manifests, the release asset lock,
and curation provenance are refreshed.

The first smoke gate is the 13 Cases diagnosed on 2026-08-27.  The production
gate is all 330 collision Cases, followed by Dataset validation and a complete
re-evaluation of the existing 330 generated videos.  The new GT and evaluator
both use SAM 3.1, so the final report explicitly records this shared-model bias
and includes geometry/motion checks that do not depend on the old GT or on
SAM-to-SAM agreement.

## Runtime

- Source revision: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- Checkpoint SHA-256:
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`
- Checkpoint path is supplied by configuration/environment, never published as
  a Dataset absolute path.
- Production generation used 54 idle A100 GPUs across seven hosts.  GPU 0 on
  `n30237` and `n30241` and all of `n30238` were excluded because they were in
  use by unrelated jobs.
