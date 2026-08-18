# V14 Full Reference-Observation Audit and In-Place Repair Design

## Objective

Audit all 916 Dataset 14.0.0 Cases across all seven scenes and establish, with
per-Case evidence, that:

1. `physics.objects.object_N` follows the scene's declared visual ordering and
   refers to the same physical subject as first-frame mask `NN`;
2. the first-frame anchor accurately segments that subject rather than a
   highlight, apparatus part, shadow, neighboring object, or background;
3. every frozen reference-observation tube preserves subject identity and has
   accurate, stable masks throughout its declared lifecycle; and
4. every detected defect is repaired directly in the current V14 assets, with
   all dependent manifests, visualizations, reviews, and hashes regenerated.

The audit covers `push_bottle` and training Cases as well as the six-scene
official evaluation subset. Existing `approved` review records are evidence to
inspect, not authority that can waive re-review.

## Current-State Findings That Motivate the Work

The initial read-only inventory found 916 Cases and 1,324 physical entities.
All Cases currently have the expected caption, physics, first-frame, mask,
reference-video, reference-observation, and visualization assets. Structural
links are internally complete: object IDs, mask IDs, physics-key prefixes, and
entity IDs have no missing or non-contiguous members.

That structural success does not establish semantic correctness. Two collision
Cases already contradict the declared left-to-right convention:

- `collision_supp_20260729_img_1073_two_ball_single_incident`;
- `collision_supp_20260729_img_1190_two_ball_single_incident`.

In both, physics and caption identify `object_1` as the left stationary large
ball, while the first frame places that large ball to the right of `object_2`.
The existing masks and tubes preserve the physics numbering, so the whole Case
is self-consistent at the ID-field level while semantically violating its own
caption and ordering contract.

A deliberately broad trajectory-only screen also flagged 353 of 1,324
entities for area change, centroid displacement, or unresolved states. Many
are expected false positives caused by high projectile speed, perspective,
contact, or boundary exit. Conversely, a tracker can follow the wrong object
smoothly and evade these metrics. The production audit must therefore combine
scene-aware metrics with visual review of every Case.

## Non-Negotiable Constraints

- Operate on the current Dataset 14.0.0 assets; do not create `v2`, `fixed`,
  `backup`, or parallel Dataset releases.
- Temporary candidates may exist only under `.local/` or an OS temporary
  directory and must be removable after acceptance.
- Never overwrite a canonical Case until its candidate bundle passes
  structural validation and has a recorded visual decision.
- Accepted replacements are installed atomically, then all transitive hashes
  are rebuilt. A partially written canonical bundle is a hard failure.
- Preserve unrelated user changes in the dirty worktree, including the current
  modifications to `assets.lock.json`.
- Automated scores prioritize review but never auto-approve semantic identity.
- Use all eight local A100 GPUs for independent Case shards. One worker owns
  one GPU and one Case at a time; workers never write the same Case.
- The default repair model is `facebook/sam2.1-hiera-large`. Existing tiny-model
  output remains a comparison candidate, not the default repair authority.

## Repository Layout

Reusable, testable logic belongs in a focused curation package:

```text
src/physbench/reference_observations/curation/
├── __init__.py          # public curation interfaces
├── catalog.py           # Case/member discovery and typed audit inputs
├── identity.py          # numbering, physics, caption, and anchor bindings
├── quality.py           # scene-aware trajectory/mask diagnostics
├── anchors.py           # independent anchor evidence and prompt construction
├── tracking.py          # SAM2 propagation, re-seeding, lifecycle states
├── visualization.py     # audit sheets and dense-event rendering
├── bundle.py            # candidate bundle generation and validation
├── install.py           # atomic in-place installation and hash refresh
└── overrides.py         # validated declarative per-Case overrides
```

Operator entry points belong together:

```text
scripts/reference_observations/
├── audit_v14.py
├── rebuild_cases.py
├── render_review_queue.py
└── repairs/             # only irreducibly algorithmic Case-specific repairs
```

Declarative exceptions are preferred to Python scripts and live at:

```text
configs/reference_observations/
├── audit_v1.json
└── overrides/<case_id>.json
```

An override can declare reviewed anchor boxes/points, additional correction
frames, lifecycle boundaries, or a scene refinement mode. It cannot directly
embed masks or bypass validation. A Python file under `repairs/` is allowed
only after the general tracker plus declarative corrections have demonstrably
failed for that Case.

The durable review ledger and summary live under `docs/audits/`:

```text
docs/audits/v14_full_reference_observation_audit_20260818.jsonl
docs/audits/v14_full_reference_observation_audit_20260818.md
```

The JSONL has exactly one authoritative row per Case. It records check results,
visual reviewer decision, evidence paths, repairs, and final asset digests. It
does not duplicate media.

## Audit Pipeline

### 1. Catalog and Structural Binding

For every indexed Case, load caption, physics, first-frame mask manifest,
reference-observation manifest, timeline, quality, review, visualization
manifest, mask NPZs, and trajectory NPZs. Validate:

- contiguous `object_1..object_N` and `01..NN` identities;
- exact correspondence between each object's formal physics keys and its mask;
- identical entity sets across physics, anchors, tube, trajectory, and review;
- first observation mask equals the reviewed anchor unless a documented
  nonzero seed policy explicitly says otherwise;
- trajectory area, centroid, and bbox are exact reductions of the stored mask;
- timeline indices, source frames, and lifecycle state arrays align;
- every stored file record and top-level asset-lock entry matches its bytes.

### 2. Scene-Aware Object Ordering and Semantics

Ordering is validated against the canonical first frame, not directory names
or XLSX order:

- `collision_1d`: visible ball centroids are numbered strictly left to right.
  Caption roles, radii/diameters, appearance labels, and pre-contact velocity
  directions must agree with that numbering. Partial border masks use their
  visible centroid but must be manually checked.
- `uniform_circular_motion`: subjects use the declared row-major order. A row
  grouping tolerance is derived from median subject height and is written to
  the audit record.
- single-object scenes: the one object must be the moving physical subject
  declared by the scene, not the track, launcher, string, spring, hand, or
  support.

When numbering is wrong, the repair is Case-wide: physics object entries and
symbols, caption roles, appearance arrays where positional, anchor files and
manifest entries, tube entity directories and records, review records, and all
dependent hashes are remapped together. Renaming mask files alone is forbidden.

### 3. Independent First-Frame Anchor Verification

Existing masks cannot be their own ground truth. Each Case receives independent
evidence built from:

- scene-specific foreground/geometry constraints;
- motion departure or arrival support across separated source frames;
- SAM2.1 Large proposals prompted from independently estimated subject boxes;
- expected subject count, relative size, and apparatus exclusion regions;
- comparison against the existing anchor using IoU, containment, component
  count, border contact, and boundary disagreement.

Every Case gets an audit image showing the unmodified first frame, numbered
existing masks, independent candidates, physics/appearance summary, and all
disagreements. The visual review decision is one of `pass`, `repair`, or
`needs_dense_review`; there is no automatic `pass` based solely on IoU.

### 4. Full-Tube Verification

The fast pass reads trajectory and lifecycle arrays without unpacking all
pixels. Diagnostics are scene-aware and include:

- robust log-area change relative to perspective and boundary state;
- centroid acceleration and direction consistency in physical time;
- component fragmentation, holes, mask-border contact, and overlap;
- identity ordering before/after interactions;
- unexpected disappearance, reappearance, `OCCLUDED`, `OUT_OF_FRAME`, and
  `UNRESOLVED` runs;
- agreement between existing tiny-model output and an independently propagated
  large-model candidate at sampled and event-dense frames.

Every Case's contact sheet is visually inspected. Cases with contact,
occlusion, boundary exit, model disagreement, QC warnings, or geometric flags
receive dense-event sheets and full overlay-video inspection. Smoothness never
proves identity by itself.

### 5. Repair Strategy

Repair escalates in this order:

1. re-propagate from a verified frame-zero anchor with SAM2.1 Large;
2. use joint multi-instance propagation with highest-positive-logit ownership;
3. add reviewed positive/negative points and boxes at the failure frame, then
   propagate forward and backward;
4. set explicit lifecycle boundaries only where dense video evidence proves
   physical occlusion or complete exit;
5. apply a declarative Case override;
6. add a narrowly scoped Case repair script if all general mechanisms fail.

Each attempted repair records the hypothesis, evidence, prompt frames, model,
configuration fingerprint, and comparison to the previous candidate. Failed
candidates remain temporary and are removed; they are not installed or kept as
Dataset versions.

### 6. Candidate Acceptance and In-Place Installation

A candidate may replace canonical files only when:

- object identity and ordering are visually correct;
- anchor and observation zero agree exactly;
- all visible masks cover the intended subject without material apparatus or
  neighbor pixels;
- contact does not cause identity swaps or uncontrolled mask fusion;
- area changes are physically/visually justified;
- lifecycle states match actual visibility;
- no `UNRESOLVED` sample remains without an explicit reviewed justification;
- generated trajectories exactly reduce from the masks;
- the candidate bundle passes schema, digest, and portability validation.

Installation writes new files to sibling temporary paths, fsyncs them, and
uses `os.replace` only after the complete Case candidate is valid. Dependent
manifests are rebuilt inside-out, followed by the release asset lock. If any
post-install validation fails, work stops on that Case before proceeding;
there is no silent partial acceptance.

## Review Evidence and Completion Criteria

The task is complete only when the following current-state evidence exists:

- the audit JSONL contains 916 unique rows and exactly matches the release Case
  set;
- every row has a final visual decision and identifies the actual evidence
  inspected;
- every multi-object Case passes scene ordering and Case-wide semantic binding;
- every first-frame anchor has an explicit visual decision based on the source
  image, not only its existing manifest;
- every full tube has an explicit visual decision; flagged event ranges have
  dense-review evidence;
- repaired Cases pass the same checks after installation;
- there are no unreviewed failures, unexplained unresolved samples, stale
  review digests, or stale visualization/asset-lock records;
- targeted curation tests, the complete project test suite, release audit, and
  `physbench validate-dataset --check-assets` all succeed;
- a final Markdown report lists per-scene reviewed/repaired counts, every
  repaired Case and root cause, any justified residual limitations, exact
  verification commands, and their outputs.

Existing review decisions are replaced or amended only from newly recorded
evidence. Passing structural tests or finding no additional automatic flags is
not sufficient evidence of completion.
