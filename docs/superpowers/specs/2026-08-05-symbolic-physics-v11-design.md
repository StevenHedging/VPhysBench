# Symbolic Physics and Directional Prompt Dataset 11.0.0 Design

## Goal

Publish `physics_video_six_scene_v11` as the official Dataset default. Every
structured-physics quantity receives a stable mathematical symbol, every
independent conditionable quantity is named by that symbol in the English Case
prompt without exposing its numeric value, and every stored scalar value is
non-negative. Direction and stationary-state semantics belong to the prompt,
not to the sign of a scalar quantity.

Dataset 10.0.0 remains immutable and loadable. Dataset 11.0.0 changes metadata
only: no video, image, mask, archive, source document, View membership, Case ID,
Scene ID, temporal fact, alignment fact, appearance fact, or provenance source
fact may change.

## Why This Is a New Release

Overwriting the 799 existing `physics.json` files would invalidate Dataset
10.0.0's asset lock and make the published V10 Cases disagree with their
Case-local documents. V11 therefore adds one new file beside each old file:

```text
datasets/assets/<scene>/<case>/physics.v11.json
```

V10 continues to reference `physics.json`. V11 references
`physics.v11.json` through the same `assets.physics_annotation` role. The old
file is not referenced or locked by V11, but remains tracked for historical V10
reproducibility.

## Schema Contract

V11 uses Dataset and Case schema `5.0`. Scene schema remains `2.0`, View A
remains `3.0`, View B remains `2.0`, and Task schema remains `4.0` because their
field shapes do not change.

Each Case quantity has exactly four fields:

```json
{
  "value": 0.03313,
  "unit": "kg",
  "annotated": true,
  "symbol": "m_1"
}
```

`symbol` is the canonical plain-text mathematical representation used verbatim
in `case.text.prompt`. Symbols use Unicode Greek letters where conventional,
ASCII underscores for indices, and no `$...$`, `\(...\)`, braces, or other
LaTeX delimiters. Examples are `m`, `m_1`, `θ`, `θ_0`, `μ_k`, `ω`, and
`F_peak`.

All quantity values must be finite numbers greater than or equal to zero. For a
quantity whose legacy key contains `velocity`, the scalar is explicitly the
velocity magnitude; its direction is stated in the prompt. Stable parameter
keys are not renamed in V11 because they are public evaluator and Baseline
interfaces. The new non-negative-value rule and prompt direction contract
remove the former signed-scalar interpretation.

The Case-local document uses schema `2.0`:

```json
{
  "schema_version": "2.0",
  "case_id": "<case_id>",
  "scene_id": "<scene_id>",
  "physics": {}
}
```

The Loader accepts historical Dataset/Case schemas 3.0 and 4.0 with three-field
quantities and V10 physics-document schema 1.0. For Dataset/Case schema 5.0 it
requires the four-field quantity contract, physics-document schema 2.0,
non-negative values, exact identity, and deep equality between file physics and
inline `case.physics` even when `check_assets=False`.

## Independent Versus Audit-only Quantities

`annotated=true` means independent, trusted, and conditionable in V11. These
quantities must appear by symbol in the prompt and remain visible to Baseline
adapters. Derived, calibration, auxiliary, or duplicate-alias quantities remain
in `case.physics` for evaluator and audit compatibility, receive symbols, use
`annotated=false`, and do not appear in the Dataset prompt.

V10 flags change only as follows:

| Scene | Parameter | V11 role | Reason |
| --- | --- | --- | --- |
| collision | `striker_initial_velocity` | audit-only | duplicates one numbered ball velocity |
| incline | `calibration_length` | audit-only | apparatus calibration quantity |
| incline | `friction_force` | audit-only | derived from the independent physical inputs |
| incline | `theoretical_acceleration` | audit-only | derived prediction quantity |
| pendulum | `pendulum_length` | audit-only | derived pivot-to-center geometry; string length and bob radius remain primary |

All existing `annotated=false` quantities remain false. No other true field is
demoted. In particular, both measured push-force summaries remain independent
annotations, while the incline's fixed zero `initial_velocity` remains
audit-only and is expressed semantically as “starts from rest.”

Scene `structured_physics_parameters` lists exactly the V11 `annotated=true`
parameter universe. Scene `non_conditionable_physics_parameters` lists every
V11 `annotated=false` parameter universe. This makes Scene, Case, compiler, and
prompt ownership agree.

## Canonical Symbol Table

Every quantity, including audit-only quantities, receives exactly the following
symbol. A symbol must be unique within a Case; reuse across different Scenes is
intentional.

| Scene | Parameter | Symbol | Prompt? |
| --- | --- | --- | --- |
| collision | `ball_1_initial_velocity` | `v_1` | yes |
| collision | `ball_1_mass` | `m_1` | yes |
| collision | `ball_1_radius` | `r_1` | yes |
| collision | `ball_2_initial_velocity` | `v_2` | yes |
| collision | `ball_2_mass` | `m_2` | yes |
| collision | `ball_2_radius` | `r_2` | yes |
| collision | `ball_3_initial_velocity` | `v_3` | when present |
| collision | `ball_3_mass` | `m_3` | when present |
| collision | `ball_3_radius` | `r_3` | when present |
| collision | `striker_initial_velocity` | `v_s` | no |
| incline | `block_length` | `l` | yes |
| incline | `block_mass` | `m` | yes |
| incline | `calibration_length` | `l_cal` | no |
| incline | `friction_force` | `F_f` | no |
| incline | `gravity_acceleration` | `g` | yes |
| incline | `incline_angle` | `θ` | yes |
| incline | `initial_velocity` | `v_0` | no; “starts from rest” is used |
| incline | `kinetic_friction_coefficient` | `μ_k` | yes |
| incline | `theoretical_acceleration` | `a_th` | no |
| projectile | `ball_mass` | `m` | yes |
| projectile | `ball_radius` | `r` | yes |
| projectile | `initial_horizontal_velocity` | `v_0` | yes |
| projectile | `launch_height` | `h` | yes |
| projectile | `photogate_block_time` | `Δt_gate` | no |
| projectile | `photogate_distance_before_launch` | `d_gate` | no |
| projectile | `ramp_angle` | `α` | no |
| projectile | `release_distance` | `s_release` | no |
| pendulum | `bob_mass` | `m` | when present |
| pendulum | `bob_radius` | `r` | yes |
| pendulum | `initial_angle` | `θ_0` | yes |
| pendulum | `pendulum_length` | `l` | no |
| pendulum | `string_length` | `l_s` | yes |
| push bottle | `bottle_height` | `h` | yes |
| push bottle | `bottle_mass` | `m` | yes |
| push bottle | `mean_applied_force` | `F_mean` | yes |
| push bottle | `peak_applied_force` | `F_peak` | yes |
| circular motion | `angular_velocity` | `ω` | yes |
| circular motion | `angular_velocity_rad_s` | `ω_rad` | no |
| circular motion | `object_1_orbit_radius` | `r_1` | yes |
| circular motion | `object_2_orbit_radius` | `r_2` | when present |

## Prompt Contract

All V11 prompts remain English with `text.language="en"`. They use
`text.annotation_source="symbolic_physics_prompt_v1"`. A prompt:

1. describes the visible physical process without ambiguity;
2. contains every and only the Case's `annotated=true` quantity symbols;
3. contains no physical numeric value, unit, formula, background, color,
   camera, viewpoint, laboratory, crop, photogate, resolution, or acquisition
   hint;
4. states motion direction and stationary roles in words;
5. does not claim a direction that cannot be established from existing Case
   facts and the first frame.

The prompt generator is deterministic and Scene-local. It does not inspect
numeric magnitude to invent a direction, except that zero versus nonzero
selects “stationary” versus “moves” after the direction has been established by
the audited collision structure.

Canonical forms are:

### Collision

Balls are numbered left to right in the initial frame. Each ball is introduced
with its `m_i`, `r_i`, and `v_i`. The process clause is selected from audited
`appearance.collision_structure` and striker indices:

- two-ball single incident: left ball stationary; right ball moves left;
- two-ball opposed incident: left ball moves right; right ball moves left;
- three-ball single incident: left ball moves right; middle and right balls are
  stationary; the left ball collides first with the middle ball.

Example form:

```text
The left ball has mass m_1, radius r_1, and initial speed v_1, while the right
ball has mass m_2, radius r_2, and initial speed v_2. The left ball is initially
stationary, the right ball moves left, and they undergo a one-dimensional
central collision.
```

### Inclined plane

```text
A block of mass m and length l starts from rest and slides down an incline of
angle θ. The kinetic friction coefficient between the block and incline is μ_k,
and the gravitational acceleration is g.
```

### Horizontal projectile

```text
A ball of mass m and radius r is launched horizontally to the left with initial
speed v_0 from a vertical height h, then follows a downward parabolic path.
```

### Pendulum

```text
A pendulum bob of radius r is released from rest at initial angle θ_0 on a
string of length l_s, then swings back and forth about the fixed pivot.
```

For the 65 Cases with annotated bob mass, “mass m” is included in the first
clause. It is absent from the other 35 Cases and is not invented.

### Push bottle

```text
An upright bottle of mass m and height h is pushed near its top by a force with
mean magnitude F_mean and peak magnitude F_peak, then tips and falls onto its
side.
```

### Uniform circular motion

One-object Cases state that the object follows an orbit of radius `r_1` with
angular speed `ω`. Two-object Cases mention both `r_1` and `r_2`. Rotation
direction is not invented because V10 contains no audited clockwise or
counterclockwise field and all legacy angular scalars are already non-negative.

## Signed-value Migration

V10 contains negative values only in collision velocity fields:

- 264 `ball_2_initial_velocity` values;
- 230 `striker_initial_velocity` values.

V11 replaces each negative scalar with its absolute value. No zero or positive
value changes. Collision direction is reconstructed from the audited
`collision_structure`, `striker_ball_index`, `opposing_striker_ball_index`, and
left-to-right object numbering, and is frozen in the prompt. The builder must
prove the old sign agrees with that audited direction before converting it; any
contradiction aborts the migration rather than being silently normalized.

All other quantities are asserted non-negative without modification. The V11
Loader rejects any negative quantity in future V11 Cases.

## Release Construction and Evidence

The deterministic builder lives in `scripts/build_dataset_v11.py` and uses
10.0.0 as its sole base. It:

1. validates the exact V10 identity and all 799 Case-local V10 documents;
2. applies the fixed symbol table, flag corrections, signed-value migration,
   and deterministic prompts;
3. writes `physics.v11.json` atomically and refuses conflicting existing bytes;
4. stages a minimal V11 Release with the same seven runtime entries as V10;
5. copies Views unchanged and copies Scenes with only structured versus
   non-conditionable parameter lists updated;
6. rebuilds a 6,038-entry asset lock and verifies every SHA-256;
7. proves all changes are confined to schema identity, prompt, physics,
   `assets.physics_annotation`, Scene parameter classification, and release
   identity;
8. publishes by atomic rename and invokes an independent validator.

Evidence lives outside the runtime Release:

```text
datasets/provenance/releases/11.0.0/
├── migration.json
└── validation.json
```

`migration.json` records every changed physics value with old/new values, every
flag change, every prompt old/new SHA-256, the symbol table digest, unchanged
View digests, asset counts, Dataset digests, and zero media changes.

## Downstream Alignment

`V11_DATASET` becomes `LATEST_DATASET`. Official Tasks change only their Task
IDs and Dataset ID to V11. Selections, Views, Scene lists, seeds, counts, and
evaluation protocol remain unchanged.

Conditionable Case projection continues to expose only `annotated=true`
quantities and now preserves `symbol`. Standard structured-text and quantity
embedding Baselines must consume the corrected independent set. Their numeric
injection remains Baseline-owned: generic Baselines see symbols but no numeric
values in the Dataset prompt, while physics Baselines may append the numeric
values under their existing audited input policy.

Evaluators retain access to the complete inline physics object, including
`annotated=false` derived quantities. They must not treat a false flag as
deletion. Stable parameter keys mean evaluator formulas and entity mapping do
not require key migration. Collision alias validation compares magnitudes in
V11 while retaining historical signed comparison for older Dataset schemas.

All current documentation, ingestion guidance, Case examples, Baseline
resources, quantity registries, maintenance defaults, and current tests move to
V11. Historical Releases, frozen runs, results, provenance, and experiment
reports remain unchanged.

## Failure Rules

- The builder performs every read-only invariant check before writing the first
  `physics.v11.json`.
- Unknown parameters, missing symbol mappings, duplicate symbols within a Case,
  signed-direction contradictions, missing independent symbols in a prompt, or
  unexpected V10 facts abort the whole build.
- Existing candidate files must be byte-identical or the build aborts before
  writing any missing candidate.
- No historical `physics.json`, Release, run, result, media, mask, archive, or
  source document is overwritten.
- A prompt is rejected if it contains a numeric physical value or forbidden
  acquisition/appearance language. Digits that are part of declared symbols
  such as `m_1` are allowed.

## Acceptance Criteria

1. V11 contains 799 Cases and exactly 799 unique `physics.v11.json` assets.
2. Every quantity has exactly `value`, `unit`, `annotated`, and `symbol`.
3. All quantities are finite and non-negative.
4. Every Case has unique symbols, and every annotated symbol occurs in its
   English prompt under the deterministic Scene template.
5. No audit-only symbol or concrete physical value is intentionally injected
   into the Dataset prompt.
6. Exactly 494 legacy negative scalar occurrences are converted to magnitudes,
   with direction preserved in collision prompts.
7. Exactly the five documented parameter families change from annotated to
   audit-only, totaling 715 Case-field flag changes.
8. V11 Views and all 5,239 non-physics locked asset records are byte-identical
   to V10; no media changes.
9. V11 has 6,038 locked assets and passes complete SHA-256 verification.
10. V10 still loads with full hashes and retains its original physics
    fingerprint.
11. V11 is the official Dataset/Task default and the official plan counts
    remain 582 finetune training Cases, 76 finetune jobs, and 658 direct jobs.
12. Migration and independent validation evidence are complete and stable.
