# Case-local Physics and Minimal Dataset 10.0.0 Design

## Goal

Publish `physics_video_six_scene_v10` as the official Dataset default. Every
one of its 799 Cases must expose a Case-local structured-physics document in
the asset tree, while preserving every existing physical quantity, value,
unit, annotation-trust flag, Case identity, video, first frame, mask, text,
Scene definition, and View assignment from Dataset 9.0.0.

The 10.0.0 release directory must contain only files required to load,
validate, select, and reproduce the runtime Dataset snapshot. Build tools and
historical evidence belong outside the immutable runtime snapshot.

Remove the 32 unreferenced pre-5.1 collision asset directories after proving
that they are legacy path remnants rather than independent Cases.

## Non-goals

- Do not normalize, rename, regroup, reorder, derive, or otherwise revise any
  structured-physics parameter.
- Do not change video, image, mask, NPZ, source archive, text, appearance,
  temporal, alignment, or provenance facts.
- Do not remove inline `case.physics` in this release.
- Do not introduce Dataset or Case schema 5.0.
- Do not delete or rewrite historical releases 1.0.0 through 9.0.0.
- Do not delete an asset path referenced by Dataset 9.0.0 or any file from a
  current descriptive Case directory.
- Do not copy 9.0.0 build scripts, reports, or legacy audit files into
  10.0.0.

## Case-local Physics Contract

For each Case in 9.0.0, derive the physical Case directory from the parent of
the Case's `canonical/` directory and materialize exactly one document:

```text
datasets/assets/<scene_id>/<physical_case_directory>/physics.json
```

The document is self-identifying:

```json
{
  "schema_version": "1.0",
  "case_id": "<case_id>",
  "scene_id": "<scene_id>",
  "physics": {
    "<parameter>": {
      "value": 0.0,
      "unit": "<unit>",
      "annotated": true
    }
  }
}
```

The `physics` object must be a deep, serialization-stable copy of the 9.0.0
Case's `physics` object. The builder may sort JSON object keys for a stable
encoding, but it must not change array order, parameter names, numeric types or
values, units, or booleans.

Each 10.0.0 Case adds one asset role:

```json
"physics_annotation": "assets/<scene>/<case>/physics.json"
```

Inline `case.physics` remains the existing runtime API for Baselines and
evaluators. `assets.physics_annotation` makes the same annotation a first-class
member of the asset Case and places it under the asset lock. Loading 10.0.0
must reject a missing document, a Case or Scene identity mismatch, or any
difference between the file's `physics` object and inline `case.physics`.

The 32 `collision_r2_*` directories outside the 799 current Case directories
are V5.0 path remnants. Their Case IDs remain active, but V5.1 moved each Case
to one descriptive physical directory and V5.1 through V9 reference only that
new directory. Each remnant contains exactly one unreferenced derived file,
`source/first_frame_source.png`, and no canonical video or source video. They
must not receive a duplicate `physics.json`; they are deleted under the legacy
cleanup contract below. The `source_archives` compatibility link is not a Case
tree and is excluded from this inventory.

Case-local `physics.json` files are source-controlled Dataset metadata even
though they live below the generally ignored media tree. Ignore rules and
staging checks must admit only those files (plus the existing asset README and
compatibility link); no video, image, mask, NPZ, or source archive may become a
new tracked file.

## Dataset 10.0.0 Runtime Snapshot

The release directory contains only:

```text
datasets/releases/10.0.0/
├── README.md
├── dataset.json
├── release.json
├── cases.jsonl
├── assets.lock.json
├── scenes/
│   └── <scene_id>.json
└── views/
    ├── view_a.json
    └── view_b.json
```

The runtime responsibilities are:

| Path | Responsibility |
| --- | --- |
| `dataset.json` | Stable entry point and paths to runtime metadata |
| `release.json` | Dataset identity, Dataset digest, and locked-asset digest |
| `cases.jsonl` | Ordered Case facts and asset references |
| `assets.lock.json` | Size and SHA-256 for every referenced asset |
| `scenes/` | Scene physics and evaluation contracts |
| `views/` | Frozen train/test and direct-evaluation membership |
| `README.md` | Human-readable scope and file responsibility map |

10.0.0 inherits all 5,239 referenced assets from 9.0.0 and adds 799 distinct
Case-local physics documents, for an expected total of 6,038 locked files.

The 9.0.0 `masks.jsonl` index is not copied. No current runtime component reads
it, while every mask used by a Case is already reachable through
`assets.first_frame_mask_manifest` and the per-object PNG/NPZ asset roles.
The descriptor therefore omits `mask_annotations`; mask cardinality, object
identity, ordering, geometry, and physics-key associations remain in each
Case-local mask manifest.

## Build and Evidence Placement

The reproducible builder lives at:

```text
scripts/build_dataset_v10.py
```

It must use 9.0.0 as its sole base, preflight all 799 Case-to-directory
mappings before writing, write Case-local physics documents atomically, build
10.0.0 metadata deterministically, and refuse to overwrite an existing
10.0.0 release unless its identity is the expected V10 identity.

V10-specific evidence lives outside the runtime snapshot:

```text
datasets/provenance/releases/10.0.0/
├── migration.json
└── validation.json
```

`migration.json` records base/output identities and digests, expected and
actual Case/file counts, the unchanged physics fingerprint, and the explicit
list of runtime files. It also records every removed legacy directory and the
relative path, byte size, and SHA-256 of its sole file before deletion.
`validation.json` records the completed invariant checks and final digests.
Historical correction catalogs, directory mappings, split audits,
mask-generation reports, and storage-upgrade reports remain in the historical
releases that produced them and are referenced from the V10 README instead of
being copied.

## Legacy Asset Cleanup Contract

The V10 build performs cleanup only after its read-only preflight has proved
all of the following for an exact, enumerated set of 32 paths:

1. every directory is an immediate child of `datasets/assets/collision_1d`;
2. every name begins with `collision_r2_`;
3. the directory is not the resolved Case directory of any 9.0.0 Case;
4. its corresponding Case ID exists in 9.0.0 and resolves to a different,
   descriptive physical directory;
5. its complete filesystem content is exactly one regular file at
   `source/first_frame_source.png` plus the two required directories;
6. no release Case asset role from 1.0.0 through 9.0.0 references that
   remaining file;
7. the file's byte size and SHA-256 have been captured in the migration
   evidence.

Deletion uses the enumerated absolute paths produced by preflight and never a
recursive wildcard rooted above an individual legacy directory. These ignored
PNG files are not recoverable from Git after deletion; their identity remains
in the V10 migration evidence and their source experiments remain represented
by the active descriptive Case directories and the preserved source archives.

## Loader and Validation

The generic Dataset loader remains compatible with releases before 10.0.0.
It enforces the Case-local physics contract only when a Case declares
`assets.physics_annotation`:

1. resolve the path under `asset_root`;
2. require a regular JSON file;
3. require exactly `schema_version`, `case_id`, `scene_id`, and `physics`;
4. require schema version `1.0`;
5. require Case and Scene identities to match the containing Case;
6. require exact deep equality with inline `case.physics`.

Asset existence and SHA-256 checks remain controlled by `check_assets` and
`check_asset_hashes`, but annotation-file semantic equality is part of Dataset
loading because it defines which structured physics belongs to the Case.

The V10 release validator additionally proves:

- 799 Cases and 799 unique `physics.json` paths;
- every 10.0.0 Case equals its 9.0.0 counterpart after removing only
  `assets.physics_annotation`;
- ordered `(case_id, physics)` fingerprints are identical in 9.0.0 and
  10.0.0;
- all 5,239 pre-existing locked paths retain their size and SHA-256;
- the 799 new paths are the only new locked assets;
- the exact 32 validated legacy directories have been removed and no other
  asset directory has been deleted;
- the release directory contains exactly the seven documented runtime
  components and no build/audit/report payloads;
- a full `check_asset_hashes=True` Dataset load succeeds.

## Official Default Migration

Add `V10_RELEASE_ROOT` and `V10_DATASET` and set `LATEST_DATASET = V10_DATASET`.
The official direct- and finetune-evaluation Tasks change their Task IDs from
the V8 suffix to V10 and bind to `physics_video_six_scene_v10`. Their Scene
selection, View selection, seeds, and evaluation protocol remain unchanged.

The Makefile, root README, Dataset architecture and ingestion documentation,
Task documentation, Baseline integration documentation, active Baseline
READMEs, maintenance-script defaults, and current-version tests must all point
to 10.0.0. Historical experiment records, runs, results, provenance records,
and release-local historical documentation remain unchanged.

## Failure and Recovery Rules

- The builder performs a read-only preflight before creating any
  `physics.json` or release metadata.
- A Case whose canonical assets do not identify exactly one asset Case
  directory aborts the entire build.
- Existing `physics.json` content must either be byte-identical to the
  deterministic candidate or cause the build to abort; the builder never
  silently replaces conflicting annotation content.
- Legacy directories are deleted only after the complete 32-directory cleanup
  preflight and migration evidence generation succeed; any unexpected member
  aborts cleanup before the first deletion.
- Release metadata is built in a temporary sibling directory and published by
  rename only after validation.
- If validation fails, 10.0.0 is not published. Deterministically written,
  byte-identical Case-local documents may remain because they are derived
  assets, but conflicting or partial documents are forbidden by preflight.

## Acceptance Criteria

1. Exactly 799 referenced asset Case directories contain `physics.json`.
2. Every file is referenced as `assets.physics_annotation` by exactly one V10
   Case and is present in the V10 asset lock.
3. V9 and V10 ordered physics fingerprints match exactly.
4. V10 contains exactly 799 Cases and 6,038 locked assets.
5. V10 loads with complete asset SHA-256 verification.
6. V10 is the official default and both official Tasks compile against it.
7. The 10.0.0 directory contains only the documented runtime snapshot.
8. Current documentation consistently describes V10, Case-local physics, and
   the slim release/provenance boundary.
9. Historical releases and non-V10 physical annotation content remain
   unchanged.
10. Exactly 32 validated legacy `collision_r2_*` directories are removed, and
    the migration evidence records the path, size, and SHA-256 of each deleted
    PNG.
