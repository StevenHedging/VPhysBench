# Single Current Physics Annotation Dataset Design

## Goal

Publish one current Dataset whose 799 Cases each contain exactly one Case-local
structured physics document named `physics.json`, using the complete corrected
V11 physics content, while removing all older runtime Releases from the active
checkout and preserving every media byte.

## Verified coverage

Dataset 10.0.0 and 11.0.0 contain the same ordered 799 Case IDs. Across all
5,286 quantities:

- parameter-key sets are identical in every Case;
- units are identical;
- 4,792 values are identical;
- the other 494 values are exactly the documented conversion from a negative
  signed collision velocity to its non-negative magnitude;
- 715 `annotated` flags are intentionally demoted from independent to audit-only;
- every V11 quantity adds one non-empty stable `symbol`;
- no other value change exists.

V11 therefore semantically replaces the V10 physics documents. It is not a
byte-for-byte superset because its direction and independent/audit semantics are
corrected.

## Considered approaches

### Selected: publish Dataset 12.0.0 and retain only its runtime Release

Create `physics_video_six_scene_v12` from V11 Case facts. Replace every
`assets.physics_annotation` path with the Case-root `physics.json`, write the
V11 schema-2 physics document at that path, and delete `physics.v11.json`.
Publish a new digest and make V12 the sole current runtime Release.

This preserves Dataset identity discipline: a published V11 digest is never
silently redefined. It also leaves the final Case directory with one obvious
physics file.

### Rejected: rewrite Dataset 11.0.0 in place

This is mechanically smaller but makes one Dataset ID and release string refer
to two different Case and asset digests. That creates more ambiguity than the
filename simplification removes.

### Rejected: keep both files and add an alias

This preserves history but does not satisfy the requested single-file layout
and retains duplicate metadata indefinitely.

## Final runtime layout

```text
datasets/
├── assets/<scene>/<case>/
│   ├── physics.json
│   ├── source/
│   └── canonical/
├── provenance/
└── releases/
    ├── README.md
    └── 12.0.0/
        ├── README.md
        ├── dataset.json
        ├── release.json
        ├── cases.jsonl
        ├── assets.lock.json
        ├── scenes/
        └── views/
```

There must be exactly 799 tracked `physics.json` files and zero tracked
`physics.v11.json` files. V12 retains 6,038 locked assets. No video, image,
mask, source archive, workbook, or other media byte changes.

## Release retirement

Runtime Release directories 1.0.0 through 11.0.0 are removed from the active
checkout. They remain recoverable from Git history but are not loadable or
discoverable in the current tree. `physbench.data_layout` exposes only V12 as
the current Dataset. Current official Tasks use V12 identities.

Historical build scripts and reports may retain textual historical identities
when they are clearly archival, but no active command, default, test, or
documentation path may depend on a retired Release.

## Construction and validation

A V12 builder consumes the committed V11 metadata before retirement, verifies
the exact 5,286-quantity coverage relation above, stages V12, rewrites the 799
Case-local documents, rebuilds the asset lock, performs a complete Loader hash
check, and only then publishes the Release. It refuses unexpected files or
content differences.

An independent V12 validator does not call migration helpers. It checks:

- V12 identity, schema, Case and Scene counts;
- exactly one `physics.json` per Case and no `physics.v11.json`;
- document schema 2.0 and deep equality with inline `case.physics`;
- non-negative finite values, stable symbols, and prompt-symbol binding;
- exact View coverage and official Task plan counts;
- all 6,038 asset sizes and SHA-256 values;
- absence of changed media relative to the V11 asset records.

## Downstream behavior

Loader semantics remain Dataset schema 5.0. Baselines and evaluators continue
to consume inline `case.physics`; no model adapter changes its numerical input.
Only Dataset identity, Case-local annotation path, asset-lock digest, Task IDs,
and documentation change.

## Acceptance criteria

- one active Release: 12.0.0;
- 799 Cases and six Scenes;
- 799 `physics.json`, zero `physics.v11.json`;
- 6,038 locked assets;
- unchanged media bytes and View membership;
- 582 official training Cases, 76 finetune jobs, 658 direct jobs;
- V12 independent validation and all current Dataset/Task/Baseline gates pass;
- no active reference to a retired Dataset path or ID.
