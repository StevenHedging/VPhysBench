# Dataset Manual Design

## Goal

Provide one authoritative Chinese-language manual that explains the current Dataset and gives an AI assistant or human a complete, repeatable workflow for importing a new Scene delivered as raw videos plus a rough XLSX annotation table.

## Information architecture

`datasets/README.md` is the only normative Dataset manual. It combines the Dataset overview, the atomic Case contract, the four mandatory pre-import questions, the import workflow, stop conditions, and acceptance checks. Historical documentation paths remain as short compatibility pointers so readers cannot accidentally follow stale or conflicting instructions.

The manual describes only the current logical Dataset and Case organization. It does not discuss release identifiers, schema history, migrations, frozen snapshots, hashes, locks, or backward compatibility.

## Content contract

The manual must:

- describe the six current Scenes and their Case counts;
- define a Case as one physical trial with a canonical video, frame-zero image, structured physics, caption, and per-subject first-frame masks;
- show the actual Case-local directory layout and distinguish Case-owned content from provenance and Dataset indexes;
- require four confirmed Scene-level decisions before formal files are written: media window/crop, XLSX field classification, caption semantics/symbols, and mask subjects/order;
- define scalar and time-series physics quantities exactly as the current loader/schema accepts;
- explain `objects.object_N` versus `environment`, and exclude background, color, camera, apparatus, derived summaries, auxiliary measurements, and audit data from formal physics;
- define caption rules, including complete symbol coverage and no numeric values or acquisition hints;
- bind object numbering, physics keys, and mask numbering using the first-frame row-major order;
- require deterministic video/XLSX matching, exclusion reports, duplicate review, visual review, index integration, and machine validation;
- provide a copyable four-question intake template and final acceptance checklist.

## Documentation changes

- Rewrite `datasets/README.md` as the canonical manual.
- Replace `docs/DATASET.md`, `docs/DATASET_INGESTION.md`, and `datasets/DATASET_OVERVIEW.md` with concise links to the canonical manual.
- Update `README.md` and `docs/NEW_SCENE_EVALUATOR.md` links to use the canonical manual.

## Verification

- Search `datasets/README.md` for forbidden release/version-history terminology.
- Parse every fenced JSON example.
- check all modified Markdown links resolve locally;
- run `git diff --check` and review the final diff.
