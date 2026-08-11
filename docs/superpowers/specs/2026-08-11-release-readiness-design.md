# VPhysBench v1 Release Readiness Design

## Goal

Restore the `2026-08-11` release gate and make a fresh benchmark checkout able
to obtain the immutable Dataset efficiently, compile either official Task v1,
and run evaluation when the evaluator extra is installed. The Git release must
remain free of registered baselines, model code, model weights, local runtime
configuration, and run outputs.

## Observed failures

GitHub Actions run `31459536891` installed only `.[hub]` and then ran the
release test target. Three test modules failed during import because the public
`physbench.evaluation` and `physbench.orchestration` packages eagerly loaded
`task_evaluator`, which loaded OpenCV. The trailing Node.js 20 message was a
warning and did not cause the failure.

The bound private Dataset revision is reachable and a single file can be
downloaded at the exact commit. A full repository download resolves thousands
of individual files and exceeds Hugging Face's resolver request quota, causing
HTTP 429 retries. The complete local Dataset itself validates, and a one-case
AtomicRun completes with full evaluation coverage, so the data and Task
semantics are sound; the defects are dependency coupling and transport shape.

## Considered approaches

### 1. Install all evaluator dependencies in the release job

This would make `cv2` available and unblock the current import, but a metadata
and release audit would then require PyTorch, SAM 2, SciPy, and OpenCV. It hides
the architectural coupling, makes CI slower and less reliable, and does not
address Dataset download throttling. Rejected.

### 2. Add retries to the existing whole-repository download

This improves transient failure handling but still asks the Hub resolver for
thousands of paths. A fresh download remains quota-sensitive and error output
remains difficult to interpret. Suitable only as compatibility behavior, not
as the primary distribution design. Rejected as the final design.

### 3. Separate capability layers and publish immutable Dataset shards

The interface layer owns metadata, Dataset/Task validation, canonical planning,
Baseline compilation, release auditing, and archive checks. The evaluation
layer is activated only by execution or reevaluation. Dataset assets are
published as a small number of size-bounded archives described by a hashed
manifest at a new immutable Hub revision. Selected.

## Architecture

### Lightweight interface layer

`physbench.evaluation` directly exposes only protocol loading without importing
scientific dependencies. Evaluator entry points remain public but are resolved
lazily when called. `compile_task_instance` and `build_task_instance` move to a
focused lightweight orchestration module. `physbench.orchestration` directly
exports these compiler functions and lazily resolves AtomicRun and reevaluation
entry points.

This establishes the following invariant: importing protocol and Task compiler
APIs in an environment containing only the core package and `hub` extra must
not import `cv2`, `numpy`, `scipy`, `torch`, or `sam2`.

### Evaluation layer

Atomic execution loads the evaluator only when it reaches evaluation. Missing
optional dependencies produce an actionable message naming the
`scene-evaluation` extra. Evaluation algorithms and Evaluation v1 outputs do
not change.

### CI topology

The required `interface-and-release` job installs `.[hub]` and runs a dedicated
`test-interface` target, the unregistered-baseline smoke test, the release
content audit, and archive verification. A separate `evaluation-contract` job
installs `.[scene-evaluation]` and runs evaluator contract tests that use small
fixtures rather than the private 23+ GiB Dataset. `make test` remains a local
convenience alias for the lightweight release suite; `make test-evaluation`
names the heavy suite explicitly.

### Dataset distribution v1

The semantic Dataset remains `physics_video_seven_scene_v13`, release
`13.0.0`, with the same descriptor, cases, scenes, views, and Dataset digest.
Only its transport revision changes.

The Hub revision contains `distribution/v1/manifest.json` and deterministic,
stored ZIP shards under `distribution/v1/shards/`. Each shard is bounded to
approximately 2 GiB, contains only relative regular-file paths under
`assets/`, and records its byte size and SHA-256 in the manifest. The manifest
also records schema version, Dataset identity, release, Dataset digest, total
file and byte counts, and a path-to-shard inventory with per-file size and
SHA-256. The exact Hub commit in `datasets/huggingface.json` authenticates the
manifest; hashes authenticate downloaded bytes.

The downloader performs these phases:

1. Validate the local immutable binding and existing Dataset metadata.
2. Download only the distribution manifest at the bound commit.
3. Validate its schema and Dataset identity before allocating extraction work.
4. Download missing shards into a revision-specific staging/cache directory;
   existing valid shards are reused, and failed downloads preserve resumable
   state.
5. Verify every shard hash and reject absolute paths, parent traversal,
   symlinks, duplicate members, and undeclared files.
6. Extract to staging, verify every extracted file against the manifest, then
   promote files into `datasets/assets/` using per-file atomic replacement.
7. Run the existing Dataset loader with asset checking and report the frozen
   descriptor path.

The CLI streams download progress and preserves stderr. Errors distinguish
authentication/authorization, rate limiting, disk exhaustion, checksum
mismatch, unsafe archive content, and final Dataset validation failure.

## Compatibility and release policy

The new downloader prefers distribution v1. The immutable Git binding is
updated only after the uploaded Hub revision has been independently validated.
No floating Hub branch is used. Dataset semantic version and official Task v1
IDs do not change because the case set, split, seeds, and evaluation protocol
do not change.

The old Hub revision remains addressable for reproducibility. The new revision
may remove the old leaf assets from its tree after the shard release is proven;
historical commits remain immutable. GitHub release archives contain only the
small binding and metadata, never the Dataset shards.

## Error handling and safety

- All paths from remote manifests and archives are normalized and constrained
  beneath their declared roots.
- Shards and extracted files are verified before promotion.
- Interrupted work is resumable and cannot turn the active Dataset into a
  partially replaced tree.
- Credentials stay in the user-level Hugging Face configuration and never enter
  Git files, logs, manifests, or command arguments.
- A failed optional evaluator import names the exact installation extra rather
  than surfacing a raw transitive `ModuleNotFoundError` without context.

## Verification and acceptance criteria

- A fresh Python 3.11 environment passes `pip install -e ".[hub]"` followed by
  `make test-interface`, `make smoke-interface`, `make release-check`, and
  `make release-archive-check`.
- The lightweight public imports succeed while heavy modules are deliberately
  blocked, and none appear in `sys.modules`.
- The full evaluator environment passes `make test-evaluation`.
- GitHub Actions reports green required release checks on the pushed commit.
- From an empty asset directory, `physbench dataset pull` downloads the exact
  bound revision without resolving thousands of leaf files, survives an
  interrupted retry, and finishes with full asset validation.
- The resulting Dataset digest remains
  `a83becec3ced3ce7584611e68e27e7d2c7770fa7b7d653cd3b7ba5ffb499f289`.
- The direct Task plan contains 658 evaluation cases across five scored scenes.
- The finetune Task plan contains 806 training cases across seven scenes and 76
  held-out evaluation cases across five scored scenes.
- A real one-case AtomicRun reaches `status=complete`, evaluation coverage 1.0,
  and a finite Evaluation v1 score.
