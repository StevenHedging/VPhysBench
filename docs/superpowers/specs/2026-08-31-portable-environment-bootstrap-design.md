# Portable environment bootstrap design

## Goal

Make a clean VPhysBench checkout self-locating and provide deterministic
one-command installation planning with pinned direct inputs and frozen SAM
source revisions, while keeping Dataset media, evaluator checkpoints,
credentials, Baseline implementations, and machine-local paths outside the
tracked benchmark release.

## Portability contract

The portable unit is the tracked checkout, not a copied virtual environment,
local Baseline override, cache, Dataset staging tree, or Run. Repository-owned
paths are derived from the executing file or an explicit project-root option.
External assets remain explicit: Dataset media is installed by the frozen Hub
binding, SAM3.1 uses `VPHYSBENCH_SAM31_CHECKPOINT`, and custom Baseline model
paths remain in ignored `baseline.local.json` files.

The metadata profile supports Python 3.11 or newer. The validated evaluation
interpreter is Python 3.12, with PyTorch 2.10.0 CUDA 12.8 wheels and the already
frozen SAM2/SAM3 source revisions. The launcher may accept newer Python minors,
but CI does not certify them. NVIDIA driver compatibility, network access, and
sufficient storage remain host prerequisites and must be diagnosed rather than
silently guessed.

## Bootstrap interface

`bash scripts/bootstrap_env.sh --profile metadata|evaluation` is the sole
pre-install entry point. The shell wrapper derives the checkout root from
`BASH_SOURCE`, selects a compatible interpreter (or honors
`VPHYSBENCH_BOOTSTRAP_PYTHON`), and invokes a standard-library-only Python
bootstrap module from `src/`.

The Python bootstrap creates or reuses `.venv`, installs the selected pinned
direct constraints and editable extras, then runs the matching doctor level.
It is idempotent and refuses to reuse an incompatible virtual environment. A
`--dry-run` mode emits the exact plan without creating files or accessing the
network. Transitive packages are resolved at installation time and captured
per run; this is not a bit-for-bit fully resolved dependency lock. It never
downloads Dataset assets or evaluator checkpoints.

## Verification interface

`physbench doctor` remains the only readiness authority. Existing metadata and
evaluation levels remain compatible; a new `runtime` level verifies system
executables, evaluator imports, PyTorch/CUDA, and a lightweight evaluator smoke
without requiring Dataset assets or a SAM3.1 checkpoint. The default `full`
level additionally authenticates the Dataset and SAM3.1 checkpoint.

Human output includes fixes. JSON output remains a single versioned document,
and any required failure returns nonzero. Dependency checks perform real
imports, not package-name discovery alone.

## Relocation certification

The release archive test extracts tracked files beneath a path containing
spaces, executes from outside the checkout, and verifies CLI help, metadata
doctor, Baseline discovery, and the interface smoke. Publication scanning
rejects common POSIX machine roots and Windows drive paths in active tracked
content. Tests intentionally ignore machine-local files because those files
are not part of the release artifact.

## Release boundary

The `2026-08-31` branch contains only benchmark runtime, evaluator dependency
profiles, tests, and documentation. It contains no Baseline registration,
training code, generated media, model weights, credentials, local overrides,
cache, or Run output.
