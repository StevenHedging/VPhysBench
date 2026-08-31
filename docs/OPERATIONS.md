# Operations

## Routine checks

```bash
bash scripts/bootstrap_env.sh --profile metadata --dry-run
physbench doctor --level metadata
make test
make smoke-interface
make release-check
make release-archive-check
make portable-release-check
```

`make portable-release-check` extracts a tracked archive under a relocated path
and runs the metadata bootstrap in dry-run mode from outside the checkout. It
does not install SAM3, download Dataset media, or fetch model weights.

For a real evaluation host, bootstrap with Python 3.12+ and the validated
PyTorch 2.10.0/CUDA 12.8 wheel set:

```bash
VPHYSBENCH_BOOTSTRAP_PYTHON=python3.12 \
  bash scripts/bootstrap_env.sh --profile evaluation
. .venv/bin/activate
physbench doctor --level runtime
```

After downloading the complete Dataset and configuring the local checkpoint:

```bash
export VPHYSBENCH_SAM31_CHECKPOINT=SAM31_CHECKPOINT_ABSOLUTE_PATH
physbench doctor --level evaluation
physbench doctor
make data-test
make test-evaluation
```

The default full doctor is the deployment preflight. `metadata` checks the
tracked binding and allows missing Dataset media as a warning; `runtime` checks
executables, evaluator imports, PyTorch/CUDA, and the lightweight evaluator
smoke without Dataset or checkpoint; `evaluation` requires Dataset media and
the SAM2 evaluator; `full` also verifies the SAM3.1 checkpoint digest. Doctor
remains read-only and never installs dependencies, downloads Dataset media or
weights, or loads a model. `physbench doctor --json` emits exactly one
versioned JSON document and exits 0 only when all required checks pass; a
not-ready external-asset host therefore correctly returns exit 1 with
`"ready": false`.

## Release portability boundary

Only a clean tracked checkout is portable. It can be relocated independently
of its original machine, but copied `.venv` directories, package caches,
Dataset media, local Baseline overrides, checkpoints, credentials, and Run
outputs are explicitly outside that contract. Configure those assets anew:
pull Dataset media from the frozen Hub binding, set
`VPHYSBENCH_SAM31_CHECKPOINT` to an absolute local checkpoint path, and place
machine-local Baseline settings in ignored `baseline.local.json` files. Do not
commit any of them.

## Baseline workflow

```bash
physbench baseline list
physbench baseline init my_model --backend managed-i2v
physbench baseline inspect my_model
physbench baseline validate my_model
```

The initial list must be empty. User bundles belong under `baselines/`; local
deployment overrides, credentials and weights are never committed.

## Run workflow

Use `atomic-run` for one baseline and `matrix-run` for multiple identities.
Without `--execute`, the system freezes and plans work but does not launch the
algorithm. Every invocation writes under `run/<run_id>/`.

Use `physbench evaluate --run-dir ... --protocol-id ... --evaluation-id ...`
to create a coexisting reevaluation. It never overwrites canonical evaluation.

## Troubleshooting

- Authentication and Dataset errors: run `hf auth whoami`, then `dataset pull`.
- Contract errors: inspect the frozen job JSON and subprocess log.
- Missing predictions: confirm the command writes exactly to `--output`.
- Media mismatch: compare video probe data with the sealed media contract.
- Evaluator import/load error: install the evaluator extra and rerun doctor.
- Failed run: preserve its logs and manifests; choose a new run ID after fixing.
