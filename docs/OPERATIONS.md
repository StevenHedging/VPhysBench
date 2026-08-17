# Operations

## Routine checks

```bash
physbench doctor --level metadata
make test
make smoke-interface
make release-check
```

After downloading the complete Dataset:

```bash
physbench doctor --level evaluation
make data-test
```

The evaluation doctor checks Dataset assets and importability of the optional
scene-evaluation modules. CUDA execution and SAM 2 model retrieval are verified
by the first real one-case run.

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
