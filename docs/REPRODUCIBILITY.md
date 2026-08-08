# Reproducibility

A reproducible VPhysBench result needs all of the following identities:

1. `git rev-parse HEAD` from the benchmark checkout.
2. Dataset repository and exact 40-character revision from
   `datasets/huggingface.json`.
3. Dataset ID and release from `datasets/releases/13.0.0/dataset.json`.
4. Official Task file and digest.
5. Evaluation protocol ID and component fingerprints.
6. Baseline bundle and deployment digests from `baseline validate`.
7. Model/checkpoint repository revision and file SHA-256, stored outside this
   clean benchmark repository.
8. Python/PyTorch/CUDA package environment and hardware description.
9. The complete `run/<run_id>/frozen/` and `task_instance/` trees.

The near-release pins SAM 2 source revision
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. PyTorch wheels remain
platform-specific; record the exact installed versions with:

```bash
python -m pip freeze > run/<run_id>/artifacts/environment.txt
nvidia-smi > run/<run_id>/artifacts/nvidia-smi.txt
git rev-parse HEAD > run/<run_id>/artifacts/vphysbench-commit.txt
```

Do not put credentials in these files. Before sharing a result, retain the run
manifest, prediction hashes, evaluation manifest, baseline descriptor,
checkpoint hash and immutable external repository revisions.

`RELEASE_MANIFEST.json` records the repository-wide anchors for this date
branch. The Git ref `2026-08-08` is the authoritative release commit; embedding
a commit's own hash inside that same commit would be self-referential.
