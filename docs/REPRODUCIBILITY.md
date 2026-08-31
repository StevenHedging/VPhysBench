# Reproducibility

A reproducible VPhysBench result needs all of the following identities:

1. `git rev-parse HEAD` from the benchmark checkout.
2. Dataset repository and exact 40-character revision from
   `datasets/huggingface.json`.
3. Dataset ID and release from `datasets/releases/14.0.0/dataset.json`.
4. Official Task file and digest.
5. Evaluation protocol ID and component fingerprints.
6. Baseline bundle and deployment digests from `baseline validate`.
7. Model/checkpoint repository revision and file SHA-256, stored outside this
   clean benchmark repository.
8. Python/PyTorch/CUDA package environment and hardware description.
9. The complete `run/<run_id>/frozen/` and `task_instance/` trees.

The bootstrap provides deterministic one-command planning, pins selected direct
distribution inputs, and freezes SAM 2 at
`2b90b9f5ceec907a1c18123530e92e794ad901a4` and SAM 3 at
`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`. It is not a bit-for-bit fully
resolved dependency lock: unlisted transitive packages are resolved at
installation time. Python 3.12 is the CI-validated evaluation interpreter;
newer minors are not certified. PyTorch wheels remain platform-specific.
Capture the resolved environment for every Run with:

```bash
python -m pip freeze > run/<run_id>/artifacts/environment.txt
nvidia-smi > run/<run_id>/artifacts/nvidia-smi.txt
git rev-parse HEAD > run/<run_id>/artifacts/vphysbench-commit.txt
```

Do not put credentials in these files. Before sharing a result, retain the run
manifest, prediction hashes, evaluation manifest, baseline descriptor,
checkpoint hash and immutable external repository revisions.

`RELEASE_MANIFEST.json` records the repository-wide anchors for this date
branch. The Git ref `2026-08-31` (or a publication tag made from it) identifies
the authoritative final release commit. Its `source_commit` field is the
pre-publication source revision: the parent/input revision used when publishing
the release, not an attempt to embed the final self-containing release commit's
own hash. Use the Git ref or publication tag, rather than `source_commit`, when
selecting the released checkout.
