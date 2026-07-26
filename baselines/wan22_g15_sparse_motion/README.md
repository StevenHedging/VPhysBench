# WAN2.2 G15 sparse-motion baseline

This Bundle registers the frozen epoch-20 G15 LoRA as
`wan22_g15_sparse_motion_r32_e20`. It reuses the repository's single WAN2.2
model-family implementation; this directory contains only deployment identity,
configuration and provenance.

This is a schema-v4 managed Bundle. Its one-line `driver.py` selects the shared
`Wan22ManagedDriver`; canonical plan compilation, prompt isolation, TaskInstance
sealing and prediction identity are provided by the common managed runtime.
The WAN media/model lifecycle remains shared with the command-based fine-tuning
Bundle through `Wan22ExecutionEngine`.

The Bundle supports `direct_eval × {generic, physics}` only. It does not
retrain on View A and cannot be used for `finetune_eval`.

## Comparability

G15 was trained before this registration on a five-scene corpus that shares
source trials with benchmark v3. The source-aware audit in
`provenance/benchmark_overlap_v3.json` finds 176/214 seen-source cases. In
particular, the model saw all 121 View A training cases, 8/23 ID test cases and
20/43 OOD1 test cases. Full scores are therefore diagnostic and are not
officially comparable to leakage-free baselines.

The 38 source-unseen cases are the legacy collision, free-fall and pendulum
test cases. A clean-subset report must explicitly name that selection.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored `baseline.local.json`,
then set the epoch-20 checkpoint, WAN pipeline, model root and Python
environment. The checkpoint is verified against the portable SHA-256 before a
task instance is built.

```bash
/root/miniconda3/envs/phybench/bin/physbench \
  baseline validate wan22_g15_sparse_motion_r32_e20
```
