# WAN2.2 SI-aware quantity-embedding Baseline

Baseline ID:

```text
wan22_ti2v_5b_lora_r32_quantity_embedding_v1
```

Bundle version: `1.0.1`.

The completed 2026-07-28 source experiment is a historical AtomicRun whose
bundle `1.0.0` and baseline digest are sealed. The repository HEAD at execution
was `918e9f7`; that commit is known provenance but is not separately sealed in
the run manifest. The current commands create the current `1.0.1` Baseline with
the official schema-4 Task and current protocol identity. Its full training evidence, canonical v1 result,
alternate v2 reevaluation, and all 66 per-Case outcomes are recorded in
[`docs/experiments/WAN22_QUANTITY_EMBEDDING_20260728.md`](../../docs/experiments/WAN22_QUANTITY_EMBEDDING_20260728.md).

This schema-v5 managed Baseline jointly fine-tunes a WAN2.2-TI2V-5B DiT
LoRA and a small quantity encoder. It consumes the same first frame, Case
prompt, and a registry-curated subset of formal physical
fields. It does not consume every annotation, nor ask UMT5 to infer a number
and unit from ordinary subword tokens.

## Model input

`adapter.py` selects scene-specific fields from `quantity_registry_v2.json`.
It never extracts quantities from arbitrary prose with a regular expression.
For every selected annotation it:

1. validates the field, unit, stable symbol, and finite non-negative value;
2. renders an `audited_prompt` containing the literal quantity;
3. converts the value to SI and records its seven-dimensional SI exponent;
4. replaces that literal in the model prompt with one unique native T5
   sentinel, such as `<extra_id_0>`.

Frozen UMT5 encodes the sentinel-bearing prompt. Immediately after UMT5, the
sentinel context slot is replaced by a learned 4096-dimensional vector:

```text
SI value -> 8 fixed numeric features -> numeric MLP (e_num)
SI exponent [L,M,T,I,Theta,N,J] -> dimension MLP (e_dimension)
semantic quantity type -> learned embedding (e_type)
concat(e_num, e_dimension, e_type) -> fusion MLP -> z_phys
```

The semantic type is necessary because SI dimensions alone cannot distinguish,
for example, an angle from a dimensionless coefficient or a radius from another
length. Injection after frozen UMT5 gives every quantity exactly one audited
token slot and avoids retaining the full UMT5 backward graph. The surrounding
text remains contextualized by UMT5; WAN DiT cross-attention receives that
context together with `z_phys`.

## Training

This Baseline currently supports the `finetune_eval` family only. The official
multi-scene experiment is View A fine-tune/eval: it trains on all five scenes
jointly, then generates every frozen ID-test job; the current Dataset no longer
defines OOD or mixed test subsets:

```bash
cd /root/Steven/physics_video_benchmark
test -e baselines/wan22_quantity_embedding/baseline.local.json || \
  cp baselines/wan22_quantity_embedding/baseline.local.example.json \
    baselines/wan22_quantity_embedding/baseline.local.json
# Edit baseline.local.json for the WAN checkout, model root, Python,
# Accelerate config, and GPUs.

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_quantity_embedding_v1

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --run-id wan22_quantity_embedding_v1_viewa_seed42 \
  --output-root runs_v2 \
  --execute
```

Omit `--execute` for a dry-run. `--stop-after-training` is a checkpoint
diagnostic and intentionally produces no benchmark result.

The combined safetensors checkpoints contain both DiT LoRA tensors and
`pipe.quantity_encoder.*` tensors. Loading is fail-closed against the frozen
WAN2.2-TI2V-5B topology: exactly 300 rank-32 A/B pairs (600 LoRA tensors,
all 30 blocks × 10 targets) and exactly 19 QuantityEncoder tensors. Before
inference, the checkpoint path, byte size, and SHA-256 must match the
run-local schema-2 checkpoint manifest. At the actual load boundary the worker
opens the non-symlink checkpoint with `O_NOFOLLOW` where the platform provides
it, reads the complete file from that one descriptor, hashes those bytes, and
passes the same byte object to `safetensors.torch.load`. It never verifies one
path read and then reopens the path for parsing, so a transient
swap-and-restore cannot substitute different weights. This hash is an internal
consistency check for a frozen AtomicRun, not an external authenticity
signature: a publisher who can rewrite both the checkpoint and its run-local
manifest can create a new self-consistent pair.

This descriptor-bound load deliberately allocates one complete authenticated
checkpoint byte buffer. The observed official rank-32 checkpoint is
175,649,752 bytes (167.5 MiB). `safetensors.torch.load(bytes)` briefly
materializes independent tensor backing while that input buffer still exists;
an actual no-GPU parse measured about 304 MiB incremental peak RSS per worker
(about 2.38 GiB if eight workers peak simultaneously), rather than only the
167.5 MiB input-buffer size. The input buffer is explicitly dropped as soon as
parsing returns, and all temporary CPU state is released after validation and
LoRA fusion; video-generation memory is otherwise unchanged.

For new bundle `1.0.1` runs, the shuffled training DataLoader owns an explicit
`torch.Generator` seeded from the trainer seed. The same sampler seed is recorded in
`training_sampling_plan.json`, `checkpoints/training_args.json`,
`checkpoints/run.env`, and `checkpoints/training_sampling_runtime.json`.
Training fails before its first step unless the repeated Dataset length is
divisible by the distributed world size, so Accelerate cannot pad an epoch
with duplicate samples. RNG sidecars include the sampler generator state, but
remain diagnostic snapshots rather than exact-resume checkpoints because the
live DataLoader iterator/permutation position is not captured.
The historical `1.0.0` source run has a sampling plan, optimizer/scheduler
state, and per-rank RNG sidecars, but no `training_sampling_runtime.json` or
sampler generator state. It passes the versioned legacy acceptance profile and
must not be described as carrying the newer runtime evidence.
Important run-local records include:

```text
runs_v2/<run_id>/
├── artifacts/wan22/checkpoints/
├── artifacts/wan22/checkpoint.json
├── artifacts/wan22/checkpoints/training_sampling_runtime.json
├── artifacts/wan22/training_quantity_token_audit.jsonl
├── artifacts/wan22/inference_quantity_token_audits/
├── artifacts/wan22/training_sampling_plan.json
├── training/training_stage.json
├── predictions/
└── evaluation/task_result.json
```

There is one authoritative publication gate:
`reporting_status.benchmark_score_publishable`. It is fail-closed and becomes
true only when the run and official Task result are complete, the score has full
coverage, every planned prediction is complete, every evaluated Case refers to
a complete prediction, the official partition/scene/Task aggregates reproduce
exactly from validated Case results, the evaluation-protocol fingerprint
matches the frozen AtomicRun, the sealed TaskInstance and prediction artifacts
verify, and all checkpoint/training/token audits pass.

Create deterministic scene/partition statistics, per-job records, official
macro-result projections, and training-evidence indexes from a terminal
`complete`, `inference_incomplete`, or `failed` AtomicRun with:

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \
  scripts/summarize_quantity_run.py \
  --run-dir runs_v2/<run_id> \
  --output-dir results/<run_id>/<summary_id>
```

Missing, failed, and evaluator-unavailable cases remain explicit; the summary
does not impute them as zero. Descriptive tables expose both the evaluated-job
micro mean and a Case-macro mean computed after averaging available inference
seeds within each Case. Official partition/scene/Task macro results are
independently recomputed with the Benchmark Task aggregator and compared with
`evaluation/task_result.json`. Per-job output preserves evaluator metrics,
quality, artifacts (including IoU curves), and provenance.

A partial or failed run remains reportable but can never publish a strict
Benchmark score. A reevaluation whose protocol fingerprint differs from the
frozen AtomicRun is explicitly rejected by the publication gate and must be
reported as a separately identified alternate-protocol result. Missing or
rejected checkpoint, recovery-state, loss, gradient, training-token, or
inference-token evidence is recorded as an integrity issue.

The report reads the checkpoint once through an `O_NOFOLLOW` descriptor, hashes
that immutable buffer, and computes its header/layout/finite/inventory evidence
from the same bytes before rechecking descriptor and path identity. Frozen
Baseline `1.0.0` runs may carry the original five-field manifest;
the recomputed pair/rank/topology/layout/finite fields are then reported as
`derived_not_declared`. Baseline `1.0.1` requires all ten hardened fields in the
manifest. A profile is selected only when `run.json`, `frozen/baseline.json`,
and the sealed TaskInstance agree on Baseline ID and version. Unknown versions,
identity drift, missing required fields, or any declared/recomputed mismatch
fail closed.

The same version gate applies to prediction identity: historical `1.0.0`
records may omit a redundant `scene_id` only when the sealed job resolves it
unambiguously, while `1.0.1` emitters always write it and the summary requires
an exact match. Unknown or inconsistent versions never inherit the legacy
exception.

## Paired comparison

For a controlled three-way comparison, run the generic, structured-text, and
quantity-embedding identities on the same frozen View A Task:

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/releases/12.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_v3_generic \
  --baseline wan22_ti2v_5b_lora_r32_v3_physics \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --matrix-id wan22_viewa_conditioning_seed42 \
  --output-root runs_v2 \
  --execute
```

Each identity is trained independently. Keep the Dataset digest, Task, seed,
WAN base model, LoRA rank/targets, optimizer, scene balancing, generation
recipe, and evaluator fixed. The registry selects the same curated
physical-field subset and precision as the structured-text Baseline. The
quantity Baseline necessarily adds trainable encoder parameters;
report its parameter inventory rather than claiming an exactly
parameter-matched ablation.

The selected fields match the structured-text comparison arm. The V2 registry
selects only current independent quantities retained as formal quantities; derived,
calibration, auxiliary and duplicate-alias fields remain evaluator/audit data
and are never injected. Each selected audit record preserves value, unit and
symbol.

The current View A design, limitations, and result summary are documented in
[`docs/WAN22_QUANTITY_EMBEDDING.md`](../../docs/WAN22_QUANTITY_EMBEDDING.md).
The completed experiment is single-seed and partial: generic and structured
text controls were not run, and the strict Task score is `null`.
