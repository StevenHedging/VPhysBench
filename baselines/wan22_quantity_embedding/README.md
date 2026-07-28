# WAN2.2 SI-aware quantity-embedding Baseline

Baseline ID:

```text
wan22_ti2v_5b_lora_r32_quantity_embedding_v1
```

This schema-v5 managed Baseline jointly fine-tunes a WAN2.2-TI2V-5B DiT
LoRA and a small quantity encoder. It consumes the same first frame, Case
prompt, and a registry-curated subset of annotated physical
fields. It does not consume every annotation, nor ask UMT5 to infer a number
and unit from ordinary subword tokens.

## Model input

`adapter.py` selects scene-specific fields from `quantity_registry.json`.
It never extracts quantities from arbitrary prose with a regular expression.
For every selected annotation it:

1. validates the field, unit, and finite numeric value;
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
jointly, then generates every frozen ID and OOD1 job:

```bash
cd /root/Steven/physics_video_benchmark
cp baselines/wan22_quantity_embedding/baseline.local.example.json \
  baselines/wan22_quantity_embedding/baseline.local.json
# Edit baseline.local.json for the WAN checkout, model root, Python,
# Accelerate config, and GPUs.

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_quantity_embedding_v1

PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  atomic-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
  --task tasks/official/five_scene_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_quantity_embedding_v1 \
  --run-id wan22_quantity_embedding_v1_viewa_seed42 \
  --output-root runs_v2 \
  --execute
```

Omit `--execute` for a dry-run. `--stop-after-training` is a checkpoint
diagnostic and intentionally produces no benchmark result.

The combined safetensors checkpoints contain both DiT LoRA tensors and
`pipe.quantity_encoder.*` tensors. Important run-local records include:

```text
runs_v2/<run_id>/
├── artifacts/wan22/checkpoints/
├── artifacts/wan22/checkpoint.json
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
  --output-dir results/<run_id>
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

## Paired comparison

For a controlled three-way comparison, run the generic, structured-text, and
quantity-embedding identities on the same frozen View A Task:

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python -m physbench \
  matrix-run \
  --dataset datasets/physics_video/releases/4.0.0/dataset.json \
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

The selected fields match the structured-text comparison arm. The registry
also records whether a selected field is primary or derived; it intentionally
does not imply that every selected field is independent.

The current View A limitations and a no-fabrication result template are
documented in
[`docs/WAN22_QUANTITY_EMBEDDING.md`](../../docs/WAN22_QUANTITY_EMBEDDING.md).
