# WAN2.2 Pair-Text Encoder Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated WAN2.2 ablation that encodes every physical symbol/value pair with frozen UMT5 and preserves the existing cross-attention training and evaluation pipeline.

**Architecture:** Add a pair renderer and frozen full-UMT5 masked-mean encoder behind an explicit conditioner mode. Give the treatment an independent bundle/run identity while reusing the existing registry, WAN LoRA, cross-attention, training loop, generation worker, and evaluator.

**Tech Stack:** Python, PyTorch, UMT5-XXL, WAN2.2-TI2V-5B, DiffSynth-Studio, Accelerate, safetensors, VPhysBench.

## Global Constraints

- Render each record exactly as `<symbol> = <rendered_value> <raw_unit>`; omit the unit only when it equals `1`.
- Encode pairs independently with frozen UMT5 final hidden states and masked mean pooling.
- Preserve one 4096-dimensional physical token per registry record.
- Do not append pair text to the original caption or condition the negative CFG branch.
- Preserve cross-attention dimensions, residual gate, LoRA topology, loss, optimizer, seed, dataset, and evaluator.
- Train eight complete epochs: 2,184 optimizer steps with the current eight-GPU sampling contract.
- Never overwrite the control baseline, checkpoints, run outputs, or existing uncommitted files.

---

### Task 1: Pair rendering and frozen UMT5 encoding

**Files:**
- Modify: `src/physbench/baselines/wan22_symbol_value_model.py`
- Test: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Produces: `render_symbol_value_pair(record) -> str`
- Produces: `encode_symbol_value_pairs(pipe, records) -> (Tensor[m,4096], audits)`
- Consumes: existing validated registry records and WAN UMT5 tokenizer/encoder.

- [ ] Add failing tests asserting exact renderings `m = 0.5156 kg`, `g = 9.80665 m/s^2`, and a dimensionless string without a trailing unit.
- [ ] Run `PYTHONPATH=src:tests:. python -m unittest tests.test_wan22_symbol_value_cross_attention -v` and confirm the new tests fail because the helpers are absent.
- [ ] Implement batched independent tokenization with special tokens, `torch.no_grad()` UMT5 forward, attention-mask pooling, finite/shape validation, and token-ID audits.
- [ ] Add a failing test that any gradient-enabled downstream loss leaves every UMT5 parameter gradient `None`.
- [ ] Run the targeted module and require all tests to pass.

### Task 2: Explicit ablation conditioner mode and checkpoint boundary

**Files:**
- Modify: `src/physbench/baselines/wan22_symbol_value_model.py`
- Modify: `scripts/wan22_symbol_value_train.py`
- Test: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- `SymbolValueConditioner.forward` consumes prebuilt `physics_embeddings` in `umt5_pair_text_mean_v1` mode.
- Existing analytic `symbol_value_cross_attention_v1` behavior and checkpoints remain unchanged.

- [ ] Add failing tests that pair mode has no numeric/dimension/unit trainable branch, retains the same attention/output modules, masks padding exactly, and rejects analytic checkpoint keys.
- [ ] Implement an explicit `physics_token_encoder` config discriminator and a focused downstream conditioner path without changing legacy defaults.
- [ ] Update prompt injection to dispatch to full UMT5 pair encoding only for the new mode and retain text-only negative CFG.
- [ ] Update gradient audits to check downstream conditioner gradients and zero UMT5 gradients.
- [ ] Run targeted tests plus `tests.test_wan22_quantity_embedding` and require zero regressions.

### Task 3: Independent baseline bundle and runtime fingerprints

**Files:**
- Create: `baselines/wan22_pair_text_cross_attention/`
- Modify: `src/physbench/baseline_plugins/wan22_symbol_value.py`
- Modify: `src/physbench/baseline_runtime/drivers/wan22_symbol_value.py`
- Test: `tests/test_wan22_symbol_value_cross_attention.py`

**Interfaces:**
- Produces baseline ID `wan22_ti2v_5b_lora_r32_pair_text_cross_attention_v1`.
- Reuses registry/adapter data semantics but fingerprints pair renderer, model mode, and all shared runtime scripts.

- [ ] Add failing baseline discovery tests for independent identity, encoder mode, eight epochs, and unchanged LoRA/generation settings.
- [ ] Add the bundle, driver, adapter shim, registry copy, and local configuration without modifying the control bundle.
- [ ] Ensure combined checkpoint validation derives the treatment's exact conditioner key set.
- [ ] Run baseline, managed-runtime, conditioning-contract, and repository portability tests.

### Task 4: Smoke train and smoke generation

**Files:**
- Runtime outputs only under a new `run/wan22_pair_text_*` directory.

**Interfaces:**
- Consumes the treatment bundle and existing Dataset 13/model assets.
- Produces one-step checkpoint, gradient audit, one generated video, and token audit.

- [ ] Compile/materialize the existing seven-scene task under the treatment identity and audit counts/fingerprints.
- [ ] Launch one optimizer step on all eight GPUs and require finite loss, finite nonzero downstream-conditioner gradients, and zero UMT5 gradients.
- [ ] Authenticate the smoke checkpoint and generate one representative scored case.
- [ ] Validate its media contract and run the scene evaluator; stop before the full run if any gate fails.

### Task 5: Full 2,184-step training

**Files:**
- Runtime outputs only under the treatment AtomicRun/checkpoint directory.

**Interfaces:**
- Produces epoch-boundary checkpoints at steps 273 through 2,184 plus optimizer/RNG state and audits.

- [ ] Launch eight epochs with seed 42 and the frozen 2,184-row balanced sampling plan.
- [ ] Monitor logs and GPU/process state, preserving artifacts and stopping on non-finite loss, gradient, or distributed failure.
- [ ] Verify final step, epoch, tensor inventory, checkpoint SHA-256, optimizer/scheduler state, and sampling contract.

### Task 6: Generation, VPhysBench evaluation, and comparison

**Files:**
- Runtime outputs under the treatment run.
- Create: `results/wan22_pair_text_cross_attention_ablation_<run-id>.md`

**Interfaces:**
- Produces frozen test predictions, v10 case results, strict aggregate, per-scene breakdown, and a causal/diagnostic comparison report.

- [ ] Generate every frozen test job with seed 42, 50 steps, CFG 5.0, and the authenticated final checkpoint.
- [ ] Reconcile planned jobs, predictions, videos, token audits, and checkpoint fingerprints one-to-one.
- [ ] Run VPhysBench v10 evaluation and record strict score, coverage, per-scene metrics, and all failure/degradation reason counts.
- [ ] Compare with the original symbol/value treatment only on matched jobs and matched training budgets; clearly mark unmatched comparisons diagnostic.
- [ ] Run the full relevant test suite and artifact audit before reporting conclusions.
