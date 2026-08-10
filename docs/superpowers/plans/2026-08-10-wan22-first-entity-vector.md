# WAN2.2 Physics-Text + First-Entity Vector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register, train for 2184 steps, generate the frozen View-A test predictions, and evaluate a WAN2.2 baseline conditioned on first frame, structured-physics-enhanced text, and one three-layer-MLP embedding of `[mass, size, initial_velocity]` from `object_1`.

**Architecture:** A custom data adapter wraps the existing structured-text adapter and adds one audited entity-vector record plus a native UMT5 sentinel. The existing WAN quantity-token runtime gains a backward-compatible entity-vector encoder mode that replaces that sentinel after frozen UMT5 encoding, then jointly trains the encoder and rank-32 DiT LoRA. The AtomicRun uses all seven scenes for training/inference and protocol v14 for expert+CSTI evaluation on the five supported scenes.

**Tech Stack:** Python 3.10 runtime, PyTorch, DiffSynth-Studio at `fb337fbb90945ff829de69dbd44ded618f73e889`, Accelerate on 8 GPUs, safetensors, unittest/pytest, VPhysBench AtomicRun v2.

## Global Constraints

- Work directly in `/root/Steven/VPhysBench`; do not create a linked worktree.
- Preserve `baselines/wan22_symbol_value_cross_attention/baseline.local.json` and all unrelated user files.
- Use Dataset 13.0.0 View A, seed 42, seven training scenes, 8 epochs, and the exact 2184-step control schedule.
- Select only `physics.objects.object_1`; never aggregate or fall back to a later object.
- The model vector has exactly three SI scalars ordered `[mass, size, initial_velocity]`.
- The entity encoder has exactly three linear layers `3->256->1024->4096` with SiLU after layers one and two and LayerNorm after layer three.
- The vector condition is positive-CFG-only; UMT5, VAE, and base DiT weights remain frozen.
- Existing quantity-embedding behavior and its 19-tensor checkpoint contract must remain unchanged.
- Never overwrite an existing run directory.

---

### Task 1: First-entity vector extraction and adapter

**Files:**
- Create: `baselines/wan22_entity_vector/__init__.py`
- Create: `baselines/wan22_entity_vector/adapter.py`
- Create: `tests/test_wan22_entity_vector.py`

**Interfaces:**
- Produces: `extract_first_entity_vector(case: dict) -> dict`
- Produces: `FirstEntityVectorDataAdapter.adapt_case(case: dict, role: str) -> dict`
- The record contains `values`, `components`, `sentinel`, `source_object`, and derivation/imputation provenance.

- [ ] **Step 1: Write failing extraction tests**

Add tests that load representative Dataset cases and assert:

```python
assert record["values"] == [0.03313, 0.01, 0.6004202942059444]
assert record["source_object"] == "object_1"
assert record["components"][1]["source_field"] == "radius"
```

Cover pendulum zero velocity, incline length, parabolic horizontal velocity, bottle height, spring radius, collision first-object-only selection, circular `math.radians(omega) * orbit_radius`, missing mass imputation, and invalid units/non-finite values.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py -k extraction
```

Expected: import failure because `baselines/wan22_entity_vector/adapter.py` does not exist.

- [ ] **Step 3: Implement strict SI extraction**

Implement field precedence and validation from the design. Return one record shaped as:

```python
{
    "schema_version": "1.0",
    "representation": "first_entity_vector_mlp_v1",
    "source_object": "object_1",
    "values": [mass_kg, size_m, velocity_m_per_s],
    "components": [mass_audit, size_audit, velocity_audit],
    "sentinel": "<extra_id_0>",
}
```

- [ ] **Step 4: Write and verify failing adapter tests**

Assert the adapter reuses `seven_scene_physics_text_v1`, seals both audited/model prompts, emits exactly one vector physics channel, preserves the first-frame media contract, and contains exactly one `<extra_id_0>`.

- [ ] **Step 5: Implement the custom adapter**

Wrap `StandardDataAdapter` with an internal structured-text-only policy, then append the audited literal vector/model sentinel clause and bind `native_inputs.physics.entity_vector`.

- [ ] **Step 6: Verify GREEN and commit**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py
git add baselines/wan22_entity_vector tests/test_wan22_entity_vector.py
git commit -m "feat: add first-entity vector adapter"
```

### Task 2: Three-layer entity encoder in the shared WAN token transport

**Files:**
- Modify: `src/physbench/baselines/wan22_quantity_model.py`
- Modify: `tests/test_wan22_entity_vector.py`
- Test: `tests/test_wan22_quantity_embedding.py`

**Interfaces:**
- `QuantityEncoder(config)` accepts `encoder_type="first_entity_vector_mlp_v1"` while retaining the existing default.
- `QuantityEncoder.encode_records(records)` returns shape `[1, 4096]` for one entity-vector record.
- `quantity_encoder_state_keys(encoder) -> frozenset[str]` derives the sealed topology.

- [ ] **Step 1: Write failing topology and forward tests**

Assert exactly three `nn.Linear` modules with shapes `(256,3)`, `(1024,256)`, `(4096,1024)`, finite `[1,4096]` output, strict single-record/three-value validation, and unchanged default quantity topology.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py -k encoder
```

Expected: the existing encoder rejects the new configuration.

- [ ] **Step 3: Implement encoder mode and dynamic state validation**

Create `self.entity_mlp = nn.Sequential(nn.Linear(3,256), nn.SiLU(), nn.Linear(256,1024), nn.SiLU(), nn.Linear(1024,4096), nn.LayerNorm(4096))` only in the new mode. Make checkpoint loaders validate against the instantiated encoder's exact state keys/count rather than the historical constants; retain constant-based assertions for default mode.

- [ ] **Step 4: Verify GREEN and quantity regression**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py \
  tests/test_wan22_quantity_embedding.py
```

- [ ] **Step 5: Commit**

```bash
git add src/physbench/baselines/wan22_quantity_model.py \
  tests/test_wan22_entity_vector.py tests/test_wan22_quantity_embedding.py
git commit -m "feat: project first-entity vectors into WAN context"
```

### Task 3: Dynamic checkpoint inventory and runtime identity

**Files:**
- Modify: `src/physbench/baselines/wan22_quantity.py`
- Modify: `scripts/wan22_quantity_train.py`
- Modify: `tests/test_wan22_entity_vector.py`
- Test: `tests/test_wan22_quantity_embedding.py`

**Interfaces:**
- `Wan22QuantityLoraAdapter` reads `conditioning_adapter_id` with default `wan22_quantity_embedding`.
- `_checkpoint_inventory_bytes(..., expected_encoder_keys=..., expected_encoder_tensor_count=...)` validates either frozen topology.
- Training audit reports `encoder_type` and entity component order.

- [ ] **Step 1: Write failing inventory and metadata tests**

Create a small combined safetensors fixture and assert the entity topology is accepted only when the new expected state keys are supplied; assert the old 19-tensor fixture is unchanged.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py -k 'checkpoint or metadata'
```

- [ ] **Step 3: Parameterize inventory and audit metadata**

Replace hard-coded entity-sensitive counts with expected values derived from `QuantityEncoder(self.config["quantity_encoder"])`; retain WAN LoRA's exact 600-tensor/rank-32 checks.

- [ ] **Step 4: Verify GREEN and commit**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py \
  tests/test_wan22_quantity_embedding.py
git add src/physbench/baselines/wan22_quantity.py \
  scripts/wan22_quantity_train.py tests
git commit -m "feat: seal entity-vector WAN checkpoints"
```

### Task 4: Register the baseline and frozen experiment task

**Files:**
- Create: `baselines/wan22_entity_vector/.gitignore`
- Create: `baselines/wan22_entity_vector/baseline.json`
- Create: `baselines/wan22_entity_vector/baseline.local.example.json`
- Create locally, untracked: `baselines/wan22_entity_vector/baseline.local.json`
- Create: `baselines/wan22_entity_vector/driver.py`
- Create: `baselines/wan22_entity_vector/README.md`
- Create: `tasks/experiments/seven_scene_entity_vector_finetune_eval.json`
- Modify: `tests/test_wan22_entity_vector.py`

**Interfaces:**
- Baseline ID: `wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1`
- Task ID: `seven_scene_entity_vector_finetune_eval_v14`
- Driver reuses `Wan22QuantityManagedDriver` and the shared WAN quantity execution engine.

- [ ] **Step 1: Write failing bundle discovery/validation/task tests**

Assert exact model, adapter, trainer, runner, seven-scene, seed, epoch, save-step, and protocol identities. Compile the task and assert 806 training cases, 110 inference jobs, and seed 42. The world-aligned scene balancer expands those 806 unique cases to 2,184 rows per epoch, which produces 273 optimizer steps per epoch across eight ranks and 2,184 total steps over eight epochs.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py -k 'bundle or task'
```

- [ ] **Step 3: Add manifests and local deployment**

Use the runtime paths from `wan22_physics_text_lora_2184_v1`:

```json
{
  "runtime": {
    "project_root": "/root/Steven/wan22_pair_text_runtime",
    "python": "/root/Steven/.venvs/wan22-pair-text/bin/python",
    "model_base": "/root/Steven/wan22_pair_text_runtime/models",
    "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "accelerate_config": "/root/Steven/wan22_pair_text_runtime/configs/accelerate/lora_8gpu.yaml"
  }
}
```

Set `encoder_type=first_entity_vector_mlp_v1`, `input_size=3`, hidden sizes 256/1024, output 4096, and the exact 2184-step control configuration.

- [ ] **Step 4: Validate and dry-run**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench \
  baseline validate wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench \
  atomic-run --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/experiments/seven_scene_entity_vector_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1 \
  --output-root run --run-id wan22_physics_text_entity_vector_mlp_2184_v1_dryrun
```

- [ ] **Step 5: Verify and commit tracked files**

```bash
PYTHONPATH=src:tests:. /root/Steven/.venvs/wan22-pair-text/bin/python \
  -m pytest -q tests/test_wan22_entity_vector.py \
  tests/test_integrated_baselines.py tests/test_conditioning_contracts.py
git add baselines/wan22_entity_vector tasks/experiments \
  tests/test_wan22_entity_vector.py
git commit -m "feat: register WAN entity-vector baseline"
```

### Task 5: Execute 2184-step training and generation

**Runtime output:**
- Create: `run/wan22_physics_text_entity_vector_mlp_2184_v1/`

- [ ] **Step 1: Launch the AtomicRun**

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench \
  atomic-run --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/experiments/seven_scene_entity_vector_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1 \
  --output-root run --run-id wan22_physics_text_entity_vector_mlp_2184_v1 \
  --execute
```

- [ ] **Step 2: Treat the first completed optimizer step as the GPU smoke gate**

Confirm finite loss, finite entity-encoder gradient norm, all eight ranks active, and no unused trainable parameter. If it fails, use systematic debugging and resume only after the failing condition has a regression test.

- [ ] **Step 3: Monitor to final checkpoint**

Require `step-2184.safetensors`, optimizer/scheduler state, checkpoint SHA-256, exact LoRA topology, and exact entity-encoder topology.

- [ ] **Step 4: Verify prediction coverage**

Require 110 unique prediction records/files with exact job, case, seed, baseline and checkpoint identities.

### Task 6: Evaluate expert and CSTI dimensions

**Runtime output:**
- Verify: `run/wan22_physics_text_entity_vector_mlp_2184_v1/evaluation/`
- Optionally create: `run/wan22_physics_text_entity_vector_mlp_2184_v1/reevaluations/csti_v14_final/`

- [ ] **Step 1: Run or refresh a coexisting v14 evaluation**

If the AtomicRun's canonical evaluation is not already v14:

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python -m physbench \
  evaluate --run-dir run/wan22_physics_text_entity_vector_mlp_2184_v1 \
  --protocol-id scene_default_v14 --evaluation-id csti_v14_final
```

- [ ] **Step 2: Verify evaluation integrity**

Require complete prediction coverage, explicit unsupported records for push-bottle and vertical-spring, expert and CSTI dimensions for the five supported scenes, no internal evaluator errors, and unchanged canonical predictions.

- [ ] **Step 3: Compare against existing controls**

Report per-scene expert score, CSTI score, coverage, unavailable/error counts, checkpoint identity, and differences against `wan22_physics_text_lora_2184_v1`, quantity embedding, and pair-text controls without mixing protocol identities.

- [ ] **Step 4: Final verification**

Run the new focused suite, existing WAN quantity suite, CSTI protocol tests, `git diff --check`, and inspect `git status` so only the intentionally untracked local deployment files and run outputs remain.
