# WAN2.2 Physics-Text + First-Entity Vector Baseline Design

## Objective

Add and run a WAN2.2-TI2V-5B fine-tuning baseline that conditions generation on three inputs:

1. the first frame;
2. the canonical caption followed by the same deterministic structured-physics text used by `wan22_physics_text_lora_2184_v1`;
3. one trainable embedding of the first physical entity vector `[mass, size, initial_velocity]`.

The experiment must remain directly comparable to the prior 2184-step WAN2.2 controls and must be evaluated with the current CSTI protocol.

## Identity and experiment scope

- Bundle directory: `baselines/wan22_entity_vector/`
- Baseline ID: `wan22_ti2v_5b_lora_r32_physics_text_entity_vector_v1`
- Representation ID: `first_entity_vector_mlp_v1`
- Run ID: `wan22_physics_text_entity_vector_mlp_2184_v1`
- Training Dataset/View: Dataset 13.0.0, View A
- Training scenes: all seven Dataset scenes
- Evaluation protocol: `scene_default_v14`; its five supported scenes report expert and CSTI dimensions, while push-bottle and vertical-spring jobs remain explicitly unsupported by the evaluator.

## First-entity vector contract

The adapter selects `physics.objects.object_1`. Selection is semantic and deterministic; it never selects another object when `object_1` exists and never aggregates multiple entities.

The vector is expressed in canonical SI units and ordered as:

```text
e = [mass_kg, size_m, initial_velocity_m_per_s]
```

Field resolution is:

- mass: `object_1.mass`;
- size: first present field in `radius`, `length`, `height`, `orbit_radius`;
- velocity: first present field in `initial_velocity`, `initial_horizontal_velocity`;
- circular-motion fallback: derive tangential speed as `r * omega`, converting annotated `deg/s` angular velocity to radians per second;
- other absent velocity: `0.0`, because the applicable Dataset cases are released or initialized from rest;
- absent mass or size: `0.0`, with an explicit `imputed_zero` provenance marker.

An absent field is distinct in the audit record even though the model vector is constrained to exactly three scalars. A present field with a non-finite value, Boolean value, unexpected unit, or negative mass/size/velocity magnitude fails adaptation. Collision velocity uses the Dataset's non-negative magnitude; direction remains in the enhanced caption, matching Dataset 13 semantics.

## Text and vector conditioning

The adapter first uses `seven_scene_physics_text_v1` to append all annotated physics clauses to the canonical caption. It then appends one clause containing the native UMT5 sentinel `<extra_id_0>`.

Two prompt forms are sealed:

- audited prompt: contains the literal SI vector;
- model prompt: contains `<extra_id_0>` in the same character position.

WAN's frozen UMT5 encoder processes the complete enhanced model prompt. After text encoding, one trainable entity-vector encoder replaces the sentinel context slot with the projected vector. The encoder is exactly three learned linear layers:

```text
3 -> 256 -> 1024 -> 4096
```

SiLU follows the first and second linear layers; LayerNorm follows the third. No Dataset-level statistics, learned missing-value tokens, object-index embeddings, or additional attention block are added.

The vector is bound only to the positive classifier-free-guidance branch. The negative branch receives only the negative text context. The first frame continues through WAN2.2's native TI2V image-conditioning path.

## Training and checkpointing

UMT5, VAE, and base WAN2.2 weights remain frozen. Training jointly updates:

- rank-32 DiT LoRA on `q,k,v,o,ffn.0,ffn.2`;
- the three-layer entity-vector encoder.

Training matches `wan22_physics_text_lora_2184_v1`:

- learning rate `1e-4`;
- Dataset repeat `1`;
- `8` epochs;
- save interval `273` steps;
- expected final step `2184`;
- per-scene world-aligned oversampling;
- gradient accumulation `1`;
- `4` workers;
- seed `42`;
- bf16;
- gradient checkpointing;
- AdamW, ConstantLR, weight decay `0.01`;
- eight GPUs through the frozen Accelerate configuration;
- inference at 50 steps, CFG 5.0, LoRA alpha 1.0.

The final safetensors checkpoint combines all LoRA tensors and the entity-vector encoder tensors. The manifest records exact topology, hashes, optimizer state, base-model asset hashes, vector schema, and runtime fingerprints.

## Runtime reuse

The implementation reuses the proven WAN quantity-token transport for:

- sentinel localization and positive-CFG binding;
- frozen UMT5 invocation;
- joint LoRA/conditioner optimization;
- combined checkpoint sealing and verification;
- single and batched generation.

The shared runtime is extended with an explicit `first_entity_vector_mlp_v1` encoder mode. Existing quantity-embedding behavior remains the default and its frozen 19-tensor topology remains unchanged.

## Failure handling and auditability

- Exactly one sentinel must be present after tokenization.
- Exactly one three-scalar vector record must be supplied per case.
- All vector values and projected activations must be finite.
- The adapter records source object, source fields, units, derivation/imputation policy, raw vector, model prompt, audited prompt, and hashes.
- Training and inference reject a checkpoint whose encoder topology or baseline identity differs from the manifest.
- Existing run directories are never overwritten; retries use resumable training state or a new run ID.

## Test strategy

Tests are written before implementation and cover:

- all seven scene extraction policies, including collision object selection and circular `omega*r` derivation;
- zero imputation and invalid-value/unit rejection;
- exact reuse of the structured-physics text renderer;
- one sentinel and one vector channel in the sealed adapter record;
- three-linear-layer topology, shape, finite outputs, and positive-CFG-only binding;
- combined checkpoint topology and manifest verification without regressing the existing quantity baseline;
- bundle discovery, validation, task compilation, dry-run generation contracts, and run identity;
- a one-case GPU smoke before the full 2184-step launch;
- full generation followed by `scene_default_v14` expert and CSTI aggregation.

## Success criteria

The task is complete when the baseline is registered and validated, the final 2184-step checkpoint is sealed, all planned inference jobs have predictions, and the run contains a completed v14 evaluation report with CSTI results for every applicable five-scene job or explicit evaluator failure records for any unavailable references.
