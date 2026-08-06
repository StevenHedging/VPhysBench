# WAN2.2 TI2V LoRA baselines

This Bundle directory registers the trainable WAN2.2-TI2V-5B FlowMatch LoRA
recipe as two schema-v5 managed Baseline identities:

- `wan22_ti2v_5b_lora_r32_v3_generic` uses the Case's canonical prompt and
  explicitly ignores structured physics.
- `wan22_ti2v_5b_lora_r32_v3_physics` appends the Case's formal structured
  physics with `six_scene_physics_clauses_v2`; all current formal quantities
  are selected, with value, unit and symbol preserved.

The distinction is Baseline-owned through `input_policy` and
`adapter.physics_transform`; both identities can run the same `direct_eval`
or `finetune_eval` Task. Their model, trainer, runner, spatial and temporal
recipes are otherwise identical.

## Layout

```text
baseline.json                 generic portable manifest
physics.baseline.json         physics-injecting portable manifest
baseline.local.example.json   deployment override template
baseline.local.json           shared local deployment, ignored by Git
driver.py                     shared managed WAN driver selector
```

The former command endpoint has been replaced by the managed driver contract.
The common `Wan22ManagedDriver` delegates media, training and inference to the
repository's shared WAN implementation. The portable manifests now expose
`adapter`, `trainer` and `runner` as separate, auditable components.

## Local deployment

Copy `baseline.local.example.json` to the Git-ignored
`baseline.local.json`, then configure the checkpoint, WAN project, model root
and Python environment. Because both manifests live in this directory, the
one local override intentionally services both identities.

The frozen step-410 checkpoint is identified by SHA-256
`7f8f28a36faa309431e7ea58e7de3c61cd58266b62653ee69c3b9f666745acfe`.
The driver refuses a configured checkpoint whose digest is missing or differs.

```bash
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  wan22_ti2v_5b_lora_r32_v3_generic
/root/miniconda3/envs/phybench/bin/physbench baseline validate \
  wan22_ti2v_5b_lora_r32_v3_physics
```
