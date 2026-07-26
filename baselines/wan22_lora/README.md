# WAN2.2 TI2V LoRA Baseline Bundle

This directory is a self-registering Physics Video Benchmark Baseline Bundle.
The core registry discovers `baseline.json`; it contains no WAN-specific branch.

## Layout

```text
baseline.json                         portable manifest
baseline.local.example.json           deployment override template
baseline.local.json                   local deployment, ignored by Git
plugin/main.py                         physbench-baseline-v1 command endpoint
```

The command TaskBuilder and DataAdapter live in
`src/physbench/baseline_plugins/wan22.py`. Its `Wan22ExecutionEngine` is also
used by the schema-v4 managed G15 Bundle, so media/model execution is not
duplicated. The five-scene profiles are shared resources in the same package.
This directory's portable digest covers its manifest and thin endpoint. Shared
implementations, profiles, legacy WAN helpers and execution scripts are hashed
separately into the TaskBuilder fingerprint and reported by
`describe.runtime_dependency_fingerprints`.

Machine paths live in `baseline.local.json`; they change the deployment digest
without changing the portable bundle digest.

To configure another machine:

```bash
cp baseline.local.example.json baseline.local.json
```

Edit only the `runtime` and `model` objects. Other override keys are rejected.

Validate from the repository root:

```bash
PYTHONPATH=src python -m physbench baseline validate \
  wan22_ti2v_5b_lora_r32_v3
```
