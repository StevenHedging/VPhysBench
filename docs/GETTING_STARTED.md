# Getting started

This guide takes a fresh checkout to a validated custom-baseline scaffold. A
GPU is not required for metadata checks or the interface smoke test. Full
scene evaluation needs the optional evaluator stack, compatible CUDA/PyTorch,
SAM 2, and the complete Dataset assets.

## 1. Install the core environment

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hub]"
```

Confirm that the tracked release surface is internally consistent:

```bash
physbench doctor --level metadata
make test
make smoke-interface
```

The interface smoke creates a baseline bundle and a video in a temporary
directory. It does not register a baseline in the repository and does not
produce a meaningful benchmark score.

## 2. Authenticate and download Dataset 13.0.0

Ask the Dataset owner to grant your Hugging Face account access, then run:

```bash
hf auth login
physbench dataset pull
```

The command reads `datasets/huggingface.json`. It always passes the bound
40-character Hub commit to `hf download`; it never resolves the floating
`main` branch. Credentials remain in the user-level Hugging Face cache and
must not be placed in repository files.

Verify full readiness:

```bash
physbench doctor --level evaluation
physbench validate-dataset \
  --dataset datasets/releases/13.0.0/dataset.json \
  --check-assets
```

`doctor --level metadata` reports missing media as a warning.
`doctor --level evaluation` treats it as an error.

## 3. Install the evaluator stack

```bash
python -m pip install -e ".[scene-evaluation]"
```

The scene-evaluation extra pins the SAM 2 source revision used by this
near-release. Install PyTorch for the local CUDA version before the extra when
the machine requires a platform-specific wheel. The official SAM 2 install
notes are at <https://github.com/facebookresearch/sam2/blob/main/INSTALL.md>.

Run the evaluation-level doctor again after installation.

## 4. Create a custom baseline

```bash
physbench baseline init my_model --backend managed-i2v
physbench baseline validate my_model
```

Edit `baselines/my_model/baseline.json` for portable algorithm settings. Copy
`baseline.local.example.json` to `baseline.local.json` for checkpoint and
machine-local paths. The latter file is ignored by Git.

See [CUSTOM_BASELINE_QUICKSTART.md](CUSTOM_BASELINE_QUICKSTART.md) for the
driver contract.

## 5. Run one real case

```bash
physbench atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline baselines/my_model \
  --case-id circular_r1_silver02cm_img_0370 \
  --run-id my_model_smoke \
  --output-root run \
  --execute
```

Inspect:

```text
run/my_model_smoke/predictions/
run/my_model_smoke/logs/
run/my_model_smoke/evaluation/
```

Only after the one-case run succeeds should you remove `--case-id` and start
the full official Task.

## Failure checklist

- `hf` unavailable: install `.[hub]` and reopen the environment.
- Dataset denied: confirm `hf auth login` and private-repository membership.
- Missing assets: rerun `physbench dataset pull`; do not change the revision.
- Baseline schema error: run `physbench baseline validate <id>`.
- Video rejection: inspect the sealed media contract and baseline log.
- Evaluator dependency error: install `.[scene-evaluation]` and rerun doctor.
