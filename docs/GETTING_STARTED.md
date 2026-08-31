# Getting started

This guide takes a fresh checkout to a validated custom-baseline scaffold. A
GPU is not required for metadata checks or the interface smoke test. Full
scene evaluation needs the optional evaluator stack, a Python 3.12+ runtime,
the validated PyTorch 2.10.0/CUDA 12.8 combination, SAM2/SAM3, a compatible
NVIDIA driver, and the complete Dataset assets.

## 1. Bootstrap the Hub and interface environment

```bash
bash scripts/bootstrap_env.sh --profile metadata
. .venv/bin/activate
```

The metadata profile requires Python 3.11 or newer. It creates or safely
reuses `.venv`, installs the pinned `hub` dependencies, and runs the metadata
doctor. `.[hub]` is sufficient for Hugging Face Dataset access, metadata
checks, and the lightweight interface smoke; it deliberately does not install
scene evaluator dependencies.

Preview the exact plan without creating files or accessing the network:

```bash
bash scripts/bootstrap_env.sh --profile metadata --dry-run
```

Confirm that the tracked release surface is internally consistent:

```bash
physbench doctor --level metadata
make test-interface
make smoke-interface
```

The interface smoke creates a baseline bundle and a video in a temporary
directory. It does not register a baseline in the repository and does not
produce a meaningful benchmark score.

## 2. Download Dataset 14.0.0

The Dataset repository is public. Download its immutable bound revision:

```bash
physbench dataset pull
```

The command reads `datasets/huggingface.json`. It always passes the bound
40-character Hub commit to `hf download`; it never resolves the floating
`main` branch. Authentication is optional for public downloads. If used for
higher rate limits, credentials remain in the user-level Hugging Face cache
and must not be placed in repository files.

The bound revision exposes the complete expanded `assets/` tree. The downloader
requests `assets/**` from that exact commit and writes it beneath the local
`datasets/assets/` directory. Reserve at least 40 GB of free disk space for the
first pull.

Downloads are resumable. After a recoverable interruption, rerun the same
command and Hugging Face will reuse completed files.

`--skip-asset-check` downloads the same assets but skips the final Dataset
readiness check.

`doctor --level metadata` reports missing media as a warning.

## 3. Bootstrap the evaluator stack

```bash
VPHYSBENCH_BOOTSTRAP_PYTHON=python3.12 \
  bash scripts/bootstrap_env.sh --profile evaluation
. .venv/bin/activate
physbench doctor --level runtime
```

The evaluation profile requires Python 3.12 or newer and installs the validated
PyTorch 2.10.0/CUDA 12.8 wheels plus the pinned evaluator dependencies. It
runs the Dataset-independent `runtime` doctor after installation. It needs
network access for packages and the pinned source dependencies, but it never
downloads Dataset media or model checkpoints. Set
`VPHYSBENCH_BOOTSTRAP_PYTHON` only when the desired Python 3.12+ interpreter is
not the one auto-selected from `PATH`.

Run the evaluation-level doctor again after installation.

```bash
physbench doctor --level evaluation
physbench validate-dataset \
  --dataset datasets/releases/14.0.0/dataset.json \
  --check-assets
```

`doctor --level evaluation` treats missing Dataset media as an error and checks
that the NumPy/OpenCV/SciPy/Torch/SAM2 evaluator modules are importable. The
first real one-case run additionally verifies CUDA execution and SAM2 model
access for that scene.

## One-command full readiness check

After the evaluation bootstrap and Dataset download, configure the
protocol-pinned SAM3.1 checkpoint and run the default full doctor:

```bash
export VPHYSBENCH_SAM31_CHECKPOINT=SAM31_CHECKPOINT_ABSOLUTE_PATH
physbench doctor
```

Replace `SAM31_CHECKPOINT_ABSOLUTE_PATH` with the checkpoint's absolute path on
the current machine.

`physbench doctor` is the only readiness authority. `metadata` requires Python
3.11+ and treats missing Dataset media as a warning; `runtime` requires Python
3.12+ and checks executables, evaluator imports, PyTorch/CUDA, and a lightweight
evaluator smoke without Dataset media or checkpoint; `evaluation` checks the
Dataset and SAM2 evaluator; `full` (the default) additionally checks the
SAM3.1 checkpoint SHA-256. It does not install, download, or load a model.
Every required failure has an actionable hint and returns non-zero. Use
`physbench doctor --json` for one machine-readable, versioned report: its
`ready` field may correctly be `false` before external assets are configured,
and the command then exits 1.

## 4. Create a custom baseline

```bash
physbench baseline init my_model --backend managed-i2v
physbench baseline validate my_model
```

Edit `baselines/my_model/baseline.json` for portable algorithm settings. Copy
`baseline.local.example.json` to `baseline.local.json` for checkpoint and
machine-local paths. The latter file is ignored by Git.

The clean release contains no model implementation or checkpoint. The
interface smoke below is runnable immediately; a real experiment starts only
after the generated command and local checkpoint settings point to the user's
own inference environment.

See [CUSTOM_BASELINE_QUICKSTART.md](CUSTOM_BASELINE_QUICKSTART.md) for the
driver contract.

## 5. Run one real case

`atomic-run` resolves and runs the scene evaluator. Confirm that
`.[scene-evaluation]` was installed in step 3 and that the complete Dataset
passes the evaluation-level doctor before running it:

```bash
physbench atomic-run \
  --dataset datasets/releases/14.0.0/dataset.json \
  --task tasks/official/six_scene_direct_eval_v1.json \
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
- Dataset denied: confirm that the frozen public Hub revision is reachable.
- Hub rate limit or interrupted transport: wait if necessary, then rerun the
  same pull; completed files are reused.
- Disk-space error: free space for the expanded assets before retrying.
- Missing assets: rerun `physbench dataset pull`; do not change the revision.
- Baseline schema error: run `physbench baseline validate <id>`.
- Video rejection: inspect the sealed media contract and baseline log.
- Evaluator dependency error: rerun the evaluation bootstrap and doctor.

## Portability contract

A clean tracked checkout is relocatable: it may be moved or extracted beneath
another path, including one containing spaces, and bootstrap derives its root
from the checked-out script. The portable unit is not a copied environment.
Do not copy or publish `.venv`, package caches, Hugging Face caches, Dataset
media, `baseline.local.json` overrides, checkpoints, credentials, or `run/`
outputs. Pull Dataset media through the frozen Hub binding, set
`VPHYSBENCH_SAM31_CHECKPOINT` to the local checkpoint, and keep machine-specific
Baseline paths in ignored `baseline.local.json` files.
