# Getting started

This guide takes a fresh checkout to a validated custom-baseline scaffold. A
GPU is not required for metadata checks or the interface smoke test. Full
scene evaluation needs the optional evaluator stack, compatible CUDA/PyTorch,
SAM 2, and the complete Dataset assets.

## 1. Install the Hub and interface environment

```bash
# Option A: python3.11 is available on PATH.
python3.11 -m venv .venv
. .venv/bin/activate

# Option B: use Conda when the host has no python3.11 executable.
# conda create -n vphysbench python=3.11 -y
# conda activate vphysbench

python -m pip install --upgrade pip
python -m pip install -e ".[hub]"
```

`.[hub]` is sufficient for Hugging Face Dataset access, metadata checks, and
the lightweight interface smoke. It deliberately does not install scene
evaluator dependencies.

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

## 3. Install the evaluator stack

```bash
python -m pip install -e ".[scene-evaluation]"
```

The scene-evaluation extra pins the SAM 2 source revision used by this
release. Install PyTorch for the local CUDA version before the extra when the
machine requires a platform-specific wheel. Installing this extra clones SAM 2
from GitHub; configure working GitHub access or preinstall that exact pinned
revision in an offline environment. The official SAM 2 install notes are at
<https://github.com/facebookresearch/sam2/blob/main/INSTALL.md>.

Run the evaluation-level doctor again after installation.

```bash
physbench doctor --level evaluation
physbench validate-dataset \
  --dataset datasets/releases/14.0.0/dataset.json \
  --check-assets
```

`doctor --level evaluation` treats missing Dataset media as an error and checks
that the NumPy/OpenCV/SciPy/Torch/SAM2 evaluator modules are importable. The
first real one-case run additionally verifies CUDA execution and SAM 2 model
access for that scene.

## One-command full readiness check

After installing both evaluator extras and downloading the Dataset, configure
the protocol-pinned SAM3.1 checkpoint and run the default full doctor:

```bash
python -m pip install -e ".[scene-evaluation,sam31-evaluation]"
export VPHYSBENCH_SAM31_CHECKPOINT=SAM31_CHECKPOINT_ABSOLUTE_PATH
physbench doctor
```

Replace `SAM31_CHECKPOINT_ABSOLUTE_PATH` with the checkpoint's absolute path on
the current machine.

The command checks Python, Git, ffmpeg/ffprobe, the frozen Dataset binding and
assets, evaluator modules, PyTorch/CUDA visibility, and the SAM3.1 checkpoint
SHA-256. It does not install, download, or load a model. Every failure includes
an actionable repair hint and produces a non-zero exit status. Use
`physbench doctor --json` for one machine-readable report. The explicit
`metadata` and `evaluation` levels remain available for lightweight and legacy
workflows.

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
- Evaluator dependency error: install `.[scene-evaluation]` and rerun doctor.
