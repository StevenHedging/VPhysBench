# Custom baseline quickstart

VPhysBench ships no integrated generation algorithm. User integrations live
under `baselines/<baseline_id>` and are discovered from their descriptors.

## Create the bundle

For image-to-video:

```bash
physbench baseline init my_model --backend managed-i2v
```

For video-to-video:

```bash
physbench baseline init my_model --backend managed-v2v
```

The generated directory contains:

```text
baselines/my_model/
├── baseline.json
├── baseline.local.example.json
├── driver.py
├── README.md
└── .gitignore
```

`baseline.json` owns the portable identity, supported Task families, input
policy, media adapter, runner type, and public command arguments.
`baseline.local.json` may override machine-local checkpoint/runtime paths and
must not be committed.

## Standard managed-I2V command

The generic driver invokes the configured command as:

```text
<command> \
  --prompt <text> \
  --image <canonical-conditioning-image> \
  --output <run-owned-mp4> \
  --seed <integer> \
  --job-spec <sealed-json>
```

Requirements:

- Write exactly one video to `--output` and return exit code 0.
- Do not choose another output path.
- Honor the canvas, FPS, start time, and frame-count rule in `--job-spec`.
- Treat the provided prompt and image as the only authorized model inputs.
- Do not read Dataset reference videos or evaluator annotations.
- Put model-specific flags in `runner.config.extra_args`.

The runtime copies/materializes the conditioning image, constructs the output
path under `run/<run_id>/predictions/`, records the command log, and validates
the generated video before evaluation.

## Configure and validate

The initial runner command is `python inference.py`. Replace it with the
portable entry point supplied by your algorithm. Put only local paths in:

```json
{
  "model": {
    "checkpoint": "../external/checkpoint"
  },
  "runtime": {
    "working_directory": "../external/model-repository"
  }
}
```

Save that object as `baseline.local.json`, then run:

```bash
physbench baseline inspect my_model
physbench baseline validate my_model
```

Validation freezes the bundle and deployment digests. Changing a descriptor,
driver, local deployment, or fingerprinted dependency changes the identity.

## Test the protocol before loading a real model

```bash
make smoke-interface
```

`examples/dummy_i2v_command.py` demonstrates only the external CLI shape. It
is not registered, is not a baseline implementation, and must never be used
as a reported result.

## Run one case, then the official Task

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

After the smoke run succeeds, repeat without `--case-id`. Never reuse a
`run_id`; every AtomicRun seals one Dataset × Task × baseline identity.
