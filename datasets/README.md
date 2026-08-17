# Dataset layout

Git tracks the compact Dataset 14.0.0 release metadata and an immutable public
Hugging Face binding. Large media assets are downloaded into this directory and
remain ignored by Git.

```text
datasets/
├── huggingface.json
├── releases/14.0.0/
│   ├── dataset.json
│   ├── cases.jsonl
│   ├── scenes/
│   └── views/
└── assets/                 downloaded media and case-local annotations
```

Download and validate:

```bash
physbench dataset pull
physbench validate-dataset \
  --dataset datasets/releases/14.0.0/dataset.json \
  --check-assets
```

`datasets/huggingface.json` stores only repository identity, Dataset identity,
release and immutable commit. It must never contain a token. Credentials belong
in the user Hugging Face cache.

The immutable revision contains an expanded `assets/` tree. `physbench dataset
pull` downloads `assets/**` directly from that exact commit into the local
`datasets/assets/` directory and performs the final Dataset readiness check.
Interrupted downloads can be resumed by rerunning the same command.

The uploaded tree contains every member frozen by `assets.lock.json`, including
each Case's first frame, reference video, masks, reference-observation tubes,
trajectories, review metadata, visualizations, caption and physics annotation.
The Dataset is authoritative and read-only during evaluation. Derived inputs,
predictions, logs and visualizations belong under `run/<run_id>/` or an
external immutable cache.
