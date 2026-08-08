# Dataset layout

Git tracks the compact Dataset 13.0.0 release metadata and an immutable private
Hugging Face binding. Large media assets are downloaded into this directory and
remain ignored by Git.

```text
datasets/
├── huggingface.json
├── releases/13.0.0/
│   ├── dataset.json
│   ├── cases.jsonl
│   ├── scenes/
│   └── views/
└── assets/                 downloaded media and case-local annotations
```

Download and validate:

```bash
hf auth login
physbench dataset pull
physbench validate-dataset \
  --dataset datasets/releases/13.0.0/dataset.json \
  --check-assets
```

`datasets/huggingface.json` stores only repository identity, Dataset identity,
release and immutable commit. It must never contain a token. Credentials belong
in the user Hugging Face cache.

Each Case references a first frame, reference video, optional mask manifest,
caption and structured physics annotation beneath `datasets/assets/`. The
Dataset is authoritative and read-only during evaluation. Derived inputs,
predictions, logs and visualizations belong under `run/<run_id>/` or an
external immutable cache.
