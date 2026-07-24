# Physics Video Benchmark figure prompt

Generation mode: OpenAI built-in image generation (`imagegen`)

Create one publication-ready landscape infographic for an AI research paper,
16:9, crisp flat vector style, white warm-gray background, generous whitespace,
precise grid, thin navy arrows, rounded rectangular modules, low-saturation
palette: dataset blue, task violet, baseline teal, artifacts amber, evaluation
green. No logos, no watermark, no photorealism, no 3D, no decorative clutter.

Exact title at top: “Physics Video Benchmark”

Small subtitle: “Decoupled architecture and baseline execution timeline”

Design a single figure with two clearly separated panels.

LEFT PANEL, about 42% width, exact heading: “A. Decoupled Benchmark”. Show three
large vertically stacked or triangular connected modules, visually emphasizing
strict separation:

1) blue module exact heading “DATASET” with short labels:

- “Video Assets”
- “Structured Physics”
- “View A / View B”
- “Prompt-free”

Use tiny icons: video frame, parameter table, split grid.

2) violet module exact heading “TASK” with two branches:

- “Task 1 · Finetune + Eval”
- “Task 2 · Direct Eval”

and two condition chips:

- “Generic”
- “Physics”

and evaluation chips:

- “ID”
- “OOD1”
- “OOD2”

3) teal module exact heading “BASELINE” with four internal blocks:

- “Data Adapter”
- “Condition Adapter”
- “Trainer”
- “Predictor”

Use a small WAN + LoRA tag as an example only: “e.g. WAN2.2 + LoRA”.

Connect DATASET → TASK → BASELINE through a narrow central orchestration rail
labeled “Atomic Runner”. Under the panel place the exact formula in a clean
monospaced capsule:

“AtomicRun = DatasetSnapshot × AtomicTask × BaselineSnapshot × Seed”

Show that prompts are produced by the baseline Condition Adapter from structured
physics, not stored in Dataset.

RIGHT PANEL, about 58% width, exact heading: “B. Baseline Evaluation Timeline”.
Draw a clean UML-style dynamic sequence diagram with six vertical lifelines and
exact lane headers:

“Runner” | “Dataset” | “Task” | “Baseline” | “Artifact” | “Evaluator”

Use numbered horizontal arrows and short exact labels:

- “1  Freeze snapshots”
- “2  Select cases · View A / B”
- “3  Adapt media · private cache”
- “4  Build condition · Generic / Physics”

Then show a clear fork with two colored paths:

- upper violet path labeled “Task 1” and steps “5a  Finetune” →
  “Independent adapter”
- lower gray-blue path labeled “Task 2” and step “5b  Load frozen model” with a
  visible “skip training” note.

Merge both paths into:

- “6  Generate videos”
- “7  Evaluate · ID / OOD1 / OOD2”
- “8  Aggregate report”

Near the Task 1 fork, show two separate small artifact boxes exactly labeled:

- “Generic adapter”
- “Physics adapter”

to make clear each condition variant trains its own adapter.

Add a small bottom legend with four concise principles:

“Immutable snapshots” · “Baseline-side adaptation” · “Independent adapters” ·
“Reproducible seeds”

Typography: modern sans-serif, high legibility, all labels horizontal, no tiny
paragraphs. Make every listed label spelled exactly as written. Style should
resemble a polished NeurIPS / CVPR / ICLR method overview figure, balanced and
information-dense but easy to scan.
