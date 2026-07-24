# Unified Data Adapter figure prompt

Generation mode: OpenAI built-in image generation (`imagegen`), editing the v1
benchmark figure as a composition/style reference.

Edit the supplied benchmark infographic while preserving its exact overall 16:9
composition, clean NeurIPS/CVPR flat-vector style, white background, palette,
typography hierarchy, two-panel structure, title, Dataset module, Task module,
sequence lifelines, Task 1/Task 2 fork, adapter artifacts, evaluation steps, and
bottom legend.

This is an architecture correction. Change only the Baseline data-adaptation
semantics and related sequence labels.

LEFT PANEL — inside the teal BASELINE module:

- Remove the separate boxes “Data Adapter” and “Condition Adapter”.
- Replace them with ONE wider container labeled exactly “Unified Data Adapter”.
- Inside that single container show five small, clearly readable stages in order:
  “Spatial” → “Temporal” → “Paradigm” → “Text” → “Physics”.
- After the unified container, retain “Trainer” → “Predictor”.
- Replace the old prompt note with:
  “Opaque model-native inputs”
  “WAN Physics → Text”
- Preserve “e.g. WAN2.2 + LoRA”.
- Physics is an optional baseline-private stage. Do not add a separate Condition
  Adapter anywhere.

RIGHT PANEL:

- Step 3: “3  Adapt media · Spatial / Temporal”
- Step 4: “4  Build native inputs · Paradigm / Text / Physics”
- Keep all other task, artifact, generation and evaluation steps unchanged.

Replace the bottom legend item “Baseline-side adaptation” with “Unified baseline
adaptation”.

Accuracy constraints: no visible “Condition Adapter”; Physics is an internal Data
Adapter stage, not a universal prompt mechanism; horizontal legible text; no
unrelated layout or style changes; no logo or watermark.
