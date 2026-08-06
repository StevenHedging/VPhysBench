# Dataset Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one current, release-agnostic Dataset explanation and new-Scene ingestion manual.

**Architecture:** Make `datasets/README.md` the sole normative source and turn the older Dataset documents into compatibility links. Base every rule and example on the current Case-local files, loader, schema, and real assets rather than historical prose.

**Tech Stack:** Markdown, JSON examples, repository Dataset assets, Python validation helpers, Git.

## Global Constraints

- Do not mention release identifiers or version history in the canonical manual.
- Do not add hashes, SHA256 checks, asset locks, duplicate annotation formats, or fallback systems.
- Do not modify Dataset media or annotations.
- Use current Case paths and contracts as the source of truth.

---

### Task 1: Write and connect the canonical Dataset manual

**Files:**
- Modify: `datasets/README.md`
- Modify: `datasets/DATASET_OVERVIEW.md`
- Modify: `docs/DATASET.md`
- Modify: `docs/DATASET_INGESTION.md`
- Modify: `docs/NEW_SCENE_EVALUATOR.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: current `datasets/assets/<scene>/<case>/` layout, `schemas/v5/case.schema.json`, Case-local JSON examples, mask manifests, and loader behavior.
- Produces: one canonical manual at `datasets/README.md` and stable compatibility links from prior entry points.

- [x] **Step 1: Replace the canonical manual**

Write the Dataset overview, atomic Case definition, Case member contracts, mandatory four-question dialogue, ingestion workflow, exclusion rules, and acceptance checklist in `datasets/README.md`.

- [x] **Step 2: Remove duplicated normative manuals**

Reduce `docs/DATASET.md`, `docs/DATASET_INGESTION.md`, and `datasets/DATASET_OVERVIEW.md` to short pointers that identify `datasets/README.md` as the sole current source.

- [x] **Step 3: Update inbound links**

Point the repository documentation list and new-evaluator guide directly at `datasets/README.md`.

- [x] **Step 4: Validate documentation**

Run:

```bash
rg -n -i 'release|v[0-9]+|版本|快照|迁移|冻结' datasets/README.md
python3 - <<'PY'
import json
import re
from pathlib import Path

text = Path("datasets/README.md").read_text()
for index, block in enumerate(re.findall(r"```json\n(.*?)\n```", text, re.S), 1):
    json.loads(block)
print(f"parsed_json_examples={index}")
PY
git diff --check
```

Expected: the terminology search returns no matches, every JSON example parses, and `git diff --check` returns no output.

- [x] **Step 5: Commit**

```bash
git add README.md datasets/README.md datasets/DATASET_OVERVIEW.md \
  docs/DATASET.md docs/DATASET_INGESTION.md docs/NEW_SCENE_EVALUATOR.md \
  docs/superpowers/specs/2026-08-06-dataset-manual-design.md \
  docs/superpowers/plans/2026-08-06-dataset-manual.md
git commit -m "docs: establish canonical dataset ingestion manual"
```
