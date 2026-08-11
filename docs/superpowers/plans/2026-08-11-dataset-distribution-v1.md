# Dataset Distribution v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace quota-sensitive leaf-file downloads with an authenticated, resumable, safely extracted immutable shard distribution for Dataset 13.0.0.

**Architecture:** A standalone distribution module validates manifests and archives, while `dataset_hub` coordinates Hugging Face transport and final Dataset validation. A release-only builder creates deterministic stored ZIP shards from precisely the assets referenced by Dataset cases; its output is uploaded to a new Hub revision and never tracked in Git.

**Tech Stack:** Python 3.11 standard library (`hashlib`, `json`, `zipfile`, `pathlib`), Hugging Face `hf` CLI, `unittest`.

## Global Constraints

- Dataset semantic identity, release `13.0.0`, cases, views, scenes, and digest remain unchanged.
- Hub access always uses a 40-character immutable commit from `datasets/huggingface.json`.
- No credentials, Dataset archives, extracted assets, caches, or staging directories enter the Git release.
- Remote archive paths must be safe before extraction and all promoted files must match declared hashes.

---

### Task 1: Define and validate distribution manifest v1

**Files:**
- Create: `src/physbench/dataset_distribution.py`
- Create: `tests/test_dataset_distribution.py`

**Interfaces:**
- Produces: `load_distribution_manifest(path, *, dataset_id, release) -> DistributionManifest` and `verify_and_extract_shard(...) -> list[Path]`.

- [ ] **Step 1: Write failing manifest tests**

  Cover a valid literal manifest and rejection of unknown schema versions,
  duplicate file paths, duplicate shard names, non-SHA256 hashes, absolute or
  parent-traversing paths, files outside `assets/`, and inconsistent totals.

- [ ] **Step 2: Implement immutable dataclasses and validation**

  Normalize only POSIX relative paths; never silently rewrite invalid input.

- [ ] **Step 3: Write failing safe-extraction tests**

  Create real temporary ZIPs covering valid extraction, undeclared members,
  symlinks, traversal, duplicate names, shard hash mismatch, file hash mismatch,
  and truncated archives.

- [ ] **Step 4: Implement verified staging extraction**

  Verify the archive hash before opening, member policy before writing, and each
  file hash/size after writing. Return only fully verified staged paths.

### Task 2: Build deterministic Dataset shards

**Files:**
- Create: `scripts/build_dataset_distribution.py`
- Modify: `tests/test_dataset_distribution.py`

**Interfaces:**
- Consumes: Dataset descriptor and indexed case asset references.
- Produces: `distribution/v1/manifest.json` and `distribution/v1/shards/*.zip`.

- [ ] **Step 1: Write a fixture builder test**

  Use hand-created asset files and assert stable member order, shard assignment,
  totals, per-file hashes, and byte-identical output across two builds.

- [ ] **Step 2: Implement the builder**

  Enumerate only paths referenced by all case `assets` mappings, deduplicate and
  sort them, verify they remain under Dataset `asset_root`, and pack stored ZIPs
  up to the requested size bound. Use fixed ZIP timestamps and permissions.

- [ ] **Step 3: Verify builder output with the runtime validator**

  The builder must read its own final manifest through production validation
  and verify every emitted shard before reporting success.

### Task 3: Add resumable shard pulling

**Files:**
- Modify: `src/physbench/dataset_hub.py`
- Modify: `tests/test_dataset_hub_cli.py`

**Interfaces:**
- Consumes: bound Hub repository/revision and the standard remote path
  `distribution/v1/manifest.json`.
- Produces: validated assets under the binding's local Dataset root.

- [ ] **Step 1: Write failing transport tests**

  Record runner commands and simulate manifest download, a valid cached shard,
  a missing shard, HTTP 429 stderr, interrupted download, invalid hash, and a
  final Dataset validation failure. Assert exact revision use and resumability.

- [ ] **Step 2: Download manifest and shards selectively**

  Invoke `hf download` with explicit remote filenames, stream output, and retain
  the revision-specific cache/staging directory on recoverable failure.

- [ ] **Step 3: Promote verified files atomically**

  Replace each destination only from verified staging, then run
  `load_dataset(..., check_assets=True)`.

- [ ] **Step 4: Improve diagnostics**

  Preserve actionable stderr and map authentication, rate limit, disk, hash,
  unsafe archive, and Dataset validation errors to concise messages.

### Task 4: Publish and bind the new immutable revision

**Files:**
- Modify after remote verification: `datasets/huggingface.json`
- Modify: `README.md`
- Modify: `docs/GETTING_STARTED.md`
- Modify: `docs/DATASET.md`

**Interfaces:**
- Produces: a new exact Hub commit containing distribution v1 and a Git binding
  to that commit.

- [ ] **Step 1: Build production shards outside the Git worktree**

  Use the complete validated Dataset source. Record manifest totals and hashes,
  and ensure no provenance/source archives are included.

- [ ] **Step 2: Upload to the private Dataset repository**

  Authenticate through the existing user-level Hugging Face credential. Upload
  the distribution directory, obtain the resulting commit, and do not place a
  token in environment output or files.

- [ ] **Step 3: Validate from an empty destination**

  Pull the exact candidate commit through the production CLI, run full asset
  validation, compare the Dataset digest, compile both Task v1 plans, and run a
  one-case AtomicRun with Evaluation v1.

- [ ] **Step 4: Update binding and documentation**

  Change only the immutable revision in the binding, document resume/cache and
  error behavior, run the full release verification, commit, push, and inspect
  GitHub checks.
