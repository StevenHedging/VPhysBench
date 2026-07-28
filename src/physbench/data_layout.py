from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PHYSICS_VIDEO_ROOT = DATASETS_ROOT / "physics_video"
PHYSICS_VIDEO_ASSETS = PHYSICS_VIDEO_ROOT / "assets"
PHYSICS_VIDEO_PROVENANCE = PHYSICS_VIDEO_ROOT / "provenance"

V1_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "1.0.0"
V1_CASES = V1_RELEASE_ROOT / "cases.jsonl"
V1_VIEW_A = V1_RELEASE_ROOT / "views" / "view_a.json"
V1_VIEW_B = V1_RELEASE_ROOT / "views" / "view_b_seed42_g5.json"

V2_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "2.0.0"
V2_DATASET = V2_RELEASE_ROOT / "dataset.json"

V3_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "3.0.0"
V3_DATASET = V3_RELEASE_ROOT / "dataset.json"

V4_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "4.0.0"
V4_DATASET = V4_RELEASE_ROOT / "dataset.json"
LATEST_DATASET = V4_DATASET
