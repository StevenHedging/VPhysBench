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

V5_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "5.0.0"
V5_DATASET = V5_RELEASE_ROOT / "dataset.json"

V51_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "5.1.0"
V51_DATASET = V51_RELEASE_ROOT / "dataset.json"

V6_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "6.0.0"
V6_DATASET = V6_RELEASE_ROOT / "dataset.json"

V7_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "7.0.0"
V7_DATASET = V7_RELEASE_ROOT / "dataset.json"

V8_RELEASE_ROOT = PHYSICS_VIDEO_ROOT / "releases" / "8.0.0"
V8_DATASET = V8_RELEASE_ROOT / "dataset.json"
LATEST_DATASET = V8_DATASET
