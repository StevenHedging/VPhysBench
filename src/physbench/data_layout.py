from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PHYSICS_VIDEO_ROOT = DATASETS_ROOT
PHYSICS_VIDEO_ASSETS = DATASETS_ROOT / "assets"
PHYSICS_VIDEO_PROVENANCE = DATASETS_ROOT / "provenance"
RELEASES_ROOT = DATASETS_ROOT / "releases"

V1_RELEASE_ROOT = RELEASES_ROOT / "1.0.0"
V1_CASES = V1_RELEASE_ROOT / "cases.jsonl"
V1_VIEW_A = V1_RELEASE_ROOT / "views" / "view_a.json"
V1_VIEW_B = V1_RELEASE_ROOT / "views" / "view_b_seed42_g5.json"

V2_RELEASE_ROOT = RELEASES_ROOT / "2.0.0"
V2_DATASET = V2_RELEASE_ROOT / "dataset.json"

V3_RELEASE_ROOT = RELEASES_ROOT / "3.0.0"
V3_DATASET = V3_RELEASE_ROOT / "dataset.json"

V4_RELEASE_ROOT = RELEASES_ROOT / "4.0.0"
V4_DATASET = V4_RELEASE_ROOT / "dataset.json"

V5_RELEASE_ROOT = RELEASES_ROOT / "5.0.0"
V5_DATASET = V5_RELEASE_ROOT / "dataset.json"

V51_RELEASE_ROOT = RELEASES_ROOT / "5.1.0"
V51_DATASET = V51_RELEASE_ROOT / "dataset.json"

V6_RELEASE_ROOT = RELEASES_ROOT / "6.0.0"
V6_DATASET = V6_RELEASE_ROOT / "dataset.json"

V7_RELEASE_ROOT = RELEASES_ROOT / "7.0.0"
V7_DATASET = V7_RELEASE_ROOT / "dataset.json"

V8_RELEASE_ROOT = RELEASES_ROOT / "8.0.0"
V8_DATASET = V8_RELEASE_ROOT / "dataset.json"

V9_RELEASE_ROOT = RELEASES_ROOT / "9.0.0"
V9_DATASET = V9_RELEASE_ROOT / "dataset.json"

# Keep the operational default pinned until Tasks are deliberately migrated
# from the v8 Dataset identity to the mask-augmented v9 identity.
LATEST_DATASET = V8_DATASET
