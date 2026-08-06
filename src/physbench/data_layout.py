from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PHYSICS_VIDEO_ROOT = DATASETS_ROOT
PHYSICS_VIDEO_ASSETS = DATASETS_ROOT / "assets"
PHYSICS_VIDEO_PROVENANCE = DATASETS_ROOT / "provenance"
RELEASES_ROOT = DATASETS_ROOT / "releases"

V13_RELEASE_ROOT = RELEASES_ROOT / "13.0.0"
V13_DATASET = V13_RELEASE_ROOT / "dataset.json"

# There is deliberately only one runtime Dataset entry. Historical releases
# remain recoverable from Git history, not from the active filesystem surface.
LATEST_DATASET = V13_DATASET
