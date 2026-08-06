from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from physbench import data_layout
from physbench.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"


class FlatDatasetLayoutTests(unittest.TestCase):
    def test_authoritative_roots_are_direct_children_of_datasets(self) -> None:
        self.assertEqual(DATASETS_ROOT, data_layout.DATASETS_ROOT)
        self.assertEqual(DATASETS_ROOT, data_layout.PHYSICS_VIDEO_ROOT)
        self.assertEqual(DATASETS_ROOT / "assets", data_layout.PHYSICS_VIDEO_ASSETS)
        self.assertEqual(
            DATASETS_ROOT / "provenance",
            data_layout.PHYSICS_VIDEO_PROVENANCE,
        )
        self.assertEqual(
            DATASETS_ROOT / "releases" / "13.0.0" / "dataset.json",
            data_layout.V13_DATASET,
        )

    def test_flat_roots_replace_the_legacy_wrapper(self) -> None:
        for name in ("assets", "provenance", "releases"):
            with self.subTest(name=name):
                self.assertTrue((DATASETS_ROOT / name).is_dir())
        self.assertFalse((DATASETS_ROOT / "physics_video").exists())

    def test_source_archives_are_provenance_with_a_release_compatibility_link(
        self,
    ) -> None:
        source_archives = DATASETS_ROOT / "provenance" / "source_archives"
        compatibility = DATASETS_ROOT / "assets" / "source_archives"
        self.assertTrue(source_archives.is_dir())
        self.assertTrue(compatibility.is_symlink())
        self.assertEqual(source_archives.resolve(), compatibility.resolve())

    def test_current_release_resolves_without_an_asset_lock(self) -> None:
        expected = {"13.0.0": 916}
        for version, case_count in expected.items():
            with self.subTest(version=version):
                descriptor = DATASETS_ROOT / "releases" / version / "dataset.json"
                dataset = load_dataset(descriptor, check_assets=True)
                self.assertEqual(case_count, len(dataset.cases))
                self.assertIsNone(dataset.asset_lock)

if __name__ == "__main__":
    unittest.main()
