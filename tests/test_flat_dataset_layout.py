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
            DATASETS_ROOT / "releases" / "9.0.0" / "dataset.json",
            getattr(data_layout, "V9_DATASET", None),
        )
        self.assertEqual(
            DATASETS_ROOT / "releases" / "10.0.0" / "dataset.json",
            getattr(data_layout, "V10_DATASET", None),
        )
        self.assertEqual(
            DATASETS_ROOT / "releases" / "11.0.0" / "dataset.json",
            getattr(data_layout, "V11_DATASET", None),
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

    def test_current_releases_resolve_and_verify_all_locked_assets(self) -> None:
        expected = {
            "8.0.0": (799, 2032),
            "9.0.0": (799, 5239),
            "10.0.0": (799, 6038),
            "11.0.0": (799, 6038),
        }
        for version, (case_count, asset_count) in expected.items():
            with self.subTest(version=version):
                descriptor = DATASETS_ROOT / "releases" / version / "dataset.json"
                dataset = load_dataset(descriptor, check_assets=True)
                self.assertEqual(case_count, len(dataset.cases))
                self.assertEqual(asset_count, len(dataset.asset_lock["files"]))

    def test_release_scripts_find_repository_when_run_outside_checkout(self) -> None:
        release_root = DATASETS_ROOT / "releases" / "9.0.0"
        for script_name in ("validate_release.py", "generate_first_frame_masks.py"):
            with self.subTest(script_name=script_name):
                script_path = release_root / script_name
                probe = (
                    "import runpy; "
                    f"values = runpy.run_path({str(script_path)!r}, run_name='layout_probe'); "
                    "print(values['REPOSITORY_ROOT'])"
                )
                result = subprocess.run(
                    [sys.executable, "-I", "-c", probe],
                    cwd="/tmp",
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(str(ROOT), result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
