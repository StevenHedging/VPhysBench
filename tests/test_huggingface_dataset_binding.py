from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from physbench.huggingface_binding import load_huggingface_dataset_binding


ROOT = Path(__file__).resolve().parents[1]


class HuggingFaceDatasetBindingTests(unittest.TestCase):
    def test_current_binding_targets_v13_with_an_immutable_revision(self) -> None:
        binding = load_huggingface_dataset_binding(
            ROOT / "datasets" / "huggingface.json",
            dataset_path=ROOT / "datasets" / "releases" / "13.0.0" / "dataset.json",
        )

        self.assertEqual("StevenHedging/VPhysBench", binding.repo_id)
        self.assertEqual("physics_video_seven_scene_v13", binding.dataset_id)
        self.assertEqual("13.0.0", binding.release)
        self.assertRegex(binding.revision, r"^[0-9a-f]{40}$")

    def test_binding_rejects_embedded_credentials(self) -> None:
        value = {
            "schema_version": "1.0",
            "provider": "huggingface",
            "repo_type": "dataset",
            "repo_id": "StevenHedging/VPhysBench",
            "revision": "a" * 40,
            "dataset_id": "physics_video_seven_scene_v13",
            "release": "13.0.0",
            "token": "must-not-be-stored",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huggingface.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_huggingface_dataset_binding(path)


if __name__ == "__main__":
    unittest.main()
