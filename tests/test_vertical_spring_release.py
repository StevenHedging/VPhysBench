from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.import_vertical_spring_oscillator import build_release_v13


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class VerticalSpringReleaseTests(unittest.TestCase):
    def test_build_release_preserves_v12_and_adds_id_spring_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            shutil.copytree(
                ROOT / "datasets/releases/12.0.0",
                repo / "datasets/releases/12.0.0",
            )
            shutil.copytree(
                ROOT / "datasets/provenance/releases/12.0.0",
                repo / "datasets/provenance/releases/12.0.0",
            )
            audits = []
            for trial, image in (("T001", "1538"), ("T002", "1539")):
                case_id = f"vertical_spring_s01_x40mm_above_img_{image}"
                asset_directory = (
                    "assets/vertical_spring_oscillator/"
                    f"spring_fixture_{image}"
                )
                case_root = (
                    repo
                    / "datasets"
                    / asset_directory
                )
                _write_json(
                    case_root / "caption.json",
                    {
                        "case_id": case_id,
                        "scene_id": "vertical_spring_oscillator",
                        "caption": "Uses m, r, x_0, k, L_0, and g.",
                    },
                )
                _write_json(
                    case_root / "physics.json",
                    {
                        "case_id": case_id,
                        "scene_id": "vertical_spring_oscillator",
                        "physics": {"objects": {}, "environment": {}},
                    },
                )
                audits.append(
                    {
                        "asset_directory": asset_directory,
                        "case_id": case_id,
                        "scene_id": "vertical_spring_oscillator",
                        "trial_id": trial,
                        "workbook_row": int(trial[1:]) + 5,
                        "source_member": f"batch/IMG_{image}.MOV",
                        "source_group": f"source_{image}",
                        "source_start_frame": 300,
                        "source_start_time_s": 1.25,
                        "source_frame_count": 900,
                        "canonical_frame_count": 600,
                        "direction": "above",
                        "signed_displacement_mm": -40.0,
                        "alignment": {"tail_trim": None},
                        "review": {"status": "approved"},
                    }
                )

            report = build_release_v13(repo, audits)

            release = repo / "datasets/releases/13.0.0"
            descriptor = json.loads((release / "dataset.json").read_text())
            cases = [
                json.loads(line)
                for line in (release / "cases.jsonl").read_text().splitlines()
            ]
            view_a = json.loads((release / "views/view_a.json").read_text())
            spring = view_a["scenes"]["vertical_spring_oscillator"]
            self.assertEqual("13.0.0", descriptor["release"])
            self.assertEqual("physics_video_seven_scene_v13", descriptor["dataset_id"])
            self.assertEqual(801, len(cases))
            self.assertEqual(801, report["case_count"])
            self.assertEqual(1, len(spring["test"]))
            self.assertEqual(1, len(spring["train"]))
            spring_case = next(
                case for case in cases if case["case_id"].endswith("img_1538")
            )
            self.assertTrue(
                spring_case["assets"]["caption"].startswith(
                    "assets/vertical_spring_oscillator/spring_fixture_1538/"
                )
            )
            test_case = spring["test"][0]
            self.assertEqual(
                "id",
                view_a["test_annotations"][test_case]["generalization_regime"],
            )
            self.assertTrue((release / "scenes/vertical_spring_oscillator.json").is_file())
            self.assertTrue(
                (repo / "datasets/releases/12.0.0/dataset.json").is_file()
            )
            provenance = repo / "datasets/provenance/releases/13.0.0/cases.jsonl"
            self.assertEqual(801, len(provenance.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
