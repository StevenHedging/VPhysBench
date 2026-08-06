from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class VerticalSpringReleaseTests(unittest.TestCase):
    def test_build_release_preserves_v12_and_adds_id_spring_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            shutil.copytree(
                ROOT / "datasets/releases/13.0.0",
                repo / "datasets/releases/12.0.0",
            )
            base_release = repo / "datasets/releases/12.0.0"
            all_cases = [
                json.loads(line)
                for line in (base_release / "cases.jsonl").read_text().splitlines()
            ]
            spring_ids = {
                case["case_id"]
                for case in all_cases
                if case["scene_id"] == "vertical_spring_oscillator"
            }
            base_cases = [
                case
                for case in all_cases
                if case["scene_id"] != "vertical_spring_oscillator"
            ]
            (base_release / "cases.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in base_cases),
                encoding="utf-8",
            )
            descriptor = json.loads((base_release / "dataset.json").read_text())
            descriptor.update(
                {"dataset_id": "physics_video_six_scene_v12", "release": "12.0.0"}
            )
            _write_json(base_release / "dataset.json", descriptor)
            (base_release / "scenes/vertical_spring_oscillator.json").unlink()
            for name in ("view_a", "view_b"):
                path = base_release / f"views/{name}.json"
                view = json.loads(path.read_text())
                view["scenes"].pop("vertical_spring_oscillator")
                if name == "view_a":
                    view["test_annotations"] = {
                        case_id: value
                        for case_id, value in view["test_annotations"].items()
                        if case_id not in spring_ids
                    }
                _write_json(path, view)
            provenance_root = repo / "datasets/provenance/releases/12.0.0"
            provenance_root.mkdir(parents=True)
            current_provenance = [
                json.loads(line)
                for line in (
                    ROOT / "datasets/provenance/releases/13.0.0/cases.jsonl"
                ).read_text().splitlines()
            ]
            provenance_root.joinpath("cases.jsonl").write_text(
                "".join(
                    json.dumps(item) + "\n"
                    for item in current_provenance
                    if item["scene_id"] != "vertical_spring_oscillator"
                ),
                encoding="utf-8",
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

            audit_path = repo / "spring_import_audit.jsonl"
            audit_path.write_text(
                "".join(json.dumps(item) + "\n" for item in audits),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_dataset_v13.py",
                    "--repo-root",
                    str(repo),
                    "--audit",
                    str(audit_path),
                ],
                check=True,
                env={"PYTHONPATH": "src"},
            )

            release = repo / "datasets/releases/13.0.0"
            descriptor = json.loads((release / "dataset.json").read_text())
            cases = [
                json.loads(line)
                for line in (release / "cases.jsonl").read_text().splitlines()
            ]
            view_a = json.loads((release / "views/view_a.json").read_text())
            report = json.loads(
                (repo / "datasets/provenance/releases/13.0.0/build.json").read_text()
            )
            migration_path = (
                repo / "datasets/provenance/releases/13.0.0/migration.json"
            )
            self.assertTrue(migration_path.is_file())
            migration = json.loads(migration_path.read_text())
            spring = view_a["scenes"]["vertical_spring_oscillator"]
            self.assertEqual("13.0.0", descriptor["release"])
            self.assertEqual("physics_video_seven_scene_v13", descriptor["dataset_id"])
            self.assertEqual(801, len(cases))
            self.assertEqual(801, report["case_count"])
            self.assertEqual("physics_video_six_scene_v12", migration["base"]["dataset_id"])
            self.assertEqual("physics_video_seven_scene_v13", migration["output"]["dataset_id"])
            self.assertEqual(2, migration["counts"]["spring_cases_added"])
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
