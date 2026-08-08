from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from physbench.cli import build_parser
from physbench.dataset_hub import diagnose_project, pull_dataset


REVISION = "520a86a64e87642e357b82d52037ee708ede5243"


class DatasetHubTests(unittest.TestCase):
    def make_project(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        release = root / "datasets" / "releases" / "13.0.0"
        release.mkdir(parents=True)
        (root / "datasets" / "huggingface.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "provider": "huggingface",
                "repo_type": "dataset",
                "repo_id": "StevenHedging/VPhysBench",
                "revision": REVISION,
                "dataset_id": "physics_video_seven_scene_v13",
                "release": "13.0.0",
            }),
            encoding="utf-8",
        )
        (release / "dataset.json").write_text(
            json.dumps({
                "schema_version": "5.0",
                "dataset_id": "physics_video_seven_scene_v13",
                "release": "13.0.0",
                "asset_root": "../..",
                "cases": "cases.jsonl",
                "scene_catalog": "scenes",
                "views": {},
            }),
            encoding="utf-8",
        )
        return root

    def test_pull_uses_bound_repo_and_immutable_revision(self) -> None:
        root = self.make_project()
        commands: list[list[str]] = []

        def run(command: list[str], **_: object) -> SimpleNamespace:
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        descriptor = pull_dataset(
            root / "datasets" / "huggingface.json",
            local_dir=root / "datasets",
            hf_executable="/tools/hf",
            runner=run,
            check_assets=False,
        )

        self.assertEqual(
            [
                "/tools/hf",
                "download",
                "StevenHedging/VPhysBench",
                "--repo-type",
                "dataset",
                "--revision",
                REVISION,
                "--local-dir",
                str((root / "datasets").resolve()),
            ],
            commands[0],
        )
        self.assertEqual(
            root / "datasets" / "releases" / "13.0.0" / "dataset.json",
            descriptor,
        )

    def test_pull_reports_missing_hf_cli(self) -> None:
        root = self.make_project()
        with patch("physbench.dataset_hub.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "Hugging Face CLI"):
                pull_dataset(
                    root / "datasets" / "huggingface.json",
                    local_dir=root / "datasets",
                    check_assets=False,
                )

    def test_doctor_distinguishes_metadata_and_evaluation_readiness(self) -> None:
        root = self.make_project()
        metadata = diagnose_project(root, level="metadata")
        evaluation = diagnose_project(root, level="evaluation")
        self.assertFalse(any(item.status == "error" for item in metadata))
        self.assertTrue(any(
            item.name == "dataset_assets" and item.status == "error"
            for item in evaluation
        ))

    def test_cli_exposes_nested_dataset_pull_and_doctor(self) -> None:
        parser = build_parser()
        pull = parser.parse_args(["dataset", "pull"])
        doctor = parser.parse_args(["doctor", "--level", "metadata"])
        self.assertEqual("pull", pull.dataset_command)
        self.assertEqual("metadata", doctor.level)


if __name__ == "__main__":
    unittest.main()
