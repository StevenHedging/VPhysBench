from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
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

    def make_distribution(self, root: Path) -> tuple[Path, dict[str, object]]:
        remote = root / "remote"
        shard = remote / "distribution" / "v1" / "shards" / "assets-00000.zip"
        shard.parent.mkdir(parents=True)
        payload = b"verified Dataset media"
        with zipfile.ZipFile(shard, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("assets/example/video.bin", payload)
        manifest: dict[str, object] = {
            "schema_version": "1.0",
            "dataset_id": "physics_video_seven_scene_v13",
            "release": "13.0.0",
            "dataset_digest": "a" * 64,
            "total_files": 1,
            "total_bytes": len(payload),
            "shards": [{
                "name": shard.name,
                "path": "distribution/v1/shards/assets-00000.zip",
                "size_bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }],
            "files": [{
                "path": "assets/example/video.bin",
                "shard": shard.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }
        manifest_path = remote / "distribution" / "v1" / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return remote, manifest

    def copying_runner(
        self,
        remote: Path,
        commands: list[list[str]],
        *,
        fail_path: str | None = None,
        stderr: str = "",
    ):
        def run(command: list[str], **_: object) -> SimpleNamespace:
            commands.append(command)
            remote_path = command[3]
            if remote_path == fail_path:
                return SimpleNamespace(returncode=1, stdout="", stderr=stderr)
            local_dir = Path(command[command.index("--local-dir") + 1])
            target = local_dir / remote_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote / remote_path, target)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        return run

    def test_pull_uses_bound_repo_and_immutable_revision(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        commands: list[list[str]] = []

        descriptor = pull_dataset(
            root / "datasets" / "huggingface.json",
            local_dir=root / "datasets",
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, commands),
            check_assets=False,
        )

        self.assertEqual(
            [
                "/tools/hf",
                "download",
                "StevenHedging/VPhysBench",
                "distribution/v1/manifest.json",
                "--repo-type",
                "dataset",
                "--revision",
                REVISION,
                "--local-dir",
                str(
                    (
                        root
                        / "datasets"
                        / ".vphysbench"
                        / "cache"
                        / REVISION
                    ).resolve()
                ),
            ],
            commands[0],
        )
        self.assertEqual(
            "distribution/v1/shards/assets-00000.zip",
            commands[1][3],
        )
        self.assertEqual(2, len(commands))
        self.assertEqual(
            b"verified Dataset media",
            (root / "datasets" / "assets" / "example" / "video.bin").read_bytes(),
        )
        self.assertEqual(
            root / "datasets" / "releases" / "13.0.0" / "dataset.json",
            descriptor,
        )

    def test_pull_reuses_a_valid_revision_specific_cached_shard(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        cache = root / "datasets" / ".vphysbench" / "cache" / REVISION
        cached_shard = cache / "distribution" / "v1" / "shards" / "assets-00000.zip"
        cached_shard.parent.mkdir(parents=True)
        shutil.copyfile(
            remote / "distribution" / "v1" / "shards" / "assets-00000.zip",
            cached_shard,
        )
        commands: list[list[str]] = []

        pull_dataset(
            root / "datasets" / "huggingface.json",
            local_dir=root / "datasets",
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, commands),
            check_assets=False,
        )

        self.assertEqual(["distribution/v1/manifest.json"], [item[3] for item in commands])

    def test_pull_preserves_rate_limit_stderr_and_resumable_cache(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        commands: list[list[str]] = []
        shard_path = "distribution/v1/shards/assets-00000.zip"

        with self.assertRaisesRegex(RuntimeError, "rate limit.*HTTP 429"):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(
                    remote,
                    commands,
                    fail_path=shard_path,
                    stderr="HTTP 429: too many requests",
                ),
                check_assets=False,
            )

        self.assertTrue(
            (
                root
                / "datasets"
                / ".vphysbench"
                / "cache"
                / REVISION
                / "distribution"
                / "v1"
                / "manifest.json"
            ).is_file()
        )

    def test_pull_resumes_after_an_interrupted_missing_shard_download(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        shard_path = "distribution/v1/shards/assets-00000.zip"

        with self.assertRaisesRegex(RuntimeError, "connection interrupted"):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(
                    remote,
                    [],
                    fail_path=shard_path,
                    stderr="connection interrupted",
                ),
                check_assets=False,
            )
        self.assertFalse((root / "datasets" / "assets" / "example" / "video.bin").exists())

        retry_commands: list[list[str]] = []
        pull_dataset(
            root / "datasets" / "huggingface.json",
            local_dir=root / "datasets",
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, retry_commands),
            check_assets=False,
        )

        self.assertEqual(
            ["distribution/v1/manifest.json", shard_path],
            [item[3] for item in retry_commands],
        )
        self.assertTrue((root / "datasets" / "assets" / "example" / "video.bin").is_file())

    def test_pull_rejects_a_downloaded_shard_with_invalid_hash(self) -> None:
        root = self.make_project()
        remote, manifest = self.make_distribution(root)
        manifest["shards"][0]["sha256"] = "0" * 64  # type: ignore[index]
        (remote / "distribution" / "v1" / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "checksum|sha256"):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=False,
            )

        self.assertFalse((root / "datasets" / "assets" / "example" / "video.bin").exists())

    def test_pull_keeps_staging_when_final_dataset_validation_fails(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)

        with patch(
            "physbench.dataset_hub.load_dataset",
            side_effect=ValueError("required view_a asset is missing"),
        ), self.assertRaisesRegex(RuntimeError, "final Dataset validation failed"):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=True,
            )

        self.assertFalse((root / "datasets" / "assets" / "example" / "video.bin").exists())
        self.assertTrue(
            (
                root
                / "datasets"
                / ".vphysbench"
                / "staging"
                / REVISION
                / "assets"
                / "example"
                / "video.bin"
            ).is_file()
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
