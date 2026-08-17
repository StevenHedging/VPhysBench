from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import physbench.dataset_hub as dataset_hub
from physbench.cli import build_parser
from physbench.dataset_hub import diagnose_project, pull_dataset
from physbench.datasets import load_dataset
from physbench.io import canonical_sha256


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
        case = {
            "case_id": "fixture_case",
            "scene_id": "pendulum",
            "assets": {
                "caption": "assets/00-caption.json",
                "first_frame": "assets/example/video.bin",
                "first_frame_mask_manifest": "assets/02-mask.bin",
                "physics_annotation": "assets/03-physics.json",
                "reference_video": "assets/example/video.bin",
            },
            "appearance": {},
            "temporal": {},
        }
        (release / "cases.jsonl").write_text(
            json.dumps(case, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (release / "scenes").mkdir()
        (release / "scenes" / "pendulum.json").write_text(
            json.dumps({
                "schema_version": "2.0",
                "scene_id": "pendulum",
                "display_name": "Pendulum",
                "structured_physics_parameters": [
                    "objects.object_1.radius",
                    "objects.object_1.initial_angle",
                    "environment.string_length",
                ],
                "generalization_factors": [],
                "constraints": [],
                "metric_spec": {},
            }),
            encoding="utf-8",
        )
        (release / "views").mkdir()
        case_digest = canonical_sha256(["fixture_case"])
        (release / "views" / "view_a.json").write_text(
            json.dumps({
                "schema_version": "3.0",
                "view_id": "view_a",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "split_semantics": {
                    "primary_partitions": ["train", "test"],
                    "generalization_regimes": ["id", "ood", "mixed"],
                    "regime_is_relative_to": "view_a.train",
                },
                "scenes": {
                    "pendulum": {"train": ["fixture_case"], "test": []}
                },
                "test_annotations": {},
            }),
            encoding="utf-8",
        )
        (release / "views" / "view_b.json").write_text(
            json.dumps({
                "schema_version": "2.0",
                "view_id": "view_b",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "scenes": {"pendulum": {"group_1": ["fixture_case"]}},
            }),
            encoding="utf-8",
        )
        descriptor_value = json.loads(
            (release / "dataset.json").read_text(encoding="utf-8")
        )
        descriptor_value["views"] = {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        }
        (release / "dataset.json").write_text(
            json.dumps(descriptor_value),
            encoding="utf-8",
        )
        assets = root / "datasets" / "assets"
        assets.mkdir()
        (assets / "README.md").write_text(
            "tracked Dataset assets scaffold\n",
            encoding="utf-8",
        )
        return root

    def asset_contents(self) -> dict[str, bytes]:
        return {
            "assets/00-caption.json": (
                b'{"case_id":"fixture_case","scene_id":"pendulum",'
                b'"caption":"fixture prompt"}'
            ),
            "assets/example/video.bin": b"verified Dataset media",
            "assets/02-mask.bin": b"mask",
            "assets/03-physics.json": (
                b'{"case_id":"fixture_case","scene_id":"pendulum",'
                b'"physics":{"objects":{"object_1":{"radius":{"value":1,'
                b'"unit":"m","symbol":"r"},"initial_angle":{"value":1,'
                b'"unit":"rad","symbol":"theta"}}},"environment":{'
                b'"string_length":{"value":1,"unit":"m","symbol":"L"}}}}'
            ),
        }

    def staged_asset(self, root: Path, relative: str) -> Path:
        matches = list(
            (
                root
                / "datasets"
                / ".vphysbench"
                / "staging"
                / REVISION
            ).glob(f"attempt-*/assets/{relative}")
        )
        self.assertEqual(1, len(matches), matches)
        return matches[0]

    def make_distribution(self, root: Path) -> tuple[Path, dict[str, object]]:
        remote = root / "remote"
        shard = remote / "distribution" / "v1" / "shards" / "assets-00000.zip"
        shard.parent.mkdir(parents=True)
        contents = self.asset_contents()
        with zipfile.ZipFile(shard, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in sorted(contents.items()):
                archive.writestr(name, payload)
        active_assets = root / "datasets" / "assets"
        for name, payload in contents.items():
            target = root / "datasets" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        dataset_digest = load_dataset(
            root / "datasets" / "releases" / "13.0.0" / "dataset.json",
            check_assets=True,
        ).digest
        for name in contents:
            (root / "datasets" / name).unlink()
        (active_assets / "example").rmdir()
        manifest: dict[str, object] = {
            "schema_version": "1.0",
            "dataset_id": "physics_video_seven_scene_v13",
            "release": "13.0.0",
            "dataset_digest": dataset_digest,
            "total_files": len(contents),
            "total_bytes": sum(len(payload) for payload in contents.values()),
            "shards": [{
                "name": shard.name,
                "path": "distribution/v1/shards/assets-00000.zip",
                "size_bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }],
            "files": [
                {
                    "path": name,
                    "shard": shard.name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(contents.items())
            ],
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
        self.assertFalse(
            (root / "datasets" / "assets" / "example" / "video.bin").exists()
        )
        self.assertEqual(
            b"verified Dataset media",
            self.staged_asset(root, "example/video.bin").read_bytes(),
        )
        self.assertEqual(
            root / "datasets" / "releases" / "13.0.0" / "dataset.json",
            descriptor,
        )

    def test_pull_direct_assets_downloads_expanded_tree_at_bound_revision(self) -> None:
        root = self.make_project()
        binding_path = root / "datasets" / "huggingface.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["delivery"] = "direct_assets_v1"
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        remote = root / "remote"
        for name, payload in self.asset_contents().items():
            target = remote / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        commands: list[list[str]] = []

        def direct_runner(command: list[str], **_: object) -> SimpleNamespace:
            commands.append(command)
            local_dir = Path(command[command.index("--local-dir") + 1])
            shutil.copytree(remote / "assets", local_dir / "assets", dirs_exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        descriptor = pull_dataset(
            binding_path,
            local_dir=root / "datasets",
            hf_executable="/tools/hf",
            runner=direct_runner,
            check_assets=True,
        )

        self.assertEqual([
            "/tools/hf",
            "download",
            "StevenHedging/VPhysBench",
            "--repo-type",
            "dataset",
            "--revision",
            REVISION,
            "--include",
            "assets/**",
            "--local-dir",
            str((root / "datasets").resolve()),
        ], commands[0])
        self.assertEqual(1, len(commands))
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

    def test_pull_reports_object_progress_and_cache_reuse(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        shard_path = "distribution/v1/shards/assets-00000.zip"
        cached = (
            root
            / "datasets"
            / ".vphysbench"
            / "cache"
            / REVISION
            / shard_path
        )
        cached.parent.mkdir(parents=True)
        shutil.copyfile(remote / shard_path, cached)
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=False,
            )

        output = stderr.getvalue()
        self.assertIn(
            "Dataset object start: distribution/v1/manifest.json",
            output,
        )
        self.assertIn(
            "Dataset object complete: distribution/v1/manifest.json",
            output,
        )
        self.assertIn(f"Dataset object start: {shard_path}", output)
        self.assertIn(f"Dataset object cache reuse: {shard_path}", output)
        self.assertIn(f"Dataset object complete: {shard_path}", output)

    def test_each_shard_uses_its_exact_extraction_budget(self) -> None:
        root = self.make_project()
        remote, manifest = self.make_distribution(root)
        first_path = "distribution/v1/shards/assets-00000.zip"
        second_path = "distribution/v1/shards/assets-00001.zip"
        shutil.copyfile(remote / first_path, remote / second_path)
        first_shard = manifest["shards"][0]  # type: ignore[index]
        second_archive = remote / second_path
        manifest["shards"] = [
            first_shard,
            {
                "name": "assets-00001.zip",
                "path": second_path,
                "size_bytes": second_archive.stat().st_size,
                "sha256": hashlib.sha256(second_archive.read_bytes()).hexdigest(),
            },
        ]
        files = manifest["files"]  # type: ignore[assignment]
        assert isinstance(files, list)
        files[-1]["shard"] = "assets-00001.zip"
        (remote / "distribution" / "v1" / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        budgets: dict[str, int] = {}
        contents = self.asset_contents()

        def record_budget(*args, **kwargs):
            shard_name = kwargs["shard_name"]
            budgets[shard_name] = kwargs["max_extracted_bytes"]
            extracted = []
            for record in files:
                if record["shard"] != shard_name:
                    continue
                target = Path(kwargs["staging_root"]) / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents[record["path"]])
                extracted.append(target)
            return extracted

        with patch.object(
            dataset_hub,
            "verify_and_extract_shard",
            side_effect=record_budget,
        ):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=False,
            )

        expected_budgets: dict[str, int] = {}
        for record in files:
            expected_budgets.setdefault(record["shard"], 0)
            expected_budgets[record["shard"]] += record["size_bytes"]
        self.assertEqual(expected_budgets, budgets)

    def test_extractor_runtime_errors_are_wrapped(self) -> None:
        runtime_root = self.make_project()
        runtime_remote, _ = self.make_distribution(runtime_root)
        with patch.object(
            dataset_hub,
            "verify_and_extract_shard",
            side_effect=RuntimeError("secure POSIX boundary unavailable"),
        ), self.assertRaisesRegex(
            RuntimeError,
            "Dataset shard .*verification failed.*secure POSIX",
        ):
            pull_dataset(
                runtime_root / "datasets" / "huggingface.json",
                local_dir=runtime_root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(runtime_remote, []),
                check_assets=False,
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
        self.assertFalse(
            (root / "datasets" / "assets" / "example" / "video.bin").exists()
        )
        self.assertTrue(self.staged_asset(root, "example/video.bin").is_file())

    def test_damaged_active_tree_is_revalidated_against_manifest(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        binding = root / "datasets" / "huggingface.json"
        destination = root / "datasets"
        pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, []),
            check_assets=True,
        )
        video = destination / "assets" / "example" / "video.bin"
        video.write_bytes(b"X" * len(b"verified Dataset media"))
        commands: list[list[str]] = []

        pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, commands),
            check_assets=True,
        )

        self.assertEqual(
            ["distribution/v1/manifest.json"],
            [command[3] for command in commands],
        )
        self.assertEqual(b"verified Dataset media", video.read_bytes())

    def test_matching_active_tree_fast_path_only_downloads_manifest(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        binding = root / "datasets" / "huggingface.json"
        destination = root / "datasets"
        pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, []),
            check_assets=True,
        )
        commands: list[list[str]] = []

        descriptor = pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, commands),
            check_assets=True,
        )

        self.assertEqual(
            ["distribution/v1/manifest.json"],
            [command[3] for command in commands],
        )
        self.assertEqual(
            destination / "releases" / "13.0.0" / "dataset.json",
            descriptor,
        )

    def test_staged_inventory_rejects_hash_special_and_empty_directory_extras(
        self,
    ) -> None:
        for mutation in ("hash", "fifo", "empty-directory"):
            with self.subTest(mutation=mutation):
                root = self.make_project()
                remote, _ = self.make_distribution(root)
                real_extract = dataset_hub.verify_and_extract_shard

                def extract_then_mutate(*args, **kwargs):
                    result = real_extract(*args, **kwargs)
                    staging = Path(kwargs["staging_root"])
                    if mutation == "hash":
                        video = staging / "assets" / "example" / "video.bin"
                        video.write_bytes(b"X" * video.stat().st_size)
                    elif mutation == "fifo":
                        os.mkfifo(staging / "assets" / "extra.pipe")
                    else:
                        (staging / "assets" / "empty-extra").mkdir()
                    return result

                with patch.object(
                    dataset_hub,
                    "verify_and_extract_shard",
                    side_effect=extract_then_mutate,
                ), self.assertRaisesRegex(
                    RuntimeError,
                    "inventory|hash|unsafe|special|directory",
                ):
                    pull_dataset(
                        root / "datasets" / "huggingface.json",
                        local_dir=root / "datasets",
                        hf_executable="/tools/hf",
                        runner=self.copying_runner(remote, []),
                        check_assets=True,
                    )

                self.assertFalse(
                    (
                        root
                        / "datasets"
                        / "assets"
                        / "example"
                        / "video.bin"
                    ).exists()
                )

    def test_manifest_readme_hash_cannot_be_overridden_by_local_scaffold(self) -> None:
        root = self.make_project()
        remote, manifest = self.make_distribution(root)
        shard = remote / "distribution" / "v1" / "shards" / "assets-00000.zip"
        remote_readme = b"immutable manifest README\n"
        with zipfile.ZipFile(
            shard,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for name, payload in sorted(self.asset_contents().items()):
                archive.writestr(name, payload)
            archive.writestr("assets/README.md", remote_readme)
        shard_record = manifest["shards"][0]  # type: ignore[index]
        shard_record["size_bytes"] = shard.stat().st_size
        shard_record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest_files = manifest["files"]  # type: ignore[assignment]
        assert isinstance(manifest_files, list)
        manifest_files.append({
            "path": "assets/README.md",
            "shard": "assets-00000.zip",
            "size_bytes": len(remote_readme),
            "sha256": hashlib.sha256(remote_readme).hexdigest(),
        })
        manifest["total_files"] = len(manifest_files)
        manifest["total_bytes"] += len(remote_readme)  # type: ignore[operator]
        (remote / "distribution" / "v1" / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "README|trusted|manifest|collision|differs",
        ):
            pull_dataset(
                root / "datasets" / "huggingface.json",
                local_dir=root / "datasets",
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=True,
            )

        self.assertEqual(
            "tracked Dataset assets scaffold\n",
            (root / "datasets" / "assets" / "README.md").read_text(
                encoding="utf-8"
            ),
        )
        self.assertFalse(
            (root / "datasets" / "assets" / "example" / "video.bin").exists()
        )

    def test_loader_after_content_or_asset_root_swap_does_not_publish(self) -> None:
        for mutation in ("content", "root"):
            with self.subTest(mutation=mutation):
                root = self.make_project()
                remote, _ = self.make_distribution(root)
                real_load = dataset_hub.load_dataset
                mutated = False

                def load_then_mutate(path, **kwargs):
                    nonlocal mutated
                    snapshot = real_load(path, **kwargs)
                    descriptor = Path(path)
                    if not mutated and ".vphysbench" in descriptor.parts:
                        mutated = True
                        assets = snapshot.asset_root
                        if mutation == "content":
                            video = assets / "assets" / "example" / "video.bin"
                            video.write_bytes(b"X" * video.stat().st_size)
                        else:
                            held = assets / "held-assets"
                            active = assets / "assets"
                            active.rename(held)
                            shutil.copytree(held, active)
                    return snapshot

                with patch.object(
                    dataset_hub,
                    "load_dataset",
                    side_effect=load_then_mutate,
                ), self.assertRaisesRegex(
                    RuntimeError,
                    "changed|inventory|hash|validation",
                ):
                    pull_dataset(
                        root / "datasets" / "huggingface.json",
                        local_dir=root / "datasets",
                        hf_executable="/tools/hf",
                        runner=self.copying_runner(remote, []),
                        check_assets=True,
                    )

                self.assertFalse(
                    (
                        root
                        / "datasets"
                        / "assets"
                        / "example"
                        / "video.bin"
                    ).exists()
                )

    def test_exchange_failure_preserves_existing_active_tree(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        binding = root / "datasets" / "huggingface.json"
        destination = root / "datasets"
        pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, []),
            check_assets=True,
        )
        sentinel = destination / "assets" / "old-active-sentinel.txt"
        sentinel.write_text("old active", encoding="utf-8")

        with patch.object(
            dataset_hub,
            "_renameat2",
            create=True,
            side_effect=OSError("forced pre-exchange failure"),
        ), self.assertRaisesRegex(RuntimeError, "pre-exchange|promot|publish"):
            pull_dataset(
                binding,
                local_dir=destination,
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=True,
            )

        self.assertEqual("old active", sentinel.read_text(encoding="utf-8"))

    def test_post_exchange_hash_failure_restores_existing_active_tree(self) -> None:
        root = self.make_project()
        remote, _ = self.make_distribution(root)
        binding = root / "datasets" / "huggingface.json"
        destination = root / "datasets"
        pull_dataset(
            binding,
            local_dir=destination,
            hf_executable="/tools/hf",
            runner=self.copying_runner(remote, []),
            check_assets=True,
        )
        sentinel = destination / "assets" / "old-active-sentinel.txt"
        sentinel.write_text("old active", encoding="utf-8")
        real_renameat2 = getattr(dataset_hub, "_renameat2", None)
        exchanged = False

        def exchange_then_corrupt(*args, **kwargs):
            nonlocal exchanged
            if real_renameat2 is None:
                raise AssertionError("production did not provide _renameat2")
            result = real_renameat2(*args, **kwargs)
            if not exchanged and kwargs.get("flags") == 2:
                exchanged = True
                video = destination / "assets" / "example" / "video.bin"
                video.write_bytes(b"X" * video.stat().st_size)
            return result

        with patch.object(
            dataset_hub,
            "_renameat2",
            create=True,
            side_effect=exchange_then_corrupt,
        ), self.assertRaisesRegex(RuntimeError, "hash|changed|restor"):
            pull_dataset(
                binding,
                local_dir=destination,
                hf_executable="/tools/hf",
                runner=self.copying_runner(remote, []),
                check_assets=True,
            )

        self.assertTrue(exchanged)
        self.assertEqual("old active", sentinel.read_text(encoding="utf-8"))

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
        self.assertTrue(self.staged_asset(root, "example/video.bin").is_file())

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
