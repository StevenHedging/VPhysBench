from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import io
import json
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from physbench.cli import build_parser, main
from physbench.dataset_hub import diagnose_project
from physbench.environment_doctor import (
    Diagnostic,
    diagnose_checkpoint,
    diagnose_runtime,
)


class EnvironmentDoctorTests(unittest.TestCase):
    def test_doctor_defaults_to_full_human_report(self) -> None:
        args = build_parser().parse_args(["doctor"])
        self.assertEqual("full", args.level)
        self.assertFalse(args.json)

        checks = [
            Diagnostic("python", "ok", "3.11.9"),
            Diagnostic(
                "ffmpeg",
                "error",
                "not found on PATH",
                "install ffmpeg and ensure it is on PATH",
            ),
        ]
        stdout = io.StringIO()
        with patch("physbench.cli.diagnose_project", return_value=checks):
            with contextlib.redirect_stdout(stdout):
                status = main(["doctor"])

        report = stdout.getvalue()
        self.assertEqual(1, status)
        self.assertIn("VPhysBench environment doctor", report)
        self.assertIn("[OK]", report)
        self.assertIn("[ERROR]", report)
        self.assertIn("install ffmpeg", report)
        self.assertIn("ready: no", report)

    def test_json_report_is_one_machine_readable_document(self) -> None:
        checks = [Diagnostic("python", "ok", "3.11.9")]
        stdout = io.StringIO()
        with patch("physbench.cli.diagnose_project", return_value=checks):
            with contextlib.redirect_stdout(stdout):
                status = main(["doctor", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(0, status)
        self.assertEqual("1.0", payload["schema_version"])
        self.assertTrue(payload["ready"])
        self.assertEqual("full", payload["level"])
        self.assertEqual("python", payload["checks"][0]["name"])

    def test_full_runtime_reports_missing_tools_and_optional_modules(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/git" if name == "git" else None

        def find_spec(name: str) -> object | None:
            return None if name in {"cv2", "sam3.model_builder"} else object()

        checks = diagnose_runtime(
            which=which,
            find_spec=find_spec,
            environ={},
            torch_probe=lambda: {
                "version": "2.5.1",
                "cuda_available": False,
                "cuda_version": None,
                "device_count": 0,
            },
            importer=lambda name: object(),
        )
        by_name = {item.name: item for item in checks}

        self.assertEqual("ok", by_name["git"].status)
        self.assertEqual("error", by_name["ffmpeg"].status)
        self.assertEqual("error", by_name["ffprobe"].status)
        self.assertEqual("error", by_name["scene_evaluation_dependencies"].status)
        self.assertIn("cv2", by_name["scene_evaluation_dependencies"].detail)
        self.assertEqual("error", by_name["sam31_dependency"].status)
        self.assertEqual("error", by_name["cuda"].status)
        self.assertEqual("error", by_name["sam31_checkpoint"].status)

    def test_runtime_rejects_dependency_that_exists_but_cannot_import(self) -> None:
        def importer(name: str) -> object:
            if name == "sam3.model_builder":
                raise ImportError("broken native extension")
            return object()

        checks = diagnose_runtime(
            which=lambda name: f"/usr/bin/{name}",
            find_spec=lambda name: object(),
            environ={},
            torch_probe=lambda: {
                "version": "2.5.1",
                "cuda_available": True,
                "cuda_version": "12.4",
                "device_count": 1,
            },
            importer=importer,
        )
        sam31 = next(item for item in checks if item.name == "sam31_dependency")
        self.assertEqual("error", sam31.status)
        self.assertIn("broken native extension", sam31.detail)

    def test_evaluation_rejects_dependency_spec_with_broken_import(self) -> None:
        """Replacing a real evaluation import with find_spec must fail."""
        module_name = "vphysbench_test_broken_native"

        class BrokenLoader(importlib.abc.Loader):
            def create_module(self, spec: object) -> None:
                return None

            def exec_module(self, module: object) -> None:
                raise ImportError("broken native extension")

        class BrokenFinder(importlib.abc.MetaPathFinder):
            def find_spec(
                self,
                fullname: str,
                path: object = None,
                target: object = None,
            ) -> importlib.machinery.ModuleSpec | None:
                if fullname != module_name:
                    return None
                return importlib.machinery.ModuleSpec(fullname, BrokenLoader())

        binding = SimpleNamespace(
            release="14.0.0",
            repo_id="example/VPhysData",
            revision="0" * 40,
        )
        finder = BrokenFinder()
        sys.meta_path.insert(0, finder)
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "physbench.dataset_hub.load_huggingface_dataset_binding",
                return_value=binding,
            ), patch(
                "physbench.dataset_hub.load_dataset",
                return_value=object(),
            ), patch(
                "physbench.dataset_hub.SCENE_EVALUATION_DEPENDENCIES",
                (module_name,),
            ):
                checks = diagnose_project(directory, level="evaluation")
        finally:
            sys.meta_path.remove(finder)
            sys.modules.pop(module_name, None)

        dependency = next(
            item
            for item in checks
            if item.name == "scene_evaluation_dependencies"
        )
        self.assertEqual("error", dependency.status)
        self.assertIn("failed imports", dependency.detail)
        self.assertIn("broken native extension", dependency.detail)
        self.assertNotIn("sam31_dependency", {item.name for item in checks})
        self.assertNotIn("cuda", {item.name for item in checks})

    def test_checkpoint_path_and_digest_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam31.pt"
            checkpoint.write_bytes(b"fixture checkpoint")

            valid = diagnose_checkpoint(
                environ={"VPHYSBENCH_SAM31_CHECKPOINT": str(checkpoint)},
                expected_sha256=(
                    "6fb82a46696074977f095af25a85171d2ea451cec7bef4bbbbb655d57fe5f61f"
                ),
            )
            mismatch = diagnose_checkpoint(
                environ={"VPHYSBENCH_SAM31_CHECKPOINT": str(checkpoint)},
                expected_sha256="0" * 64,
            )

        self.assertEqual("ok", valid.status)
        self.assertEqual("error", mismatch.status)
        self.assertIn("SHA-256", mismatch.detail)

    def test_metadata_and_evaluation_levels_remain_available(self) -> None:
        parser = build_parser()
        self.assertEqual(
            "metadata",
            parser.parse_args(["doctor", "--level", "metadata"]).level,
        )
        self.assertEqual(
            "evaluation",
            parser.parse_args(["doctor", "--level", "evaluation"]).level,
        )

    def test_python_requirement_is_level_sensitive(self) -> None:
        version_info = namedtuple(
            "VersionInfo", "major minor micro releaselevel serial"
        )
        expectations = {
            version_info(3, 11, 9, "final", 0): {
                "metadata": "ok",
                "evaluation": "error",
                "runtime": "error",
                "full": "error",
            },
            version_info(3, 12, 0, "final", 0): {
                "metadata": "ok",
                "evaluation": "ok",
                "runtime": "ok",
                "full": "ok",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for version, expected_statuses in expectations.items():
                with patch(
                    "physbench.dataset_hub.diagnose_runtime", return_value=[]
                ), patch("physbench.dataset_hub.sys.version_info", version):
                    for level, expected_status in expected_statuses.items():
                        with self.subTest(version=version[:3], level=level):
                            checks = diagnose_project(directory, level=level)
                            python = next(
                                item for item in checks if item.name == "python"
                            )
                            self.assertEqual(expected_status, python.status)

    def test_runtime_level_checks_runtime_without_dataset_or_checkpoint(self) -> None:
        parser = build_parser()
        self.assertEqual(
            "runtime",
            parser.parse_args(["doctor", "--level", "runtime"]).level,
        )

        evaluator_smoke = Diagnostic(
            "evaluator_smoke",
            "ok",
            "CSTI evaluator import smoke passed",
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "physbench.dataset_hub.diagnose_runtime",
                return_value=[evaluator_smoke],
            ) as diagnose_runtime_mock:
                checks = diagnose_project(directory, level="runtime")

        self.assertEqual(
            ["python", "hf_cli", "evaluator_smoke"],
            [item.name for item in checks],
        )
        self.assertNotIn("dataset_binding", {item.name for item in checks})
        self.assertNotIn("dataset_assets", {item.name for item in checks})
        self.assertNotIn("sam31_checkpoint", {item.name for item in checks})
        diagnose_runtime_mock.assert_called_once_with(include_checkpoint=False)

    def test_full_doctor_keeps_diagnosing_after_broken_dataset_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Diagnostic("git", "ok", "/usr/bin/git")
            with patch("physbench.dataset_hub.diagnose_runtime", return_value=[runtime]):
                checks = diagnose_project(directory, level="full")

        self.assertTrue(any(item.name == "dataset_binding" for item in checks))
        self.assertIn(runtime, checks)

    def test_json_mode_converts_unexpected_probe_failure_to_diagnostic(self) -> None:
        stdout = io.StringIO()
        with patch(
            "physbench.cli.diagnose_project",
            side_effect=PermissionError("dataset metadata is unreadable"),
        ):
            with contextlib.redirect_stdout(stdout):
                status = main(["doctor", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(1, status)
        self.assertFalse(payload["ready"])
        self.assertEqual("doctor_internal", payload["checks"][0]["name"])
        self.assertIn("PermissionError", payload["checks"][0]["detail"])


if __name__ == "__main__":
    unittest.main()
