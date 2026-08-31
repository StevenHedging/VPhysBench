from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physbench import bootstrap


class BootstrapEnvironmentTests(unittest.TestCase):
    def _project_root(self, directory: str) -> Path:
        root = Path(directory) / "checkout with spaces"
        (root / "constraints").mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        (root / "constraints" / "metadata.txt").write_text("\n")
        (root / "constraints" / "evaluation-cu128.txt").write_text("\n")
        return root

    def _venv_with_current_python(self, root: Path, name: str = ".venv") -> Path:
        venv = root / name
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n"
        )
        (venv / "bin" / "python").symlink_to(sys.executable)
        return venv

    def test_metadata_plan_is_literal_and_uses_default_venv(self) -> None:
        """A missing hub install or metadata doctor command must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            plan = bootstrap.plan_bootstrap(
                project_root=root,
                profile="metadata",
                python=Path("/opt/python/bin/python3.11"),
                python_version=(3, 11),
            )

        self.assertEqual(
            [
                "/opt/python/bin/python3.11 -m venv '" + str(root / ".venv") + "'",
                "'" + str(root / ".venv" / "bin" / "python")
                + "' -m pip install --constraint '"
                + str(root / "constraints" / "metadata.txt")
                + "' -e '"
                + str(root)
                + "[hub]'",
                "'" + str(root / ".venv" / "bin" / "python")
                + "' -m physbench doctor --project-root '"
                + str(root)
                + "' --level metadata",
            ],
            [command.render() for command in plan.commands],
        )

    def test_evaluation_plan_installs_cuda_torch_before_constrained_extras(self) -> None:
        """Reordering CUDA torch after evaluator extras must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            plan = bootstrap.plan_bootstrap(
                project_root=root,
                profile="evaluation",
                python=Path("/opt/python/bin/python3.12"),
                python_version=(3, 12),
            )

        rendered = [command.render() for command in plan.commands]
        self.assertIn(
            "-m pip install --index-url https://download.pytorch.org/whl/cu128 "
            "torch==2.10.0 torchvision==0.25.0",
            rendered[1],
        )
        self.assertIn("--constraint", rendered[2])
        self.assertIn("[hub,scene-evaluation,sam31-evaluation]", rendered[2])
        self.assertTrue(rendered[3].endswith("--level runtime"))

    def test_incompatible_python_is_rejected_before_any_command(self) -> None:
        """Accepting Python 3.11 for evaluation must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            with self.assertRaisesRegex(bootstrap.BootstrapError, "Python 3.12"):
                bootstrap.plan_bootstrap(
                    project_root=root,
                    profile="evaluation",
                    python=Path("/opt/python/bin/python3.11"),
                    python_version=(3, 11),
                )

    def test_existing_compatible_venv_is_reused_without_creation(self) -> None:
        """Adding a venv creation command for an existing compatible venv fails."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = self._venv_with_current_python(root, "already there")
            plan = bootstrap.plan_bootstrap(
                project_root=root,
                profile="metadata",
                python=Path(sys.executable),
                python_version=(sys.version_info.major, sys.version_info.minor),
                venv=venv,
            )

        self.assertEqual("reuse venv", plan.actions[0])
        self.assertNotIn(" -m venv ", "\n".join(
            command.render() for command in plan.commands
        ))

    def test_existing_venv_without_interpreter_is_refused_before_planning(self) -> None:
        """Trusting pyvenv.cfg when bin/python is absent must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = root / ".venv"
            venv.mkdir()
            (venv / "pyvenv.cfg").write_text("version = 3.11.9\n")

            with self.assertRaisesRegex(bootstrap.BootstrapError, "interpreter"):
                bootstrap.plan_bootstrap(
                    project_root=root,
                    profile="metadata",
                    python=Path(sys.executable),
                    python_version=(sys.version_info.major, sys.version_info.minor),
                    venv=venv,
                )

    def test_existing_venv_mismatch_is_rejected_before_install_commands(self) -> None:
        """Running pip before detecting a swapped interpreter must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = root / ".venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text(
                f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n"
            )
            interpreter = venv / "bin" / "python"
            interpreter.write_text("#!/bin/sh\nprintf '9.9\\n'\n")
            interpreter.chmod(0o755)

            plan = bootstrap.plan_bootstrap(
                project_root=root,
                profile="metadata",
                python=Path(sys.executable),
                python_version=(sys.version_info.major, sys.version_info.minor),
                venv=venv,
            )
            with patch.object(bootstrap, "_run_command") as run_command:
                with self.assertRaisesRegex(bootstrap.BootstrapError, "incompatible"):
                    bootstrap.execute_plan(plan)

        run_command.assert_not_called()

    def test_existing_venv_need_not_match_bootstrap_minor_version(self) -> None:
        """Rejecting two metadata-compatible Python minors must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = self._venv_with_current_python(root)
            bootstrap_minor = sys.version_info.minor + 1
            plan = bootstrap.plan_bootstrap(
                project_root=root,
                profile="metadata",
                python=Path(f"/opt/python/bin/python3.{bootstrap_minor}"),
                python_version=(3, bootstrap_minor),
                venv=venv,
            )
            with patch.object(bootstrap, "_run_command") as run_command:
                bootstrap.execute_plan(plan)

        self.assertEqual(2, run_command.call_count)

    def test_incompatible_existing_venv_is_refused_without_recreation(self) -> None:
        """Replacing an existing Python 3.11 venv for Python 3.12 must fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = self._venv_with_current_python(root)
            with self.assertRaisesRegex(bootstrap.BootstrapError, "incompatible"):
                bootstrap.plan_bootstrap(
                    project_root=root,
                    profile="evaluation",
                    python=Path("/opt/python/bin/python3.12"),
                    python_version=(3, 12),
                    venv=venv,
                )

    def test_dry_run_prints_plan_and_mutates_nothing(self) -> None:
        """Creating the venv or invoking the runner in dry-run must fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            stdout = io.StringIO()
            with patch.object(bootstrap, "_run_command") as run_command, contextlib.redirect_stdout(stdout):
                status = bootstrap.main([
                    "--project-root", str(root),
                    "--profile", "metadata",
                    "--dry-run",
                ])

            self.assertEqual(0, status)
            self.assertFalse((root / ".venv").exists())
            run_command.assert_not_called()
            self.assertEqual(
                "VPhysBench bootstrap plan (metadata)\n"
                "action: create venv\n"
                + sys.executable
                + " -m venv '"
                + str(root / ".venv")
                + "'\n",
                stdout.getvalue()[:len(
                    "VPhysBench bootstrap plan (metadata)\n"
                    "action: create venv\n"
                    + sys.executable
                    + " -m venv '"
                    + str(root / ".venv")
                    + "'\n"
                )],
            )

    def test_dry_run_does_not_execute_an_existing_venv_interpreter(self) -> None:
        """A reused venv interpreter side effect during dry-run must fail this test."""
        with tempfile.TemporaryDirectory() as directory:
            root = self._project_root(directory)
            venv = root / ".venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text(
                f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n"
            )
            marker = Path(directory) / "interpreter-was-run"
            interpreter = venv / "bin" / "python"
            interpreter.write_text(
                "#!/bin/sh\n"
                ": > \"$VPHYSBENCH_TEST_MARKER\"\n"
                f"printf '{sys.version_info.major}.{sys.version_info.minor}\\n'\n"
            )
            interpreter.chmod(0o755)

            with patch.dict(
                os.environ, {"VPHYSBENCH_TEST_MARKER": str(marker)}
            ), contextlib.redirect_stdout(io.StringIO()):
                status = bootstrap.main([
                    "--project-root", str(root),
                    "--profile", "metadata",
                    "--venv", str(venv),
                    "--dry-run",
                ])

            self.assertEqual(0, status)
            self.assertFalse(marker.exists())

    def test_shell_launcher_runs_dry_plan_from_outside_checkout(self) -> None:
        """Using the caller's directory rather than BASH_SOURCE must fail."""
        root = Path(__file__).resolve().parents[1]
        launcher = root / "scripts" / "bootstrap_env.sh"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["/bin/bash", str(launcher), "--profile", "metadata", "--dry-run"],
                cwd=Path(directory),
                env={**os.environ, "VPHYSBENCH_BOOTSTRAP_PYTHON": sys.executable},
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("VPhysBench bootstrap plan (metadata)", result.stdout)
        self.assertIn(str(root / ".venv"), result.stdout)

    def test_shell_launcher_auto_selects_a_compatible_interpreter(self) -> None:
        """A launcher that cannot select Python 3.11+ must fail this test."""
        root = Path(__file__).resolve().parents[1]
        launcher = root / "scripts" / "bootstrap_env.sh"
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory) / "tools"
            tools.mkdir()
            (tools / "python3.11").symlink_to(sys.executable)
            result = subprocess.run(
                ["bash", str(launcher), "--profile", "metadata", "--dry-run"],
                cwd=Path(directory),
                env={
                    **os.environ,
                    "PATH": str(tools) + ":/usr/bin:/bin",
                    "VPHYSBENCH_BOOTSTRAP_PYTHON": "",
                },
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("VPhysBench bootstrap plan (metadata)", result.stdout)

    def test_shell_launcher_rejects_python_311_for_evaluation(self) -> None:
        """Selecting Python 3.11 for evaluation must fail this test."""
        root = Path(__file__).resolve().parents[1]
        launcher = root / "scripts" / "bootstrap_env.sh"
        with tempfile.TemporaryDirectory() as directory:
            tools = Path(directory) / "tools"
            tools.mkdir()
            # The launcher asks candidates to prove their version with `-c`;
            # this shim models a Python 3.11 candidate failing that 3.12 gate.
            candidate = tools / "python3"
            candidate.write_text("#!/bin/sh\nexit 1\n")
            candidate.chmod(0o755)
            (tools / "dirname").symlink_to("/usr/bin/dirname")
            result = subprocess.run(
                ["/bin/bash", str(launcher), "--profile", "evaluation", "--dry-run"],
                cwd=Path(directory),
                env={
                    **os.environ,
                    "PATH": str(tools),
                    "VPHYSBENCH_BOOTSTRAP_PYTHON": "",
                },
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(2, result.returncode)
        self.assertIn("no Python 3.12+ interpreter found", result.stderr)


if __name__ == "__main__":
    unittest.main()
