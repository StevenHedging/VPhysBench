from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.verify_release_archive import _metadata_doctor_issues, verify_archive


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArchiveTests(unittest.TestCase):
    def test_head_archive_is_self_contained_and_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vphysbench-archive-test-") as directory:
            issues = verify_archive(ROOT, Path(directory))

        self.assertEqual([], issues)

    def test_archive_runs_relocation_gates_outside_checkout(self) -> None:
        """Would fail if a relocated gate runs from inside the checkout.

        The tracked fixture records every required command and rejects an
        in-checkout working directory. Its metadata doctor mirrors the clean
        archive contract: missing Dataset media is a warning, not an error.
        """
        with tempfile.TemporaryDirectory(prefix="vphysbench-archive-") as directory:
            temporary_root = Path(directory) / "release verification with spaces"
            repository = temporary_root / "fixture repository"
            self._create_relocation_fixture(repository)

            issues = verify_archive(repository, temporary_root)

            self.assertEqual([], issues)
            self.assertEqual(
                [
                    "bootstrap",
                    "help",
                    "doctor",
                    "baseline-list",
                    "interface-smoke",
                ],
                (temporary_root / "archive-command-log.txt").read_text(
                    encoding="utf-8"
                ).splitlines(),
            )

    def test_metadata_doctor_requires_ready_coherent_expected_checks(self) -> None:
        """Accepting a failed, incomplete, or incoherent doctor must fail."""
        valid = {
            "schema_version": "1.0",
            "level": "metadata",
            "ready": True,
            "summary": {"ok": 2, "warnings": 2, "errors": 0},
            "checks": [
                {"name": "python", "status": "ok", "detail": "3.12.14"},
                {
                    "name": "hf_cli",
                    "status": "warning",
                    "detail": "not installed",
                },
                {
                    "name": "dataset_binding",
                    "status": "ok",
                    "detail": "example/VPhysData@" + "0" * 40,
                },
                {
                    "name": "dataset_assets",
                    "status": "warning",
                    "detail": "not installed",
                },
            ],
        }

        def completed(
            report: dict[str, object], returncode: int = 0
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                ["physbench", "doctor"],
                returncode,
                stdout=json.dumps(report),
                stderr="",
            )

        self.assertEqual([], _metadata_doctor_issues(completed(valid)))
        cases = {
            "nonzero exit": completed(valid, returncode=1),
            "not ready": completed({**valid, "ready": False}),
            "missing expected check": completed(
                {**valid, "checks": valid["checks"][:-1]}
            ),
            "incoherent summary": completed(
                {
                    **valid,
                    "summary": {"ok": 4, "warnings": 0, "errors": 0},
                }
            ),
        }
        for label, result in cases.items():
            with self.subTest(label=label):
                self.assertNotEqual([], _metadata_doctor_issues(result))

    def _create_relocation_fixture(self, root: Path) -> None:
        for relative in (
            "README.md",
            "RELEASE_MANIFEST.json",
            "datasets/huggingface.json",
            "datasets/releases/14.0.0/dataset.json",
            "docs/GETTING_STARTED.md",
            "docs/CUSTOM_BASELINE_QUICKSTART.md",
            "run/README.md",
            "baselines/README.md",
            "pyproject.toml",
            "scripts/bootstrap_env.sh",
            "src/physbench/bootstrap.py",
            "src/physbench/cli.py",
            "constraints/metadata.txt",
            "constraints/evaluation-cu128.txt",
            "tasks/official/six_scene_direct_eval_v1.json",
            "tasks/official/six_scene_train_six_scene_eval_v1.json",
            "configs/evaluation/protocols/scene_default_v1.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        bootstrap = root / "src/physbench/bootstrap.py"
        bootstrap.write_text(
            """\
import sys
from pathlib import Path


CHECKOUT = Path(__file__).resolve().parents[2]
LOG = CHECKOUT.parent / "archive-command-log.txt"
if Path.cwd().is_relative_to(CHECKOUT):
    raise SystemExit("command executed inside checkout")
if sys.argv[1:] != ["--profile", "metadata", "--dry-run"]:
    raise SystemExit(f"unexpected bootstrap arguments: {sys.argv[1:]!r}")
LOG.write_text(
    (LOG.read_text(encoding="utf-8") if LOG.exists() else "")
    + "bootstrap\\n",
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        launcher = root / "scripts/bootstrap_env.sh"
        launcher.write_text(
            """\
#!/usr/bin/env bash
set -eu
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
exec "$VPHYSBENCH_BOOTSTRAP_PYTHON" "$project_root/src/physbench/bootstrap.py" "$@"
""",
            encoding="utf-8",
        )
        launcher.chmod(0o755)

        (root / "src/physbench/__init__.py").write_text("", encoding="utf-8")
        (root / "src/physbench/__main__.py").write_text(
            """\
import json
import sys
from pathlib import Path


CHECKOUT = Path(__file__).resolve().parents[2]
LOG = CHECKOUT.parent / "archive-command-log.txt"


def main() -> int:
    if Path.cwd().is_relative_to(CHECKOUT):
        print("command executed inside checkout", file=sys.stderr)
        return 3
    if "--help" in sys.argv:
        LOG.write_text(LOG.read_text(encoding="utf-8") + "help\\n", encoding="utf-8")
        return 0
    if sys.argv[1:4] == ["doctor", "--level", "metadata"]:
        LOG.write_text(LOG.read_text(encoding="utf-8") + "doctor\\n", encoding="utf-8")
        print(json.dumps({
            "schema_version": "1.0",
            "level": "metadata",
            "ready": True,
            "summary": {"ok": 2, "warnings": 2, "errors": 0},
            "checks": [
                {"name": "python", "status": "ok", "detail": "3.12.14"},
                {"name": "hf_cli", "status": "warning", "detail": "not installed"},
                {"name": "dataset_binding", "status": "ok", "detail": "bound"},
                {"name": "dataset_assets", "status": "warning", "detail": "not installed"},
            ],
        }))
        return 0
    if sys.argv[1:3] == ["baseline", "list"]:
        LOG.write_text(LOG.read_text(encoding="utf-8") + "baseline-list\\n", encoding="utf-8")
        print("[]")
        return 0
    return 2


raise SystemExit(main())
""",
            encoding="utf-8",
        )
        smoke = root / "scripts/smoke_custom_baseline.py"
        smoke.parent.mkdir(parents=True, exist_ok=True)
        smoke.write_text(
            """\
import sys
from pathlib import Path


CHECKOUT = Path(__file__).resolve().parents[1]
if Path.cwd().is_relative_to(CHECKOUT):
    raise SystemExit("command executed inside checkout")
(CHECKOUT.parent / "archive-command-log.txt").write_text(
    (CHECKOUT.parent / "archive-command-log.txt").read_text(encoding="utf-8")
    + "interface-smoke\\n",
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=VPhysBench test",
                "-c",
                "user.email=vphysbench@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
