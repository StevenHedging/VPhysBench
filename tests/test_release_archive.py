from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.verify_release_archive import verify_archive


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArchiveTests(unittest.TestCase):
    def test_head_archive_is_self_contained_and_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vphysbench-archive-test-") as directory:
            issues = verify_archive(ROOT, Path(directory))

        self.assertEqual([], issues)

    def test_archive_runs_relocation_gates_outside_checkout(self) -> None:
        """Would fail if a relocated gate runs from inside the checkout.

        The tracked fixture records every required command and rejects an
        in-checkout working directory.  Its metadata doctor deliberately
        reports not-ready so the archive gate must validate the JSON document
        rather than require external Dataset assets.
        """
        with tempfile.TemporaryDirectory(prefix="vphysbench-archive-") as directory:
            temporary_root = Path(directory) / "release verification with spaces"
            repository = temporary_root / "fixture repository"
            self._create_relocation_fixture(repository)

            issues = verify_archive(repository, temporary_root)

            self.assertEqual([], issues)
            self.assertEqual(
                ["help", "doctor", "baseline-list", "interface-smoke"],
                (temporary_root / "archive-command-log.txt").read_text(
                    encoding="utf-8"
                ).splitlines(),
            )

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
            "src/physbench/cli.py",
            "tasks/official/six_scene_direct_eval_v1.json",
            "tasks/official/six_scene_train_six_scene_eval_v1.json",
            "configs/evaluation/protocols/scene_default_v1.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

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
        LOG.write_text("help\\n", encoding="utf-8")
        return 0
    if sys.argv[1:4] == ["doctor", "--level", "metadata"]:
        LOG.write_text(LOG.read_text(encoding="utf-8") + "doctor\\n", encoding="utf-8")
        print(json.dumps({
            "schema_version": "1.0",
            "level": "metadata",
            "ready": False,
            "summary": {"ok": 0, "warnings": 0, "errors": 1},
            "checks": [],
        }))
        return 1
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
