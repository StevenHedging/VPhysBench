from __future__ import annotations

import subprocess
import tomllib
import unittest
from pathlib import Path

from _paths import ROOT


FORBIDDEN = (
    "/root/",
    "/mnt/",
    "/absolute/",
    "physics_video_benchmark",
)
FROZEN_PROTOCOLS = {
    f"configs/evaluation/protocols/scene_default_v{version}.json"
    for version in range(4, 8)
}
PRESERVED_PREFIXES = ("tests/",)


def _active_tracked_text() -> list[tuple[str, str]]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    entries: list[tuple[str, str]] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        if relative in FROZEN_PROTOCOLS or relative.startswith(
            PRESERVED_PREFIXES
        ):
            continue
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        entries.append((relative, text))
    return entries


class RepositoryPortabilityTest(unittest.TestCase):
    def test_active_publication_surface_has_no_machine_paths(self) -> None:
        violations = [
            f"{relative}: {token}"
            for relative, content in _active_tracked_text()
            for token in FORBIDDEN
            if token in content
        ]
        self.assertEqual([], violations)

    def test_public_brand_and_distribution_name_are_vphysbench(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertTrue(readme.startswith("# VPhysBench\n"))
        self.assertEqual("vphysbench", pyproject["project"]["name"])

    def test_checkout_root_contains_the_python_project(self) -> None:
        self.assertTrue((ROOT / "pyproject.toml").is_file())


if __name__ == "__main__":
    unittest.main()
