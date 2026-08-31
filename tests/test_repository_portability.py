from __future__ import annotations

import re
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from _paths import ROOT


PRESERVED_PREFIXES = ("docs/", "tests/")
ACTIVE_SUFFIXES = {".bash", ".cfg", ".ini", ".json", ".mk", ".py", ".pyi", ".sh", ".toml", ".yaml", ".yml"}
ACTIVE_FILENAMES = {"Makefile"}
MACHINE_PATHS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\\\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:public|fsx)(?=/|\b)"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:workspace|workspaces)(?=/|\b)"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
)


def _active_tracked_text(root: Path = ROOT) -> list[tuple[str, str]]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    entries: list[tuple[str, str]] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        if relative.startswith(PRESERVED_PREFIXES) or "/fixtures/" in relative:
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in ACTIVE_SUFFIXES and path.name not in ACTIVE_FILENAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        entries.append((relative, text))
    return entries


def _machine_path_violations(root: Path = ROOT) -> list[str]:
    return [
        f"{relative}: {match.group(0)}"
        for relative, content in _active_tracked_text(root)
        for pattern in MACHINE_PATHS
        for match in pattern.finditer(content)
    ]


class RepositoryPortabilityTest(unittest.TestCase):
    def test_active_publication_surface_has_no_machine_paths(self) -> None:
        self.assertEqual([], _machine_path_violations())

    def test_active_tracked_config_rejects_posix_and_windows_machine_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vphysbench-portability-") as directory:
            root = Path(directory)
            active = root / "configs" / "machine_paths.json"
            active.parent.mkdir(parents=True)
            active.write_text(
                '{"linux": "/home/alice/data", "shared": "/public/dataset", '
                '"cluster": "/fsx/team", "workspace": "/workspace/project"}\n',
                encoding="utf-8",
            )
            source = root / "src" / "machine_paths.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                'MAC_ROOT = "/Users/alice/project"\n'
                'DATA_ROOT = r"C:\\Users\\Alice\\data"\n',
                encoding="utf-8",
            )
            ignored = root / "local-machine-paths.json"
            ignored.write_text('{"path": "/Users/alice/private"}\n', encoding="utf-8")
            docs = root / "docs" / "portability.md"
            docs.parent.mkdir()
            docs.write_text("Avoid /fsx/user/benchmark in tracked configuration.\n", encoding="utf-8")
            fixture = root / "tests" / "fixtures" / "paths.txt"
            fixture.parent.mkdir(parents=True)
            fixture.write_text("/workspace/demo C:\\\\work\\\\demo\n", encoding="utf-8")
            (root / ".gitignore").write_text("local-machine-paths.json\n", encoding="utf-8")
            self._track_fixture(root)

            violations = _machine_path_violations(root)

        self.assertIn("configs/machine_paths.json: /home/alice", violations)
        self.assertIn("configs/machine_paths.json: /public", violations)
        self.assertIn("configs/machine_paths.json: /fsx", violations)
        self.assertIn("configs/machine_paths.json: /workspace", violations)
        self.assertIn("src/machine_paths.py: /Users/alice", violations)
        self.assertIn("src/machine_paths.py: C:" + chr(92), violations)
        self.assertFalse(any("docs/" in item for item in violations))
        self.assertFalse(any("tests/" in item for item in violations))
        self.assertFalse(any("local-machine-paths" in item for item in violations))

    @staticmethod
    def _track_fixture(root: Path) -> None:
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)

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
