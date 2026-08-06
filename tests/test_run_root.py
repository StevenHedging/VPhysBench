from __future__ import annotations

import subprocess
import unittest

from _paths import ROOT


class RunRootContractTest(unittest.TestCase):
    def test_only_run_root_is_active(self) -> None:
        generated = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "run/example/generated.json",
            ],
            cwd=ROOT,
            check=False,
        )
        readme = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "run/README.md",
            ],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(0, generated.returncode)
        self.assertEqual(1, readme.returncode)
        self.assertTrue((ROOT / "run" / "README.md").is_file())
        self.assertFalse((ROOT / "runs").exists())
        self.assertFalse((ROOT / "runs_v2").exists())


if __name__ == "__main__":
    unittest.main()
