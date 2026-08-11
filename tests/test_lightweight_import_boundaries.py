"""Regression coverage for public imports that must remain lightweight."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_MODULE_ROOTS = ("cv2", "numpy", "scipy", "torch", "sam2")
IMPORT_CONTRACT_SCRIPT = """
import importlib.abc
import sys

BLOCKED_MODULE_ROOTS = {blocked_module_roots!r}


class BlockedOptionalModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in BLOCKED_MODULE_ROOTS:
            raise ModuleNotFoundError(
                "blocked optional module: {{}}".format(fullname), name=fullname
            )
        return None


sys.meta_path.insert(0, BlockedOptionalModuleFinder())

from physbench.evaluation import load_evaluation_protocol
from physbench.orchestration import compile_task_instance

load_evaluation_protocol("scene_default_v1")

loaded_blocked_roots = {{
    name.split('.', 1)[0]
    for name in sys.modules
    if name.split('.', 1)[0] in BLOCKED_MODULE_ROOTS
}}
assert not loaded_blocked_roots, loaded_blocked_roots
""".format(blocked_module_roots=BLOCKED_MODULE_ROOTS)


class LightweightPublicImportBoundaryTests(unittest.TestCase):
    def test_public_imports_and_protocol_loading_avoid_optional_runtime_modules(self) -> None:
        """Catch eager optional-runtime imports from the public lightweight APIs."""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in (ROOT / "src", ROOT / "tests", ROOT)
        )
        result = subprocess.run(
            [sys.executable, "-c", IMPORT_CONTRACT_SCRIPT],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
