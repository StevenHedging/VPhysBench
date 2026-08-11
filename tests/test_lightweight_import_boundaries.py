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

EVALUATOR_EXTRA_ERROR_SCRIPT = """
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

try:
    from physbench.evaluation import evaluate_task
except (ModuleNotFoundError, RuntimeError) as exc:
    assert ".[scene-evaluation]" in str(exc), str(exc)
else:
    raise AssertionError("evaluate_task did not require scene-evaluation extras")
""".format(blocked_module_roots=BLOCKED_MODULE_ROOTS)


class LightweightPublicImportBoundaryTests(unittest.TestCase):
    def _run_subprocess(self, script: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in (ROOT / "src", ROOT / "tests", ROOT)
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_public_imports_and_protocol_loading_avoid_optional_runtime_modules(self) -> None:
        """Catch eager optional-runtime imports from the public lightweight APIs."""
        result = self._run_subprocess(IMPORT_CONTRACT_SCRIPT)

        self.assertEqual(0, result.returncode, result.stderr)

    def test_evaluator_access_explains_required_scene_evaluation_extra(self) -> None:
        """Catch evaluator imports that expose missing optional modules directly."""
        result = self._run_subprocess(EVALUATOR_EXTRA_ERROR_SCRIPT)

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
