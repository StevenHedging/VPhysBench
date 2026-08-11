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

EVALUATOR_IMPORT_FAILURE_SCRIPT = """
import importlib.abc
import sys

FAILURE_KIND = {failure_kind!r}


class EvaluatorImportFailureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            FAILURE_KIND == "optional"
            and fullname == "physbench.evaluation.task_evaluator"
        ):
            raise ModuleNotFoundError(
                "blocked optional module: cv2", name="cv2"
            )
        if (
            FAILURE_KIND == "internal"
            and fullname == "physbench.evaluation.task_evaluator"
        ):
            raise ModuleNotFoundError(
                "injected missing internal module",
                name="physbench.evaluation.common.csti",
            )
        if (
            FAILURE_KIND == "runtime"
            and fullname == "physbench.evaluation.task_evaluator"
        ):
            raise RuntimeError("injected evaluator runtime failure")
        return None


sys.meta_path.insert(0, EvaluatorImportFailureFinder())

try:
    from physbench.evaluation import evaluate_task
except Exception as exc:
    if FAILURE_KIND == "optional":
        assert type(exc).__name__ == "SceneEvaluationDependencyError", type(exc)
        assert ".[scene-evaluation]" in str(exc), str(exc)
        assert type(exc.__cause__) is ModuleNotFoundError, type(exc.__cause__)
        assert exc.__cause__.name == "cv2", exc.__cause__.name
    elif FAILURE_KIND == "internal":
        assert type(exc) is ModuleNotFoundError, type(exc)
        assert exc.name == "physbench.evaluation.common.csti", exc.name
        assert str(exc) == "injected missing internal module", str(exc)
    else:
        assert type(exc) is RuntimeError, type(exc)
        assert type(exc).__name__ == "RuntimeError", type(exc).__name__
        assert str(exc) == "injected evaluator runtime failure", str(exc)
else:
    raise AssertionError("evaluate_task did not trigger the injected failure")
"""

SPRING_PACKAGE_IMPORT_SCRIPT = """
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

import physbench.evaluation.scenes.vertical_spring_oscillator as spring

loaded_blocked_roots = {{
    name.split('.', 1)[0]
    for name in sys.modules
    if name.split('.', 1)[0] in BLOCKED_MODULE_ROOTS
}}
assert not loaded_blocked_roots, loaded_blocked_roots

for export_name in ("SpringTrace", "VerticalSpringOscillatorCaseEvaluator"):
    try:
        getattr(spring, export_name)
    except ModuleNotFoundError as exc:
        assert exc.name.split('.', 1)[0] in BLOCKED_MODULE_ROOTS, exc.name
    else:
        raise AssertionError(export_name + " did not load its real dependency graph")
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

    def test_spring_scene_package_defers_all_heavy_exports(self) -> None:
        """Catch eager imports from the public spring scene package itself."""
        result = self._run_subprocess(SPRING_PACKAGE_IMPORT_SCRIPT)

        self.assertEqual(0, result.returncode, result.stderr)

    def test_evaluator_access_explains_required_scene_evaluation_extra(self) -> None:
        """Catch missing optional evaluator extras that lack install guidance."""
        result = self._run_subprocess(
            EVALUATOR_IMPORT_FAILURE_SCRIPT.format(failure_kind="optional")
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_evaluator_access_preserves_missing_internal_modules(self) -> None:
        """Catch lazy imports that misclassify a missing package module as an extra."""
        result = self._run_subprocess(
            EVALUATOR_IMPORT_FAILURE_SCRIPT.format(failure_kind="internal")
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_evaluator_access_preserves_runtime_failures(self) -> None:
        """Catch lazy imports that rewrite evaluator runtime failures as extras."""
        result = self._run_subprocess(
            EVALUATOR_IMPORT_FAILURE_SCRIPT.format(failure_kind="runtime")
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
