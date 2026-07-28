from __future__ import annotations

from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import BaselinePlugin
from ..domain import BaselineBundle, BaselineTaskInstance
from ..io import sha256_file
from .adapter_loader import load_data_adapter
from .compiler import ManagedTaskBuilder
from .driver import load_managed_driver


def _runtime_dependencies(
    extra: dict[str, Path],
    *,
    submission: bool = False,
) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths: dict[str, Path] = {
        "src/physbench/baseline_runtime/adapter.py": root / "adapter.py",
        "src/physbench/baseline_runtime/adapter_loader.py": (
            root / "adapter_loader.py"
        ),
        "src/physbench/baseline_runtime/compiler.py": root / "compiler.py",
        "src/physbench/baseline_runtime/driver.py": root / "driver.py",
        "src/physbench/baseline_runtime/input_contract.py": (
            root / "input_contract.py"
        ),
        "src/physbench/baseline_runtime/plugin.py": root / "plugin.py",
        "src/physbench/baseline_runtime/task_instance_validation.py": (
            root / "task_instance_validation.py"
        ),
    }
    if submission:
        paths[
            "src/physbench/baseline_runtime/submission.py"
        ] = root / "submission.py"
    overlap = sorted(set(paths) & set(extra))
    if overlap:
        raise ValueError(
            f"managed driver dependency names shadow runtime files: {overlap}"
        )
    paths.update(extra)
    missing = [
        f"{name}={path}"
        for name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"managed runtime dependencies missing: {missing}"
        )
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def verify_managed_instance(
    bundle: BaselineBundle,
    task_builder: ManagedTaskBuilder,
    instance: BaselineTaskInstance,
) -> None:
    instance.verify()
    identity = instance.value["identity"]
    baseline = identity["baseline"]
    if (
        baseline.get("baseline_id") != bundle.baseline_id
        or baseline.get("baseline_version") != bundle.baseline_version
        or baseline.get("digest") != bundle.digest
        or baseline.get("deployment_digest") != bundle.deployment_digest
    ):
        raise ValueError(
            "task instance targets a different managed Baseline deployment"
        )
    if (
        identity["task_builder"].get("fingerprint")
        != task_builder.fingerprint
    ):
        raise ValueError(
            "managed TaskBuilder fingerprint does not match deployment"
        )
    adapter = identity["data_adapter"]
    if (
        adapter.get("fingerprint")
        != task_builder.data_adapter.fingerprint
        or adapter.get("materialization_fingerprint")
        != task_builder.data_adapter.materialization_fingerprint
    ):
        raise ValueError(
            "managed DataAdapter fingerprint does not match deployment"
        )


class ManagedBaselinePlugin(BaselinePlugin):
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self.data_adapter = load_data_adapter(bundle)
        self.driver = load_managed_driver(bundle)
        driver_paths = self.driver.dependency_paths()
        adapter_paths = self.data_adapter.dependency_paths()
        overlap = sorted(set(driver_paths) & set(adapter_paths))
        if overlap:
            raise ValueError(
                "managed driver and DataAdapter dependencies collide: "
                f"{overlap}"
            )
        dependencies = _runtime_dependencies(
            {**driver_paths, **adapter_paths}
        )
        self.task_builder = ManagedTaskBuilder(
            bundle,
            dependencies,
            self.data_adapter,
        )

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        verify_managed_instance(self.bundle, self.task_builder, instance)
        return self.driver.run_task(
            instance=instance,
            run_dir=run_dir,
            execute=execute,
            stop_after_training=stop_after_training,
        )
