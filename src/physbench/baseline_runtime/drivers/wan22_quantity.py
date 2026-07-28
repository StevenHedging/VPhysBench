from __future__ import annotations

from pathlib import Path
from typing import Any

from ...baseline_plugins.wan22_quantity import (
    Wan22QuantityExecutionEngine,
)
from .wan22 import Wan22ManagedDriver


class Wan22QuantityManagedDriver(Wan22ManagedDriver):
    """Managed runtime for WAN LoRA plus SI-aware quantity embeddings."""

    def _execution_engine_class(self):
        return Wan22QuantityExecutionEngine

    def _adapter_recipe(self) -> dict[str, Any]:
        return self.bundle.value["adapter"]["config"]

    def dependency_paths(self) -> dict[str, Path]:
        repository_root = Path(__file__).resolve().parents[4]
        paths = super().dependency_paths()
        paths.update({
            "src/physbench/baseline_runtime/drivers/wan22_quantity.py": (
                Path(__file__)
            ),
            "src/physbench/baseline_plugins/wan22_quantity.py": (
                repository_root
                / "src"
                / "physbench"
                / "baseline_plugins"
                / "wan22_quantity.py"
            ),
            "src/physbench/baselines/wan22_quantity.py": (
                repository_root
                / "src"
                / "physbench"
                / "baselines"
                / "wan22_quantity.py"
            ),
            "src/physbench/baselines/wan22_quantity_model.py": (
                repository_root
                / "src"
                / "physbench"
                / "baselines"
                / "wan22_quantity_model.py"
            ),
            "scripts/train_wan22_quantity.sh": (
                repository_root / "scripts" / "train_wan22_quantity.sh"
            ),
            "scripts/wan22_quantity_train.py": (
                repository_root / "scripts" / "wan22_quantity_train.py"
            ),
            "scripts/wan22_quantity_generate.py": (
                repository_root / "scripts" / "wan22_quantity_generate.py"
            ),
            "scripts/wan22_quantity_generate_batch.py": (
                repository_root
                / "scripts"
                / "wan22_quantity_generate_batch.py"
            ),
        })
        missing = [
            f"{name}={path}"
            for name, path in paths.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"WAN quantity runtime dependencies missing: {missing}"
            )
        return paths


__all__ = ["Wan22QuantityManagedDriver"]
