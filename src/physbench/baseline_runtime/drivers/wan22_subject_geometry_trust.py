from __future__ import annotations

from pathlib import Path

from ...baseline_plugins.wan22_subject_geometry_trust import (
    Wan22SubjectGeometryTrustExecutionEngine,
)
from .wan22_subject_anchor_trust import Wan22SubjectAnchorTrustManagedDriver


class Wan22SubjectGeometryTrustManagedDriver(
    Wan22SubjectAnchorTrustManagedDriver
):
    """Managed runtime for fixed-probe subject-geometry WAN training."""

    def _execution_engine_class(self):
        return Wan22SubjectGeometryTrustExecutionEngine

    def dependency_paths(self) -> dict[str, Path]:
        root = Path(__file__).resolve().parents[4]
        paths = super().dependency_paths()
        relative = (
            "src/physbench/baseline_runtime/drivers/wan22_subject_geometry_trust.py",
            "src/physbench/baseline_plugins/wan22_subject_geometry_trust.py",
            "src/physbench/baselines/wan22_subject_geometry_trust.py",
            "src/physbench/baselines/wan22_subject_geometry_trust_model.py",
            "scripts/train_wan22_subject_geometry_trust.sh",
            "scripts/wan22_subject_geometry_trust_train.py",
        )
        paths.update({name: root / name for name in relative})
        missing = [
            f"{name}={path}" for name, path in paths.items() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"WAN subject-geometry dependencies missing: {missing}"
            )
        return paths


__all__ = ["Wan22SubjectGeometryTrustManagedDriver"]
