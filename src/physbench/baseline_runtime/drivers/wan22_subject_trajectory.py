from __future__ import annotations

from pathlib import Path

from ...baseline_plugins.wan22_subject_trajectory import (
    Wan22SubjectTrajectoryExecutionEngine,
)
from .wan22_subject_motion import Wan22SubjectMotionManagedDriver


class Wan22SubjectTrajectoryManagedDriver(Wan22SubjectMotionManagedDriver):
    """Managed runtime for WAN centroid-trajectory auxiliary training."""

    def _execution_engine_class(self):
        return Wan22SubjectTrajectoryExecutionEngine

    def dependency_paths(self) -> dict[str, Path]:
        root = Path(__file__).resolve().parents[4]
        paths = super().dependency_paths()
        relative = (
            "src/physbench/baseline_runtime/drivers/wan22_subject_trajectory.py",
            "src/physbench/baseline_plugins/wan22_subject_trajectory.py",
            "src/physbench/baselines/wan22_subject_trajectory.py",
            "src/physbench/baselines/wan22_subject_trajectory_model.py",
            "scripts/train_wan22_subject_trajectory.sh",
            "scripts/wan22_subject_trajectory_train.py",
        )
        paths.update({name: root / name for name in relative})
        missing = [
            f"{name}={path}"
            for name, path in paths.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"WAN subject-trajectory dependencies missing: {missing}"
            )
        return paths


__all__ = ["Wan22SubjectTrajectoryManagedDriver"]
