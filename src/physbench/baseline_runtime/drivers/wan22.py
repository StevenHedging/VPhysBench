from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...baseline_plugins.wan22 import Wan22ExecutionEngine
from ...baselines.wan22_media import Wan22MediaAdapter
from ...domain import BaselineTaskInstance
from ...io import sha256_file
from ..driver import ManagedDriver


class Wan22ManagedDriver(ManagedDriver):
    """Managed direct-eval bridge to the shared WAN execution engine."""

    def validate_deployment(self) -> None:
        model = self.bundle.value["model"]
        checkpoint = model.get("frozen_lora_checkpoint")
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"WAN frozen LoRA checkpoint not found: {checkpoint}"
            )
        expected = model.get("checkpoint_sha256")
        if checkpoint and not expected:
            raise ValueError(
                "WAN frozen LoRA checkpoint requires model.checkpoint_sha256"
            )
        if checkpoint:
            actual = sha256_file(checkpoint)
            if actual != expected:
                raise ValueError(
                    "WAN frozen LoRA checkpoint digest mismatch: "
                    f"expected={expected}, actual={actual}"
                )
        python = Path(self.bundle.value["runtime"]["python"])
        if not python.is_file():
            raise FileNotFoundError(
                f"WAN runtime python not found: {python}"
            )

    def dependency_paths(self) -> dict[str, Path]:
        repository_root = Path(__file__).resolve().parents[4]
        paths = {
            "src/physbench/baseline_runtime/drivers/wan22.py": (
                Path(__file__)
            ),
            "src/physbench/baseline_plugins/wan22.py": (
                repository_root
                / "src"
                / "physbench"
                / "baseline_plugins"
                / "wan22.py"
            ),
            "src/physbench/baselines/wan22_lora.py": (
                repository_root
                / "src"
                / "physbench"
                / "baselines"
                / "wan22_lora.py"
            ),
            "src/physbench/baselines/wan22_media.py": (
                repository_root
                / "src"
                / "physbench"
                / "baselines"
                / "wan22_media.py"
            ),
            "scripts/wan22_generate.py": (
                repository_root / "scripts" / "wan22_generate.py"
            ),
            "scripts/wan22_generate_batch.py": (
                repository_root / "scripts" / "wan22_generate_batch.py"
            ),
            "scripts/plot_wan22_loss.py": (
                repository_root / "scripts" / "plot_wan22_loss.py"
            ),
        }
        missing = [
            f"{name}={path}"
            for name, path in paths.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"WAN managed runtime dependencies missing: {missing}"
            )
        return paths

    def _execution_engine_class(self):
        return Wan22ExecutionEngine

    def _adapter_recipe(self) -> dict[str, Any]:
        return self.bundle.value["adapter"]

    def _media_config(self) -> dict[str, Any]:
        adapter = self._adapter_recipe()
        spatial = adapter["spatial"]
        temporal = adapter["temporal"]
        profiles = spatial["scene_profiles"]
        buckets: dict[str, dict[str, Any]] = {}
        for scene_id, profile in profiles.items():
            name = str(profile["name"])
            dimensions = (
                int(profile["width"]),
                int(profile["height"]),
            )
            bucket = buckets.setdefault(name, {
                "width": dimensions[0],
                "height": dimensions[1],
                "scene_ids": [],
            })
            if (
                int(bucket["width"]),
                int(bucket["height"]),
            ) != dimensions:
                raise ValueError(
                    f"WAN profile name {name!r} has conflicting dimensions"
                )
            bucket["scene_ids"].append(scene_id)
        first = next(iter(profiles.values()))
        return {
            "width": int(first["width"]),
            "height": int(first["height"]),
            "spatial_policy": spatial.get(
                "policy",
                "scene_aspect_ratio_bucket_then_fit_and_pad",
            ),
            "pad_color": spatial.get("pad_color", "black"),
            "pad_mode": spatial.get("pad_mode", "edge"),
            "aspect_ratio_buckets": {
                "enabled": True,
                "buckets": buckets,
            },
            "fps": int(temporal["fps"]),
            "max_frames": int(temporal["max_frames"]),
            "min_frames": int(temporal.get("min_frames", 5)),
            "temporal_policy": temporal.get(
                "policy",
                "physical-time prefix; preserve valid 4n+1 frame counts",
            ),
            "cache_policy": adapter.get(
                "cache_policy",
                "content_addressed_shared_immutable",
            ),
        }

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        media_config = self._media_config()
        runtime_adapter = SimpleNamespace(
            media_config=media_config,
            media=Wan22MediaAdapter(media_config),
        )
        task_builder = SimpleNamespace(
            fingerprint=instance.value["identity"]["task_builder"][
                "fingerprint"
            ],
            data_adapter=runtime_adapter,
        )
        engine = self._execution_engine_class()(
            self.bundle,
            task_builder,
        )
        training, predictions = engine.run_task(
            instance=instance,
            run_dir=run_dir,
            execute=execute,
            stop_after_training=stop_after_training,
        )
        for prediction in predictions:
            prediction.pop("text_transform_id", None)
            prediction.pop("evaluation_reference_video", None)
            prediction.pop("visual_reference_video", None)
        return training, predictions
