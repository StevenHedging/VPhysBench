from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..baseline_api.interfaces import BaselinePlugin
from ..domain import BaselineBundle, BaselineTaskInstance
from ..io import sha256_file
from .adapter_loader import load_data_adapter
from .compiler import ManagedTaskBuilder
from .driver import load_managed_driver


def _validate_managed_outputs(
    bundle: BaselineBundle,
    instance: BaselineTaskInstance,
    training: Any,
    predictions: Any,
) -> None:
    """Bind driver outputs back to the frozen inference jobs."""

    if not isinstance(training, dict):
        raise TypeError("managed driver training result must be an object")
    if not isinstance(predictions, list) or any(
        not isinstance(record, dict) for record in predictions
    ):
        raise TypeError(
            "managed driver predictions must be a list of objects"
        )
    expected = {
        job["job_id"]: job
        for job in instance.value["inference"]["jobs"]
    }
    by_job: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        job_id = prediction.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(
                f"managed prediction[{index}] requires a non-empty job_id"
            )
        if job_id in by_job:
            raise ValueError(
                f"managed driver returned duplicate prediction {job_id}"
            )
        by_job[job_id] = prediction
    missing = sorted(set(expected) - set(by_job))
    extra = sorted(set(by_job) - set(expected))
    if missing or extra:
        raise ValueError(
            "managed prediction coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    forbidden = {
        "conditioning",
        "prompt_profile_id",
        "evaluation_reference_video",
        "visual_reference_video",
        "reference_video",
        "physics_reference_video",
    }
    statuses = {"planned", "staged", "complete", "failed"}
    for job_id, job in expected.items():
        prediction = by_job[job_id]
        leaked = sorted(forbidden & set(prediction))
        if leaked:
            raise ValueError(
                f"managed prediction {job_id} exposes legacy/evaluator "
                f"fields: {leaked}"
            )
        expected_identity = {
            "case_id": job["case_id"],
            "baseline_id": bundle.baseline_id,
            "evaluation_partition": job["evaluation_partition"],
            "seed": int(job["seed"]),
        }
        actual_identity = {
            key: prediction.get(key) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"managed prediction identity mismatch for {job_id}: "
                f"expected={expected_identity}, actual={actual_identity}"
            )
        if prediction.get("status") not in statuses:
            raise ValueError(
                f"managed prediction {job_id} has invalid status "
                f"{prediction.get('status')!r}"
            )
        video_path = prediction.get("video_path")
        if video_path is not None and (
            not isinstance(video_path, str) or not video_path
        ):
            raise ValueError(
                f"managed prediction {job_id} video_path must be null or "
                "a non-empty string"
            )


def _seal_prediction_spatial_alignment(
    instance: BaselineTaskInstance,
    predictions: list[dict[str, Any]],
) -> None:
    """Bind driver output to the compiler-sealed I2V spatial contract."""
    jobs = {
        job["job_id"]: job
        for job in instance.value["inference"]["jobs"]
    }
    for prediction in predictions:
        job = jobs[prediction["job_id"]]
        expected = job["native_inputs"].get("spatial_alignment")
        observed = prediction.get("spatial_alignment")
        if expected is None:
            if observed is not None:
                raise ValueError(
                    "driver advertised an unsealed spatial alignment for "
                    f"{prediction['job_id']}"
                )
            continue
        if observed is not None and observed != expected:
            raise ValueError(
                "driver spatial alignment differs from the compiled contract "
                f"for {prediction['job_id']}"
            )
        prediction["spatial_alignment"] = copy.deepcopy(expected)


def _merge_dependency_paths(
    *groups: dict[str, Path],
    label: str,
) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise TypeError(f"{label} dependencies must be an object")
        for name, path in group.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(path, Path)
            ):
                raise TypeError(
                    f"{label} dependencies must map non-empty names to "
                    "Path objects"
                )
            previous = merged.get(name)
            if (
                previous is not None
                and previous.resolve() != path.resolve()
            ):
                raise ValueError(
                    f"{label} dependency {name!r} maps to both "
                    f"{previous} and {path}"
                )
            merged[name] = path
    return merged


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
        "src/physbench/baseline_runtime/bundle_loader.py": (
            root / "bundle_loader.py"
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
    paths = _merge_dependency_paths(
        paths,
        extra,
        label="managed runtime",
    )
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
    task_builder.validate_compiled_instance(instance.value)


class ManagedBaselinePlugin(BaselinePlugin):
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self.data_adapter = load_data_adapter(bundle)
        self.driver = load_managed_driver(bundle)
        driver_paths = self.driver.dependency_paths()
        adapter_paths = self.data_adapter.dependency_paths()
        extra_paths = _merge_dependency_paths(
            driver_paths,
            adapter_paths,
            label="managed driver/DataAdapter",
        )
        dependencies = _runtime_dependencies(
            extra_paths
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
        training, predictions = self.driver.run_task(
            instance=instance,
            run_dir=run_dir,
            execute=execute,
            stop_after_training=stop_after_training,
        )
        _validate_managed_outputs(
            self.bundle,
            instance,
            training,
            predictions,
        )
        _seal_prediction_spatial_alignment(instance, predictions)
        return training, predictions
