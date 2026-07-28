from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Any

from ..domain import BaselineBundle, BaselineTaskInstance
from ..io import write_json


class ManagedDriver(ABC):
    """Advanced model hook beneath the managed compiler and DataAdapter."""

    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle

    def validate_deployment(self) -> None:
        """Validate model/runtime identity without loading model weights."""

    def dependency_paths(self) -> dict[str, Path]:
        """Return output-affecting dependencies outside the Bundle."""
        return {}

    @abstractmethod
    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raise NotImplementedError


class DirectManagedDriver(ManagedDriver):
    """Default direct-eval lifecycle; subclasses only adapt and execute jobs."""

    @abstractmethod
    def prepare_job(
        self,
        *,
        job: dict[str, Any],
        case: dict[str, Any],
        adaptation: dict[str, Any],
        source_root: Path,
        run_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute_job(
        self,
        spec: dict[str, Any],
        *,
        log_path: Path,
    ) -> dict[str, Any]:
        """Execute one job and return return_code/command metadata."""
        raise NotImplementedError

    def execute_jobs(
        self,
        specs: list[dict[str, Any]],
        *,
        run_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        results = {}
        for spec in specs:
            result = self.execute_job(
                spec,
                log_path=(
                    run_dir
                    / "logs"
                    / self.bundle.baseline_id
                    / f"{spec['job_id']}.log"
                ),
            )
            results[spec["job_id"]] = result
        return results

    @staticmethod
    def _case_input_view(
        case: dict[str, Any],
        adaptation: dict[str, Any],
    ) -> dict[str, Any]:
        """Expose only contract-declared assets and non-physics metadata."""
        allowed = set(
            adaptation["input_contract"]["asset_access"]
        )
        return {
            key: case[key]
            for key in (
                "schema_version",
                "case_id",
                "scene_id",
                "appearance",
                "temporal",
                "ood",
            )
            if key in case
        } | {
            "assets": {
                key: value
                for key, value in case.get("assets", {}).items()
                if key in allowed
            }
        }

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        value = instance.value
        if value["semantics"]["family"] != "direct_eval":
            raise ValueError(
                "DirectManagedDriver only supports direct_eval"
            )
        source_root = Path(value["source"]["asset_root"])
        cases = {
            case["case_id"]: case for case in value["source"]["cases"]
        }
        adaptations = {
            item["adaptation_id"]: item for item in value["adaptations"]
        }
        conditioning = value["semantics"]["conditioning"]
        training = {
            "operation_id": "train",
            "status": "not_requested",
            "note": "Managed direct-evaluation Baseline.",
        }
        specs = []
        common_by_job: dict[str, dict[str, Any]] = {}
        for job in value["inference"]["jobs"]:
            case = cases[job["case_id"]]
            adaptation = adaptations[job["adaptation_id"]]
            spec = self.prepare_job(
                job=job,
                case=self._case_input_view(case, adaptation),
                adaptation=adaptation,
                source_root=source_root,
                run_dir=run_dir,
            )
            if spec.get("job_id") != job["job_id"]:
                raise ValueError(
                    "managed driver changed or omitted canonical job_id"
                )
            forbidden_spec_fields = {
                "evaluation_reference_video",
                "visual_reference_video",
                "reference_video",
                "physics_reference_video",
            }
            leaked = sorted(forbidden_spec_fields & set(spec))
            if leaked:
                raise ValueError(
                    "managed job spec contains evaluator-only reference "
                    f"fields: {leaked}"
                )
            output = Path(spec["output_video"]).resolve()
            try:
                output.relative_to((run_dir / "predictions").resolve())
            except ValueError as exc:
                raise ValueError(
                    f"managed output must be inside run predictions: {output}"
                ) from exc
            spec_path = run_dir / "jobs" / f"{job['job_id']}.json"
            write_json(spec_path, spec)
            specs.append(spec)
            common_by_job[job["job_id"]] = {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "baseline_id": self.bundle.baseline_id,
                "conditioning": conditioning,
                "evaluation_partition": job["evaluation_partition"],
                "manual_scores": {},
                "job_spec": str(spec_path),
                "seed": int(job["seed"]),
            }
        if stop_after_training:
            return training, [
                {
                    **common_by_job[spec["job_id"]],
                    "status": "staged",
                    "video_path": None,
                }
                for spec in specs
            ]
        if not execute:
            return training, [
                {
                    **common_by_job[spec["job_id"]],
                    "status": "planned",
                    "video_path": None,
                    "note": "Managed generation is planned; execution is disabled.",
                }
                for spec in specs
            ]

        execution = self.execute_jobs(specs, run_dir=run_dir)
        predictions = []
        for spec in specs:
            job_id = spec["job_id"]
            result = execution.get(job_id)
            if not isinstance(result, dict):
                raise ValueError(
                    f"managed driver omitted execution result for {job_id}"
                )
            output = Path(spec["output_video"]).resolve()
            return_code = int(result.get("return_code", 1))
            complete = return_code == 0 and output.is_file()
            protected = {
                "job_id",
                "case_id",
                "baseline_id",
                "conditioning",
                "evaluation_partition",
                "status",
                "video_path",
                "seed",
                "prompt_profile_id",
                "evaluation_reference_video",
                "visual_reference_video",
            }
            collision = sorted(protected & set(result))
            if collision:
                raise ValueError(
                    f"managed driver result overrides protected fields for "
                    f"{job_id}: {collision}"
                )
            predictions.append({
                **common_by_job[job_id],
                **result,
                "status": "complete" if complete else "failed",
                "video_path": str(output) if output.is_file() else None,
                "return_code": return_code,
            })
        return training, predictions


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load managed driver module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_managed_driver(bundle: BaselineBundle) -> ManagedDriver:
    relative = bundle.value["implementation"]["driver"]
    path = (bundle.root / relative).resolve()
    try:
        path.relative_to(bundle.root)
    except ValueError as exc:
        raise ValueError(
            f"managed driver escapes Baseline bundle: {relative}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"managed driver not found: {path}")
    module = _load_module(
        path,
        (
            f"_physbench_driver_{bundle.baseline_id}_"
            f"{bundle.digest[:12]}"
        ),
    )
    driver_type = getattr(module, "Driver", None)
    if not isinstance(driver_type, type) or not issubclass(
        driver_type, ManagedDriver
    ):
        raise TypeError(
            f"managed driver must export Driver subclass: {path}"
        )
    driver = driver_type(bundle)
    driver.validate_deployment()
    return driver
