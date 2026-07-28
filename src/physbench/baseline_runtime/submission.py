from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import import_prediction_video
from ..baseline_api.interfaces import BaselinePlugin
from ..domain import BaselineBundle, BaselineTaskInstance
from ..io import load_jsonl, sha256_file
from .adapter_loader import load_data_adapter
from .compiler import ManagedTaskBuilder
from .plugin import _runtime_dependencies, verify_managed_instance


class SubmissionBaselinePlugin(BaselinePlugin):
    """Output-only Baseline that imports a complete frozen job submission."""

    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        raw = bundle.value.get("runtime", {}).get(
            "submission_manifest"
        )
        self.submission_manifest = (
            Path(raw).resolve() if raw else None
        )
        extra = {}
        if self.submission_manifest is not None:
            if not self.submission_manifest.is_file():
                raise FileNotFoundError(
                    "submission manifest not found: "
                    f"{self.submission_manifest}"
                )
            extra["external/submission_manifest.jsonl"] = (
                self.submission_manifest
            )
        self.submission_manifest_sha256 = (
            sha256_file(self.submission_manifest)
            if self.submission_manifest is not None
            else None
        )
        self.data_adapter = load_data_adapter(bundle)
        adapter_paths = self.data_adapter.dependency_paths()
        overlap = sorted(set(extra) & set(adapter_paths))
        if overlap:
            raise ValueError(
                "submission and DataAdapter dependencies collide: "
                f"{overlap}"
            )
        self.task_builder = ManagedTaskBuilder(
            bundle,
            _runtime_dependencies(
                {**extra, **adapter_paths},
                submission=True,
            ),
            self.data_adapter,
        )

    def _submission_records(self) -> dict[str, dict[str, Any]]:
        if self.submission_manifest is None:
            raise ValueError(
                "submission Baseline requires runtime.submission_manifest"
            )
        path = self.submission_manifest
        if not path.is_file():
            raise FileNotFoundError(
                f"submission manifest not found: {path}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != self.submission_manifest_sha256:
            raise ValueError(
                "submission manifest changed after Baseline deployment "
                f"was loaded: expected={self.submission_manifest_sha256}, "
                f"actual={actual_digest}"
            )
        records = load_jsonl(path)
        by_job: dict[str, dict[str, Any]] = {}
        for record in records:
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(
                    "submission record requires non-empty job_id"
                )
            if job_id in by_job:
                raise ValueError(
                    f"duplicate submission job_id: {job_id}"
                )
            by_job[job_id] = record
        return by_job

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        verify_managed_instance(self.bundle, self.task_builder, instance)
        value = instance.value
        conditioning = value["semantics"]["conditioning"]
        training = {
            "operation_id": "train",
            "status": "not_requested",
            "note": "Output-only submission Baseline.",
        }
        if stop_after_training or not execute:
            status = "staged" if stop_after_training else "planned"
            return training, [
                {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": self.bundle.baseline_id,
                    "conditioning": conditioning,
                    "evaluation_partition": job["evaluation_partition"],
                    "status": status,
                    "video_path": None,
                    "manual_scores": {},
                    "seed": int(job["seed"]),
                }
                for job in value["inference"]["jobs"]
            ]

        submitted = self._submission_records()
        expected_ids = {
            job["job_id"] for job in value["inference"]["jobs"]
        }
        extra = sorted(set(submitted) - expected_ids)
        missing = sorted(expected_ids - set(submitted))
        if extra or missing:
            raise ValueError(
                "submission coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        predictions = []
        for job in value["inference"]["jobs"]:
            record = submitted[job["job_id"]]
            expected = {
                "case_id": job["case_id"],
                "conditioning": conditioning,
                "seed": int(job["seed"]),
            }
            actual = {
                "case_id": record.get("case_id"),
                "conditioning": record.get("conditioning"),
                "seed": (
                    int(record["seed"])
                    if isinstance(record.get("seed"), int)
                    and not isinstance(record.get("seed"), bool)
                    else record.get("seed")
                ),
            }
            if actual != expected:
                raise ValueError(
                    f"submission identity mismatch for {job['job_id']}: "
                    f"expected={expected}, actual={actual}"
                )
            imported = import_prediction_video(
                source=record["video_path"],
                run_dir=run_dir,
                baseline_id=self.bundle.baseline_id,
                case_id=job["case_id"],
                job_id=job["job_id"],
                seed=int(job["seed"]),
            )
            predictions.append({
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "baseline_id": self.bundle.baseline_id,
                "conditioning": conditioning,
                "evaluation_partition": job["evaluation_partition"],
                "status": "complete",
                "video_path": imported["destination_path"],
                "video_sha256": imported["sha256"],
                "manual_scores": {},
                "seed": int(job["seed"]),
                "import_provenance": imported["manifest_path"],
            })
        return training, predictions
