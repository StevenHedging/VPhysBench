from __future__ import annotations

import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


ARTIFACT_POLICY = {
    "prediction_storage": "run_local",
    "external_prediction_references_allowed": False,
    "external_dependencies_allowed": [
        "model_code",
        "model_weights",
        "rebuildable_cache",
    ],
}

PREDICTION_STATUSES = {"planned", "staged", "complete", "failed"}
FORBIDDEN_PREDICTION_FIELDS = {
    "conditioning",
    "prompt_profile_id",
    "evaluation_reference_video",
    "visual_reference_video",
    "reference_video",
    "physics_reference_video",
}


def _safe_component(value: str, field: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(
            f"{field} must contain only letters, digits, '.', '_' or '-': "
            f"{value!r}"
        )
    return value


def _run_local(path: Path, run_dir: Path, label: str) -> Path:
    resolved = path.resolve()
    root = run_dir.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be stored inside AtomicRun {root}: {resolved}"
        ) from exc
    return resolved


def prediction_artifact_manifest(
    predictions: list[dict[str, Any]],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Validate and fingerprint every prediction video advertised by a plugin."""
    root = Path(run_dir).resolve()
    records = []
    for prediction in predictions:
        status = prediction.get("status")
        value = prediction.get("video_path")
        if value is None:
            if status == "complete":
                raise ValueError(
                    "complete prediction has no run-local video_path: "
                    f"{prediction.get('job_id')}"
                )
            continue
        path = _run_local(
            Path(value),
            root,
            f"prediction video for {prediction.get('job_id')}",
        )
        if not path.is_file():
            raise FileNotFoundError(f"prediction video not found: {path}")
        records.append({
            "baseline_id": prediction.get("baseline_id"),
            "case_id": prediction.get("case_id"),
            "job_id": prediction.get("job_id"),
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "status": status,
        })
    return {
        "schema_version": "1.0",
        "policy": ARTIFACT_POLICY,
        "prediction_videos": records,
    }


def validate_prediction_records(
    predictions: list[dict[str, Any]],
    *,
    jobs: list[dict[str, Any]],
    baseline_id: str,
    run_dir: str | Path,
    expected_artifact_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate prediction identity, coverage, and run-local artifacts.

    This is the common trust boundary used both immediately after a Baseline
    returns and when an existing AtomicRun is re-evaluated.  Re-evaluation may
    never bypass the identity and artifact checks that applied to the original
    execution.
    """

    if not isinstance(predictions, list) or any(
        not isinstance(record, dict) for record in predictions
    ):
        raise TypeError("predictions must be a list of objects")
    if not isinstance(jobs, list) or any(
        not isinstance(job, dict) for job in jobs
    ):
        raise TypeError("jobs must be a list of objects")
    expected = {
        job["job_id"]: job
        for job in jobs
    }
    if len(expected) != len(jobs):
        raise ValueError("frozen inference jobs contain duplicate job_id values")
    actual: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        job_id = prediction.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(
                f"prediction[{index}] requires a non-empty job_id"
            )
        if job_id in actual:
            raise ValueError(f"duplicate prediction record: {job_id}")
        actual[job_id] = prediction
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "prediction coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    for job_id, job in expected.items():
        prediction = actual[job_id]
        leaked = sorted(FORBIDDEN_PREDICTION_FIELDS & set(prediction))
        if leaked:
            raise ValueError(
                f"prediction {job_id} exposes legacy/evaluator fields: "
                f"{leaked}"
            )
        expected_identity = {
            "case_id": job["case_id"],
            "baseline_id": baseline_id,
            "evaluation_partition": job["evaluation_partition"],
            "seed": job["seed"],
        }
        actual_identity = {
            key: prediction.get(key)
            for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"prediction identity mismatch for {job_id}: "
                f"expected={expected_identity}, actual={actual_identity}"
            )
        if prediction.get("status") not in PREDICTION_STATUSES:
            raise ValueError(
                f"prediction {job_id} has invalid status "
                f"{prediction.get('status')!r}"
            )
        expected_spatial_alignment = job.get("native_inputs", {}).get(
            "spatial_alignment"
        )
        observed_spatial_alignment = prediction.get("spatial_alignment")
        if observed_spatial_alignment != expected_spatial_alignment:
            raise ValueError(
                f"prediction {job_id} spatial_alignment differs from the "
                "compiled I2V contract"
            )
        video_path = prediction.get("video_path")
        if video_path is not None and (
            not isinstance(video_path, str) or not video_path
        ):
            raise ValueError(
                f"prediction {job_id} video_path must be null or a "
                "non-empty string"
            )

    manifest = prediction_artifact_manifest(predictions, run_dir)
    artifacts_by_job = {
        record["job_id"]: record
        for record in manifest["prediction_videos"]
    }
    for job_id, prediction in actual.items():
        advertised_digest = prediction.get("video_sha256")
        if advertised_digest is None:
            continue
        artifact = artifacts_by_job.get(job_id)
        if artifact is None or advertised_digest != artifact["sha256"]:
            raise ValueError(
                f"prediction {job_id} video_sha256 does not match the "
                "run-local artifact"
            )

    if expected_artifact_manifest is not None:
        if (
            expected_artifact_manifest.get("schema_version") != "1.0"
            or expected_artifact_manifest.get("policy") != ARTIFACT_POLICY
            or not isinstance(
                expected_artifact_manifest.get("prediction_videos"),
                list,
            )
        ):
            raise ValueError("stored prediction artifact manifest is invalid")
        key = lambda item: (item.get("job_id"), item.get("path"))
        recorded = sorted(
            expected_artifact_manifest["prediction_videos"],
            key=key,
        )
        observed = sorted(manifest["prediction_videos"], key=key)
        if observed != recorded:
            raise ValueError(
                "prediction artifacts differ from the frozen AtomicRun "
                "manifest"
            )
    return manifest


def import_prediction_video(
    *,
    source: str | Path,
    run_dir: str | Path,
    baseline_id: str,
    case_id: str,
    job_id: str,
    seed: int,
) -> dict[str, Any]:
    """Copy an existing external prediction into a run-owned artifact tree."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"prediction import source not found: {source_path}")
    baseline_id = _safe_component(baseline_id, "baseline_id")
    _safe_component(case_id, "case_id")
    job_id = _safe_component(job_id, "job_id")
    root = Path(run_dir).resolve()
    suffix = source_path.suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise ValueError(f"unsupported prediction video extension: {suffix}")
    destination = (
        root / "predictions" / baseline_id / f"{job_id}{suffix}"
    )
    _run_local(destination, root, "prediction import destination")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_digest = sha256_file(source_path)
    if destination.exists():
        if not destination.is_file():
            raise ValueError(
                f"prediction import destination is not a file: {destination}"
            )
        destination_digest = sha256_file(destination)
        if destination_digest != source_digest:
            raise FileExistsError(
                "prediction import destination already exists with different "
                f"content: {destination}"
            )
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".importing",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source_path, temporary)
            if sha256_file(temporary) != source_digest:
                raise RuntimeError(
                    f"prediction import digest mismatch while copying {source_path}"
                )
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    record = {
        "baseline_id": baseline_id,
        "case_id": case_id,
        "destination": str(destination.relative_to(root)),
        "job_id": job_id,
        "seed": int(seed),
        "sha256": source_digest,
        "size_bytes": destination.stat().st_size,
        "source_path": str(source_path),
    }
    manifest_path = root / "provenance" / "prediction_imports.json"
    manifest = (
        load_json(manifest_path)
        if manifest_path.is_file()
        else {
            "schema_version": "1.0",
            "policy": ARTIFACT_POLICY,
            "imports": [],
        }
    )
    imports = manifest.get("imports")
    if not isinstance(imports, list):
        raise ValueError(
            f"prediction import manifest has invalid imports: {manifest_path}"
        )
    existing = next(
        (
            item
            for item in imports
            if item.get("destination") == record["destination"]
        ),
        None,
    )
    if existing is not None:
        comparable = {
            key: existing.get(key)
            for key in record
        }
        if comparable != record:
            raise ValueError(
                "prediction import provenance conflicts with existing record: "
                f"{record['destination']}"
            )
        result = existing
    else:
        result = {
            **record,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        imports.append(result)
        imports.sort(key=lambda item: item["destination"])
        manifest["imports"] = imports
        write_json(manifest_path, manifest)
    return {
        **result,
        "destination_path": str(destination),
        "manifest_path": str(manifest_path),
    }
