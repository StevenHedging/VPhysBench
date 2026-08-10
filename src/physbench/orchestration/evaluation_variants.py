from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifacts import validate_prediction_records
from ..datasets import load_dataset
from ..domain import BaselineTaskInstance
from ..evaluation import (
    aggregate_task_results,
    evaluate_task,
    load_evaluation_protocol,
)
from ..evaluation.contracts import CaseEvaluationResult
from ..identifiers import require_safe_id
from ..io import (
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)
from ..tasks import load_task


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORE_SOURCE_FILES = (
    "plan.json",
    "component_fingerprints.json",
    "run.json",
    "state.json",
    "report.md",
    "frozen/dataset.json",
    "frozen/task.json",
    "frozen/cases.jsonl",
    "frozen/assets.lock.json",
    "task_instance/manifest.json",
    "predictions.jsonl",
    "artifacts/prediction_artifacts.json",
    "evaluation/manifest.json",
    "evaluation/case_results.jsonl",
    "evaluation/task_result.json",
    "evaluation/case_metrics.jsonl",
    "evaluation/summary.json",
)
_DEPENDENCY_DISTRIBUTIONS = (
    "physics-video-benchmark",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "torch",
    "matplotlib",
    "transformers",
    "huggingface-hub",
    "sam-2",
)
_SAM_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_file(root: Path, relative: str) -> Path:
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"AtomicRun source path must not contain symlinks: {relative}"
            )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"AtomicRun source file is missing or escapes the run: {relative}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"AtomicRun source is not a regular file: {relative}")
    return resolved


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_component_fingerprints(
    root: Path,
    *,
    instance: BaselineTaskInstance,
    task_digest: str,
) -> dict[str, str]:
    components = load_json(_run_file(root, "component_fingerprints.json"))
    value = instance.value
    identity = value["identity"]
    expected = {
        "dataset": identity["dataset"]["digest"],
        "task": task_digest,
        "baseline": identity["baseline"]["digest"],
        "baseline_deployment": identity["baseline"]["deployment_digest"],
        "task_builder": identity["task_builder"]["fingerprint"],
        "task_instance": instance.digest,
        "data_adapter": identity["data_adapter"]["fingerprint"],
        "data_adapter_materialization": identity["data_adapter"][
            "materialization_fingerprint"
        ],
    }
    mismatched = {
        name: {
            "expected": digest,
            "actual": components.get(name),
        }
        for name, digest in expected.items()
        if components.get(name) != digest
    }
    if mismatched:
        raise ValueError(
            "AtomicRun component fingerprints do not match the sealed "
            f"TaskInstance: {mismatched}"
        )
    protocol_fingerprint = components.get("evaluation_protocol")
    if (
        not isinstance(protocol_fingerprint, str)
        or _SHA256_RE.fullmatch(protocol_fingerprint) is None
    ):
        raise ValueError(
            "AtomicRun component_fingerprints.evaluation_protocol is invalid"
        )
    return components


def _validate_run_identity(
    root: Path,
    *,
    instance: BaselineTaskInstance,
    task_result: dict[str, Any],
) -> dict[str, Any]:
    run = load_json(_run_file(root, "run.json"))
    identity = instance.value["identity"]
    expected = {
        "dataset_id": identity["dataset"]["dataset_id"],
        "dataset_digest": identity["dataset"]["digest"],
        "task_id": identity["task"]["task_id"],
        "baseline_id": identity["baseline"]["baseline_id"],
        "baseline_digest": identity["baseline"]["digest"],
        "baseline_deployment_digest": identity["baseline"][
            "deployment_digest"
        ],
        "task_instance_id": instance.instance_id,
        "task_instance_digest": instance.digest,
        "evaluation_status": task_result["status"],
        "evaluation_coverage": task_result["coverage"],
        "evaluation_score": task_result["score"],
    }
    mismatched = {
        name: {"expected": value, "actual": run.get(name)}
        for name, value in expected.items()
        if run.get(name) != value
    }
    if mismatched:
        raise ValueError(
            f"AtomicRun run.json identity/evaluation mismatch: {mismatched}"
        )
    return run


def _validate_cases(
    cases: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"frozen cases[{index}] requires a non-empty case_id"
            )
        if case_id in by_id:
            raise ValueError(f"frozen cases contain duplicate case_id: {case_id}")
        by_id[case_id] = case
    for job in plan["jobs"]:
        case = by_id.get(job["case_id"])
        if case is None:
            raise ValueError(
                f"frozen cases omit planned case {job['case_id']}"
            )
        if case.get("scene_id") != job["scene_id"]:
            raise ValueError(
                f"frozen case scene mismatch for job {job['job_id']}"
            )
    return by_id


def _authenticate_frozen_dataset(
    root: Path,
    *,
    instance: BaselineTaskInstance,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Bind frozen Dataset files to the sealed original release digest."""
    descriptor = load_json(_run_file(root, "frozen/dataset.json"))
    cases = load_jsonl(_run_file(root, "frozen/cases.jsonl"))
    views = load_json(_run_file(root, "frozen/views.json"))
    asset_lock = load_json(_run_file(root, "frozen/assets.lock.json"))
    release = descriptor.get("release")
    if (
        not isinstance(release, str)
        or not release
        or Path(release).name != release
        or release in {".", ".."}
    ):
        raise ValueError("frozen Dataset release is not path-safe")

    source_asset_root = Path(
        instance.value["source"]["asset_root"]
    ).resolve(strict=True)
    if not source_asset_root.is_dir():
        raise ValueError(
            f"sealed Dataset asset_root is not a directory: {source_asset_root}"
        )
    candidate = (
        source_asset_root / "releases" / release / "dataset.json"
    )
    try:
        source_descriptor = candidate.resolve(strict=True)
        source_descriptor.relative_to(source_asset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "cannot authenticate frozen Dataset: the sealed release descriptor "
            f"is missing or escapes asset_root: {candidate}"
        ) from exc
    if not source_descriptor.is_file():
        raise ValueError(
            "cannot authenticate frozen Dataset: release descriptor is not a "
            f"regular file: {source_descriptor}"
        )

    source = load_dataset(source_descriptor)
    identity = instance.value["identity"]["dataset"]
    if source.dataset_id != identity["dataset_id"]:
        raise ValueError(
            "source Dataset ID differs from sealed TaskInstance identity"
        )
    if source.digest != identity["digest"]:
        raise ValueError(
            "source Dataset digest differs from sealed TaskInstance identity"
        )
    if source.asset_root.resolve() != source_asset_root:
        raise ValueError(
            "source Dataset resolves a different asset_root than the sealed "
            "TaskInstance"
        )
    frozen_documents = {
        "descriptor": descriptor,
        "cases": cases,
        "views": views,
        "asset_lock": asset_lock,
    }
    source_documents = {
        "descriptor": source.descriptor,
        "cases": list(source.cases),
        "views": source.views,
        "asset_lock": source.asset_lock,
    }
    mismatched = [
        name
        for name in frozen_documents
        if frozen_documents[name] != source_documents[name]
    ]
    if mismatched:
        raise ValueError(
            "frozen Dataset files differ from the sealed source release: "
            f"{mismatched}"
        )
    authentication = {
        "policy": "recomputed_release_digest_matches_sealed_task_instance",
        "source_descriptor": str(source_descriptor),
        "source_descriptor_sha256": sha256_file(source_descriptor),
        "dataset_id": source.dataset_id,
        "release": release,
        "dataset_digest": source.digest,
        "frozen_documents": {
            name: canonical_sha256(value)
            for name, value in frozen_documents.items()
        },
    }
    return cases, descriptor, asset_lock, authentication


def _validate_native_evaluation(
    root: Path,
    *,
    plan: dict[str, Any],
    predictions: list[dict[str, Any]],
    native_protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    case_results = load_jsonl(
        _run_file(root, "evaluation/case_results.jsonl")
    )
    task_result = load_json(_run_file(root, "evaluation/task_result.json"))
    manifest = load_json(_run_file(root, "evaluation/manifest.json"))
    protocol_identity = {
        "id": native_protocol["protocol_id"],
        "fingerprint": native_protocol["fingerprint"],
    }
    actual_protocol = task_result.get("protocol", {})
    if {
        "id": actual_protocol.get("id"),
        "fingerprint": actual_protocol.get("fingerprint"),
    } != protocol_identity:
        raise ValueError(
            "native evaluation task_result protocol does not match the "
            "frozen Task protocol"
        )
    manifest_protocol = manifest.get("protocol", {})
    if {
        "id": manifest_protocol.get("id"),
        "fingerprint": manifest_protocol.get("fingerprint"),
    } != protocol_identity:
        raise ValueError(
            "native evaluation manifest protocol does not match the "
            "frozen Task protocol"
        )
    expected_jobs = {job["job_id"]: job for job in plan["jobs"]}
    if len(expected_jobs) != len(plan["jobs"]):
        raise ValueError("canonical plan contains duplicate job_id values")
    actual_jobs: dict[str, dict[str, Any]] = {}
    result_contract_fields = {
        "job_id",
        "case_id",
        "scene_id",
        "evaluator",
        "status",
        "score",
        "reason_code",
        "reason",
        "metrics",
        "quality",
        "artifacts",
        "provenance",
    }
    for index, result in enumerate(case_results):
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(
                f"native case result[{index}] requires a non-empty job_id"
            )
        if job_id in actual_jobs:
            raise ValueError(
                f"native case results contain duplicate job_id: {job_id}"
            )
        expected_fields = result_contract_fields | {
            "evaluation_partition",
            "seed",
        }
        if set(result) != expected_fields:
            raise ValueError(
                f"native case result fields differ from the contract for "
                f"{job_id}: missing={sorted(expected_fields - set(result))}, "
                f"extra={sorted(set(result) - expected_fields)}"
            )
        payload = {
            name: result[name]
            for name in result_contract_fields
        }
        validated = CaseEvaluationResult.from_dict(payload).to_dict()
        if validated != payload:
            raise ValueError(
                f"native case result does not round-trip its contract: {job_id}"
            )
        actual_jobs[job_id] = result
    missing = sorted(set(expected_jobs) - set(actual_jobs))
    extra = sorted(set(actual_jobs) - set(expected_jobs))
    if missing or extra:
        raise ValueError(
            "native case-result coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if [result["job_id"] for result in case_results] != [
        job["job_id"] for job in plan["jobs"]
    ]:
        raise ValueError(
            "native case results are not in canonical plan order"
        )
    for job_id, job in expected_jobs.items():
        result = actual_jobs[job_id]
        expected_identity = {
            "case_id": job["case_id"],
            "scene_id": job["scene_id"],
            "evaluation_partition": job["evaluation_partition"],
            "seed": job["seed"],
        }
        actual_identity = {
            name: result.get(name) for name in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"native case-result identity mismatch for {job_id}: "
                f"expected={expected_identity}, actual={actual_identity}"
            )
        case_artifact = load_json(
            _run_file(
                root,
                f"evaluation/cases/{job_id}/result.json",
            )
        )
        if case_artifact != result:
            raise ValueError(
                f"native per-case result artifact differs for {job_id}"
            )
    planned_job_ids = manifest.get("planned_job_ids")
    if (
        not isinstance(planned_job_ids, list)
        or planned_job_ids != [job["job_id"] for job in plan["jobs"]]
        or len(planned_job_ids) != len(set(planned_job_ids))
    ):
        raise ValueError(
            "native evaluation manifest planned_job_ids are incomplete"
        )
    if manifest.get("prediction_records") != len(predictions):
        raise ValueError(
            "native evaluation manifest prediction record count mismatch"
        )
    if task_result.get("integrity_issues") != []:
        raise ValueError(
            "native evaluation contains unresolved integrity issues"
        )
    expected_task_result = {
        "schema_version": "1.0",
        "task_id": plan["task_id"],
        "task_family": plan["family"],
        "protocol": task_result["protocol"],
        "integrity_issues": [],
        **aggregate_task_results(
            plan=plan,
            case_results=case_results,
            general_metrics=native_protocol.get("general_metrics"),
        ),
    }
    if task_result != expected_task_result:
        raise ValueError(
            "native task_result does not equal a fresh aggregation of "
            "case_results"
        )
    if manifest.get("protocol") != task_result["protocol"]:
        raise ValueError(
            "native evaluation manifest protocol differs from task_result"
        )
    compatibility_results = load_jsonl(
        _run_file(root, "evaluation/case_metrics.jsonl")
    )
    compatibility_summary = load_json(
        _run_file(root, "evaluation/summary.json")
    )
    if compatibility_results != case_results:
        raise ValueError(
            "native case_metrics.jsonl differs from case_results.jsonl"
        )
    if compatibility_summary != task_result:
        raise ValueError(
            "native summary.json differs from task_result.json"
        )
    return case_results, task_result, manifest


def _load_asset_lock(
    root: Path,
    *,
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(_run_file(root, "frozen/assets.lock.json"))
    if lock.get("schema_version") != "1.0":
        raise ValueError("frozen asset lock must use schema_version=1.0")
    for key in ("dataset_id", "release"):
        if lock.get(key) != dataset.get(key):
            raise ValueError(f"frozen asset lock {key} mismatch")
    files = lock.get("files")
    if not isinstance(files, list) or any(
        not isinstance(item, dict) for item in files
    ):
        raise ValueError("frozen asset lock files must be an array of objects")
    if lock.get("files_digest") != canonical_sha256(files):
        raise ValueError("frozen asset lock files_digest mismatch")
    by_path: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(files):
        value = item.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"frozen asset lock files[{index}].path is invalid")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"frozen asset lock path is unsafe: {value}")
        if value in by_path:
            raise ValueError(f"frozen asset lock contains duplicate path: {value}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 0
        ):
            raise ValueError(
                f"frozen asset lock digest/size is invalid: {value}"
            )
        by_path[value] = item
    return lock, by_path


def _reference_binding(
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> tuple[str, str, str | None]:
    scene_id = case.get("scene_id")
    assets = case.get("assets", {})
    child_value = assets.get("physics_reference_video")
    force_parent = (
        scene_id == "pendulum"
        and case.get("ood", {}).get("level") == "ood1"
    )
    if scene_id == "pendulum" and not child_value:
        raise ValueError(
            f"pendulum case {case.get('case_id')} has no physics reference"
        )
    if (
        not force_parent
        and child_value
        and case.get("has_real_reference_video", False)
    ):
        return str(child_value), "same_case_reference", None
    parent_id = case.get("provenance", {}).get("parent_case_id")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError(
            f"case {case.get('case_id')} has no trustworthy reference parent"
        )
    parent = catalog.get(parent_id)
    if parent is None:
        raise ValueError(
            f"case {case.get('case_id')} reference parent is missing: {parent_id}"
        )
    if parent.get("scene_id") != scene_id:
        raise ValueError(
            f"case {case.get('case_id')} reference parent scene mismatch"
        )
    if parent.get("physics") != case.get("physics"):
        raise ValueError(
            f"case {case.get('case_id')} reference parent physics mismatch"
        )
    value = parent.get("assets", {}).get("physics_reference_video")
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference parent {parent_id} has no physics video")
    if force_parent and child_value != value:
        raise ValueError(
            f"pendulum OOD case {case.get('case_id')} does not bind its "
            "parent physics reference"
        )
    return value, "parent_physics_reference", parent_id


def _validate_reference_assets(
    *,
    plan: dict[str, Any],
    predictions: list[dict[str, Any]],
    protocol: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
    locked_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = asset_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"AtomicRun asset_root is not a directory: {root}")
    predictions_by_job = {
        prediction["job_id"]: prediction
        for prediction in predictions
    }
    grouped: dict[str, dict[str, Any]] = {}
    for job in plan["jobs"]:
        evaluator_type = (
            protocol["scenes"]
            .get(job["scene_id"], {"type": "unsupported"})
            .get("type", "unsupported")
        )
        prediction = predictions_by_job[job["job_id"]]
        if (
            evaluator_type == "unsupported"
            or prediction.get("status") != "complete"
        ):
            continue
        case = catalog[job["case_id"]]
        value, mode, parent_id = _reference_binding(case, catalog)
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"reference path is unsafe: {value}")
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"reference asset is missing or escapes asset_root: {value}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"reference asset is not a regular file: {value}")
        locked = locked_by_path.get(value)
        if locked is None:
            raise ValueError(
                f"reference asset is absent from the frozen asset lock: {value}"
            )
        roles = locked.get("roles")
        if (
            not isinstance(roles, list)
            or "physics_reference_video" not in roles
        ):
            raise ValueError(
                f"reference asset lacks physics_reference_video role: {value}"
            )
        locked_case_ids = locked.get("case_ids")
        required_case_ids = {job["case_id"]}
        if parent_id is not None:
            required_case_ids.add(parent_id)
        if (
            not isinstance(locked_case_ids, list)
            or not required_case_ids <= set(locked_case_ids)
        ):
            raise ValueError(
                f"reference asset lock case binding mismatch: {value}"
            )
        size = path.stat().st_size
        if size != locked["size_bytes"]:
            raise ValueError(
                f"reference asset size differs from frozen lock: {value}"
            )
        entry = grouped.setdefault(
            value,
            {
                "path": value,
                "resolved_path": str(path),
                "sha256": locked["sha256"],
                "size_bytes": size,
                "bindings": [],
            },
        )
        entry["bindings"].append({
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "mode": mode,
            "parent_case_id": parent_id,
        })
    for value, entry in sorted(grouped.items()):
        actual = sha256_file(entry["resolved_path"])
        if actual != entry["sha256"]:
            raise ValueError(
                f"reference asset SHA-256 differs from frozen lock: {value}"
            )
        entry["bindings"].sort(key=lambda item: item["job_id"])
    records = [grouped[value] for value in sorted(grouped)]
    return {
        "schema_version": "1.0",
        "asset_root": str(root),
        "unique_reference_assets": len(records),
        "records": records,
        "records_digest": canonical_sha256(records),
    }


def _source_file_manifest(root: Path) -> dict[str, Any]:
    paths = {_run_file(root, relative) for relative in _CORE_SOURCE_FILES}
    for relative_root in ("frozen", "task_instance", "evaluation"):
        directory = root / relative_root
        if directory.is_symlink():
            raise ValueError(
                f"AtomicRun source path must not be a symlink: {relative_root}"
            )
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    "AtomicRun source tree must not contain symlinks: "
                    f"{path.relative_to(root)}"
                )
            if path.is_file():
                paths.add(path.resolve(strict=True))
    records = [
        _file_record(path, root=root)
        for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix())
    ]
    return {
        "files": records,
        "files_digest": canonical_sha256(records),
    }


def _git_metadata(project_root: Path) -> dict[str, Any]:
    command = ["git", "-C", str(project_root)]
    try:
        commit = subprocess.run(
            [*command, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status_output = subprocess.run(
            [
                *command,
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/physbench/evaluation",
                "src/physbench/orchestration/evaluation_variants.py",
                "configs/evaluation/protocols",
                "schemas/v2/evaluation_protocol.schema.json",
                "schemas/v3/evaluation_protocol.schema.json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    scoped_status = [
        line for line in status_output.splitlines() if line.strip()
    ]
    return {
        "available": True,
        "commit": commit,
        "scoped_dirty": bool(scoped_status),
        "scoped_status": scoped_status,
    }


def _evaluator_source_manifest(project_root: Path) -> dict[str, Any]:
    evaluator_root = project_root / "src" / "physbench" / "evaluation"
    paths = sorted(evaluator_root.rglob("*.py"))
    paths.extend([
        project_root
        / "src"
        / "physbench"
        / "orchestration"
        / "evaluation_variants.py",
        project_root
        / "schemas"
        / "v2"
        / "evaluation_protocol.schema.json",
        project_root
        / "schemas"
        / "v3"
        / "evaluation_protocol.schema.json",
    ])
    records = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(paths))
    ]
    return {
        "tree_sha256": canonical_sha256(records),
        "files": records,
        "git": _git_metadata(project_root),
    }


def _dependency_manifest() -> dict[str, Any]:
    distributions: dict[str, str | None] = {}
    for name in _DEPENDENCY_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "distributions": distributions,
    }


def _huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get(
        "HF_HUB_CACHE"
    )
    if explicit:
        roots.append(Path(explicit).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _sam_model_integrity(protocol: dict[str, Any]) -> dict[str, Any]:
    model_ids = sorted({
        str(config["sam2"]["model_id"])
        for config in protocol["scenes"].values()
        if isinstance(config, dict)
        and isinstance(config.get("sam2"), dict)
        and config["sam2"].get("model_id")
    })
    records: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_slug = f"models--{model_id.replace('/', '--')}"
        candidates: list[dict[str, Any]] = []
        for cache_root in _huggingface_cache_roots():
            model_root = cache_root / model_slug
            if not model_root.is_dir():
                continue
            ref_main: str | None = None
            ref_path = model_root / "refs" / "main"
            if ref_path.is_file():
                ref_main = ref_path.read_text(encoding="utf-8").strip()
            snapshots = model_root / "snapshots"
            if not snapshots.is_dir():
                continue
            for snapshot in sorted(snapshots.iterdir()):
                if not snapshot.is_dir():
                    continue
                weights = []
                for path in sorted(snapshot.rglob("*")):
                    if (
                        path.is_file()
                        and path.suffix.lower() in _SAM_WEIGHT_SUFFIXES
                    ):
                        weights.append({
                            "path": path.relative_to(snapshot).as_posix(),
                            "resolved_path": str(path.resolve()),
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        })
                candidates.append({
                    "cache_root": str(cache_root),
                    "snapshot_path": str(snapshot.resolve()),
                    "snapshot_revision": snapshot.name,
                    "selected_by_main_ref": snapshot.name == ref_main,
                    "main_ref": ref_main,
                    "weights": weights,
                    "weights_digest": canonical_sha256(weights),
                })
        selected = [
            item for item in candidates if item["selected_by_main_ref"]
        ]
        if len(selected) == 1:
            resolution = "main_ref_resolved"
        elif len(candidates) == 1:
            resolution = "single_local_snapshot"
        elif candidates:
            resolution = "ambiguous_local_snapshots"
        else:
            resolution = "not_available_locally"
        records.append({
            "model_id": model_id,
            "resolution": resolution,
            "local_snapshots": candidates,
        })
    return {
        "models": records,
        "models_digest": canonical_sha256(records),
    }


def _source_integrity(
    *,
    root: Path,
    run: dict[str, Any],
    native_protocol: dict[str, Any],
    target_protocol: dict[str, Any],
    source_files: dict[str, Any],
    dataset_authentication: dict[str, Any],
    prediction_artifacts: dict[str, Any],
    references: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "captured_at": _utc_now(),
        "run": {
            "run_id": run["run_id"],
            "path": str(root),
        },
        "native_protocol": {
            "id": native_protocol["protocol_id"],
            "fingerprint": native_protocol["fingerprint"],
        },
        "target_protocol": {
            "id": target_protocol["protocol_id"],
            "fingerprint": target_protocol["fingerprint"],
        },
        "atomic_run_source": source_files,
        "dataset_authentication": dataset_authentication,
        "prediction_authentication": {
            "policy": (
                "recomputed_run_local_artifacts_match_canonical_manifest"
            ),
            "authenticity_boundary": (
                "self_consistent_runtime_output_not_externally_sealed"
            ),
            "manifest_digest": canonical_sha256(prediction_artifacts),
            "prediction_videos": len(
                prediction_artifacts["prediction_videos"]
            ),
        },
        "native_evaluation_authentication": {
            "policy": (
                "case_contract_per_case_projection_and_task_aggregation_recomputed"
            ),
            "authenticity_boundary": (
                "self_consistent_runtime_output_not_externally_sealed"
            ),
        },
        "reference_assets": references,
        "evaluator_source": _evaluator_source_manifest(PROJECT_ROOT),
        "dependencies": _dependency_manifest(),
        "sam2": _sam_model_integrity(target_protocol),
    }


def _candidate_variant_path(
    root: Path,
    *,
    protocol_id: str,
    protocol_fingerprint: str,
    evaluation_id: str,
) -> Path:
    require_safe_id(protocol_id, label="protocol_id")
    require_safe_id(evaluation_id, label="evaluation_id")
    if _SHA256_RE.fullmatch(protocol_fingerprint) is None:
        raise ValueError("protocol fingerprint must be a lowercase SHA-256")
    return (
        root
        / "reevaluations"
        / protocol_id
        / protocol_fingerprint
        / evaluation_id
    )


def _check_variant_path(root: Path, destination: Path) -> None:
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("reevaluation destination escapes AtomicRun") from exc
    current = root
    for part in destination.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(
                "reevaluation destination path must not contain symlinks: "
                f"{current.relative_to(root)}"
            )
    if os.path.lexists(destination):
        raise FileExistsError(
            f"reevaluation destination already exists: {destination}"
        )


def _exclusive_create_variant(root: Path, destination: Path) -> None:
    current = root
    parts = destination.relative_to(root).parts
    for part in parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise ValueError(
                    "reevaluation parent is not a safe directory: "
                    f"{current.relative_to(root)}"
                )
        else:
            os.mkdir(current)
        resolved = current.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "reevaluation parent escapes AtomicRun: "
                f"{current.relative_to(root)}"
            ) from exc
    try:
        os.mkdir(destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"reevaluation destination already exists: {destination}"
        ) from exc


def _artifact_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    records = []
    for path in sorted(root.rglob("*")):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise ValueError(
                f"reevaluation artifact must not be a symlink: {path}"
            )
        if path.is_file():
            records.append(_file_record(path, root=root))
    return {
        "schema_version": "1.0",
        "scope": "all regular files below variant root except this manifest",
        "files": records,
        "files_digest": canonical_sha256(records),
    }


def _render_report(record: dict[str, Any], summary: dict[str, Any]) -> str:
    score = "—" if summary["score"] is None else f"{summary['score']:.4f}"
    return "".join([
        f"# Evaluation variant: {record['evaluation_id']}\n\n",
        f"- Canonical run: `{record['run_id']}`\n",
        f"- Protocol: `{record['protocol']['id']}` / "
        f"`{record['protocol']['fingerprint']}`\n",
        f"- Workflow status: `{record['workflow_status']}`\n",
        f"- Evaluation status: `{summary['status']}`\n",
        f"- Coverage: `{summary['coverage']:.4f}`\n",
        f"- Score: `{score}`\n\n",
        "This directory is append-only relative to the canonical AtomicRun. "
        "The canonical evaluation, run metadata, report, state, and component "
        "fingerprints were not modified.\n",
    ])


def reevaluate_atomic_variant(
    run_dir: str | Path,
    *,
    protocol_id: str,
    evaluation_id: str,
    protocol_root: str | Path | None = None,
    save_visualizations: bool = False,
) -> dict[str, Any]:
    """Create an immutable, coexisting evaluation of a sealed AtomicRun."""
    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"AtomicRun is not a directory: {root}")
    target_protocol = load_evaluation_protocol(
        protocol_id,
        **(
            {"protocol_root": protocol_root}
            if protocol_root is not None
            else {}
        ),
    )
    destination = _candidate_variant_path(
        root,
        protocol_id=target_protocol["protocol_id"],
        protocol_fingerprint=target_protocol["fingerprint"],
        evaluation_id=evaluation_id,
    )
    _check_variant_path(root, destination)

    instance_document = load_json(
        _run_file(root, "task_instance/manifest.json")
    )
    sealed_instance = BaselineTaskInstance.from_document(instance_document)
    instance = sealed_instance.value
    plan = load_json(_run_file(root, "plan.json"))
    if plan != instance["canonical_plan"]:
        raise ValueError(
            "AtomicRun plan differs from the sealed BaselineTaskInstance"
        )
    task_path = _run_file(root, "frozen/task.json")
    task = load_task(task_path)
    if task.digest != instance["identity"]["task"]["digest"]:
        raise ValueError(
            "AtomicRun task differs from the sealed BaselineTaskInstance"
        )
    components = _load_component_fingerprints(
        root,
        instance=sealed_instance,
        task_digest=task.digest,
    )
    native_protocol_id = task.value.get("evaluation", {}).get(
        "protocol", "scene_default_v1"
    )
    native_protocol = load_evaluation_protocol(
        native_protocol_id,
        **(
            {"protocol_root": protocol_root}
            if protocol_root is not None
            else {}
        ),
    )
    if (
        native_protocol["fingerprint"]
        != components["evaluation_protocol"]
    ):
        raise ValueError(
            "native protocol fingerprint differs from "
            "component_fingerprints.evaluation_protocol"
        )
    (
        cases,
        dataset,
        frozen_asset_lock,
        dataset_authentication,
    ) = _authenticate_frozen_dataset(
        root,
        instance=sealed_instance,
    )
    catalog = _validate_cases(cases, plan)
    predictions = load_jsonl(_run_file(root, "predictions.jsonl"))
    prediction_artifacts = load_json(
        _run_file(root, "artifacts/prediction_artifacts.json")
    )
    observed_prediction_artifacts = validate_prediction_records(
        predictions,
        jobs=instance["inference"]["jobs"],
        baseline_id=instance["identity"]["baseline"]["baseline_id"],
        run_dir=root,
        expected_artifact_manifest=prediction_artifacts,
    )
    _, native_task_result, _ = _validate_native_evaluation(
        root,
        plan=plan,
        predictions=predictions,
        native_protocol=native_protocol,
    )
    run = _validate_run_identity(
        root,
        instance=sealed_instance,
        task_result=native_task_result,
    )
    _, locked_by_path = _load_asset_lock(root, dataset=dataset)
    if frozen_asset_lock.get("files") is None:
        raise ValueError("authenticated frozen asset lock has no files")
    references = _validate_reference_assets(
        plan=plan,
        predictions=predictions,
        protocol=target_protocol,
        catalog=catalog,
        asset_root=Path(instance["source"]["asset_root"]),
        locked_by_path=locked_by_path,
    )
    source_files = _source_file_manifest(root)
    source_integrity = _source_integrity(
        root=root,
        run=run,
        native_protocol=native_protocol,
        target_protocol=target_protocol,
        source_files=source_files,
        dataset_authentication=dataset_authentication,
        prediction_artifacts=observed_prediction_artifacts,
        references=references,
    )

    _exclusive_create_variant(root, destination)
    protocol_document = {
        key: value
        for key, value in target_protocol.items()
        if key not in {"path", "fingerprint"}
    }
    write_json(destination / "protocol.json", protocol_document)
    write_json(destination / "source_integrity.json", source_integrity)
    write_json(destination / "reference_assets.json", references)
    protocol_snapshot = {
        **protocol_document,
        "path": str((destination / "protocol.json").resolve()),
        "fingerprint": target_protocol["fingerprint"],
    }
    record = {
        "schema_version": "1.0",
        "evaluation_id": evaluation_id,
        "run_id": run["run_id"],
        "run_path": str(root),
        "variant_path": str(destination),
        "workflow_status": "running",
        "save_visualizations": bool(save_visualizations),
        "created_at": _utc_now(),
        "native_protocol": {
            "id": native_protocol["protocol_id"],
            "fingerprint": native_protocol["fingerprint"],
        },
        "protocol": {
            "id": target_protocol["protocol_id"],
            "fingerprint": target_protocol["fingerprint"],
            "snapshot": "protocol.json",
        },
        "source_integrity": "source_integrity.json",
        "reference_assets": "reference_assets.json",
        "artifact_manifest": "artifact_manifest.json",
        "canonical_mutation_policy": "forbidden",
    }
    write_json(destination / "reevaluation.json", record)
    try:
        _, summary = evaluate_task(
            plan=plan,
            cases=cases,
            predictions=predictions,
            asset_root=instance["source"]["asset_root"],
            protocol=protocol_snapshot,
            output_dir=destination / "evaluation",
            run_id=run["run_id"],
            save_visualizations=save_visualizations,
        )
        record.update({
            "workflow_status": "complete",
            "evaluation_status": summary["status"],
            "evaluation_coverage": summary["coverage"],
            "evaluation_score": summary["score"],
            "completed_at": _utc_now(),
        })
        write_json(destination / "reevaluation.json", record)
        (destination / "report.md").write_text(
            _render_report(record, summary),
            encoding="utf-8",
        )
    except BaseException as exc:
        record.update({
            "workflow_status": "failed",
            "failed_at": _utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        })
        write_json(destination / "reevaluation.json", record)
        write_json(
            destination / "artifact_manifest.json",
            _artifact_manifest(destination),
        )
        raise
    artifact_manifest = _artifact_manifest(destination)
    write_json(destination / "artifact_manifest.json", artifact_manifest)
    return {
        "variant_path": str(destination),
        "reevaluation": record,
        "task_result": summary,
        "artifact_manifest": artifact_manifest,
    }


__all__ = ["reevaluate_atomic_variant"]
