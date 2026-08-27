#!/usr/bin/env python3
"""Build staged SAM 3.1 GT candidates for sharded collision Cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Sequence

import cv2

from physbench.datasets.loader import load_dataset
from physbench.reference_observations.curation import load_curation_cases
from physbench.reference_observations.curation.bundle import (
    validate_candidate_bundle,
)
from physbench.reference_observations.curation.sam31_bundle import (
    write_sam31_candidate_bundle,
)
from physbench.reference_observations.curation.sam31_predictor import (
    Sam31GtPredictor,
)
from physbench.reference_observations.curation.sam31_rebuild import (
    Sam31GtConfig,
    rebuild_collision_case,
)


def shard_case_ids(
    case_ids: Iterable[str],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    if shard_count <= 0:
        raise ValueError("SAM3.1 GT shard count must be positive")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, case_id in enumerate(sorted(set(case_ids))):
        shards[index % shard_count].append(case_id)
    return tuple(tuple(shard) for shard in shards)


def select_collision_cases(
    cases: Sequence[Any],
    *,
    requested_ids: set[str] | None,
) -> tuple[Any, ...]:
    by_id = {str(case.case_id): case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("SAM3.1 GT Case catalog contains duplicate IDs")
    if requested_ids is not None:
        unknown = requested_ids - set(by_id)
        if unknown:
            raise ValueError(f"unknown requested Cases: {sorted(unknown)}")
        invalid = sorted(
            case_id
            for case_id in requested_ids
            if by_id[case_id].scene_id != "collision_1d"
        )
        if invalid:
            raise ValueError(f"requested Cases are not collision_1d: {invalid}")
    return tuple(
        case
        for case in cases
        if case.scene_id == "collision_1d"
        and (requested_ids is None or case.case_id in requested_ids)
    )


def candidate_is_complete(path: str | Path, *, case_id: str) -> bool:
    try:
        bundle = validate_candidate_bundle(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return bundle.case_id == case_id


def decode_timeline_frames(
    path: str | Path,
    *,
    source_frame_indices: Iterable[int],
) -> tuple[Any, ...]:
    requested = tuple(int(value) for value in source_frame_indices)
    if not requested or requested[0] < 0 or any(
        right < left for left, right in zip(requested, requested[1:])
    ):
        raise ValueError("SAM3.1 GT source frame indices must be nondecreasing")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open SAM3.1 GT reference video: {path}")
    requested_set = set(requested)
    decoded: dict[int, Any] = {}
    try:
        frame_index = 0
        while frame_index <= requested[-1]:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in requested_set:
                decoded[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(requested_set - set(decoded))
    if missing:
        raise ValueError(
            f"SAM3.1 GT video misses requested source frames: {missing[:8]}"
        )
    return tuple(decoded[index].copy() for index in requested)


def _read_case_ids(path: Path) -> set[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError("SAM3.1 GT Case list contains a duplicate")
    return set(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("SAM3.1 GT shard index lies outside shard count")
    config_value = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config_value, dict) or config_value.get("schema_version") != "1.0":
        raise ValueError("SAM3.1 GT config must use schema_version 1.0")
    model_config = config_value.get("model")
    curation_config = config_value.get("curation")
    if not isinstance(model_config, dict) or not isinstance(curation_config, dict):
        raise ValueError("SAM3.1 GT config requires model and curation mappings")

    snapshot = load_dataset(arguments.dataset)
    all_catalog_ids = {
        str(case["case_id"]): str(case["scene_id"])
        for case in snapshot.cases
    }
    requested = (
        None if arguments.case_list is None else _read_case_ids(arguments.case_list)
    )
    if requested is not None:
        unknown = requested - set(all_catalog_ids)
        if unknown:
            raise ValueError(f"unknown requested Cases: {sorted(unknown)}")
        noncollision = sorted(
            case_id
            for case_id in requested
            if all_catalog_ids[case_id] != "collision_1d"
        )
        if noncollision:
            raise ValueError(f"requested Cases are not collision_1d: {noncollision}")
    collision_ids = {
        case_id
        for case_id, scene_id in all_catalog_ids.items()
        if scene_id == "collision_1d" and (requested is None or case_id in requested)
    }
    cases = load_curation_cases(arguments.dataset, case_ids=collision_ids)
    cases = select_collision_cases(cases, requested_ids=collision_ids)
    selected_ids = set(
        shard_case_ids((case.case_id for case in cases), arguments.shard_count)[
            arguments.shard_index
        ]
    )
    selected = tuple(case for case in cases if case.case_id in selected_ids)

    predictor = Sam31GtPredictor(model_config)
    curation = Sam31GtConfig.from_mapping(curation_config)
    ledger_path = (
        arguments.candidate_root
        / "ledgers"
        / f"shard-{arguments.shard_index:04d}.jsonl"
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for completed, case in enumerate(selected, start=1):
        case_started = time.perf_counter()
        bundle_path = (
            arguments.candidate_root
            / case.scene_id
            / case.case_id
            / "candidate_bundle.json"
        )
        if arguments.resume and candidate_is_complete(bundle_path, case_id=case.case_id):
            row = {
                "case_id": case.case_id,
                "status": "resumed",
                "candidate_bundle": str(bundle_path),
                "elapsed_seconds": 0.0,
            }
        else:
            try:
                frames = decode_timeline_frames(
                    case.reference_video_path,
                    source_frame_indices=(
                        item["source_frame_index"] for item in case.timeline["samples"]
                    ),
                )
                candidate = rebuild_collision_case(
                    case,
                    frames=frames,
                    predictor=predictor,
                    config=curation,
                )
                bundle_path = write_sam31_candidate_bundle(
                    case,
                    candidate,
                    frames=frames,
                    output_root=arguments.candidate_root,
                    config_fingerprint=_sha256(arguments.config),
                    code_revision=arguments.code_revision,
                )
                row = {
                    "case_id": case.case_id,
                    "status": "candidate_pass" if candidate.accepted else "qa_rejected",
                    "candidate_bundle": str(bundle_path),
                    "candidate_digest": _sha256(bundle_path),
                    "finding_codes": [item.code for item in candidate.findings],
                    "discovery_attempts": list(candidate.discovery_attempts),
                    "elapsed_seconds": time.perf_counter() - case_started,
                }
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": time.perf_counter() - case_started,
                }
        rows.append(row)
        _atomic_write_jsonl(ledger_path, rows)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "shard_index": arguments.shard_index,
                    "completed": completed,
                    "selected": len(selected),
                    "case_id": case.case_id,
                    "status": row["status"],
                    "cases_per_hour": completed / max(elapsed, 1e-9) * 3600,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "shard_index": arguments.shard_index,
        "shard_count": arguments.shard_count,
        "selected": len(selected),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = ledger_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
