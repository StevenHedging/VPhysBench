#!/usr/bin/env python3
"""Generate reviewed SAM2 repair candidates; canonical installation is opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from physbench.evaluation.common.masks.sam2 import Sam2VideoSegmenter
from physbench.reference_observations import (
    EntityObservation,
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import (
    load_curation_cases,
    render_anchor_sheet,
    render_dense_event_sheet,
)
from physbench.reference_observations.curation.anchors import (
    AnchorCandidate,
    build_independent_anchor_candidates,
)
from physbench.reference_observations.curation.install import install_candidate_bundle
from physbench.reference_observations.curation.overrides import validate_override
from physbench.reference_observations.curation.tracking import CuratedSam2Tracker


def shard_case_ids(case_ids: Iterable[str], *, shard_count: int) -> tuple[tuple[str, ...], ...]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, case_id in enumerate(sorted(set(case_ids))):
        shards[index % shard_count].append(case_id)
    return tuple(tuple(shard) for shard in shards)


def read_case_id_list(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != len(set(values)):
        raise ValueError("case list contains a duplicate Case")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def require_install_authorization(
    ledger_path: str | Path,
    *,
    case_id: str,
    candidate_manifest: str | Path,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"install requires one accepted review decision for {case_id}")
    row = matches[0]
    if row.get("anchor_decision") != "pass" or row.get("tube_decision") != "pass":
        raise ValueError(f"install requires an accepted review decision for {case_id}")
    observed = _sha256(Path(candidate_manifest))
    if row.get("candidate_digest") != observed:
        raise ValueError(f"accepted candidate digest is stale for {case_id}")
    return row


def _load_anchor(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["masks"])
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    return (masks > 0).astype(np.uint8)


def _decode_video(
    path: Path,
    *,
    source_frame_indices: Iterable[int],
) -> list[np.ndarray]:
    requested = tuple(int(value) for value in source_frame_indices)
    if not requested or requested[0] < 0 or any(
        right <= left for left, right in zip(requested, requested[1:])
    ):
        raise ValueError("timeline source frame indices must be strictly increasing")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open reference video: {path}")
    frames: list[np.ndarray] = []
    requested_set = set(requested)
    last = requested[-1]
    try:
        index = 0
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            if index in requested_set:
                frames.append(frame)
            if index >= last:
                break
            index += 1
    finally:
        capture.release()
    if len(frames) != len(requested):
        raise ValueError(
            f"reference video does not cover timeline frames: requested "
            f"{len(requested)}, decoded {len(frames)}"
        )
    return frames


def _existing_anchors(case: Any) -> tuple[AnchorCandidate, ...]:
    return tuple(
        AnchorCandidate.from_mask(entity.identity.object_id, _load_anchor(entity.anchor_npz_path))
        for entity in case.entities
    )


def _trajectory(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = masks.shape[0]
    centroid = np.full((count, 2), np.nan, np.float32)
    bbox = np.full((count, 4), np.nan, np.float32)
    area = masks.reshape(count, -1).sum(axis=1, dtype=np.int64)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            centroid[index] = (float(xs.mean()), float(ys.mean()))
            bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    return centroid, bbox, area


def rebuild_case(
    case: Any,
    *,
    candidate_root: Path,
    config: dict[str, Any],
    anchor_source: str,
    override_root: Path | None,
    segmenter: Sam2VideoSegmenter | None = None,
) -> Path:
    frames = _decode_video(
        case.reference_video_path,
        source_frame_indices=(
            sample["source_frame_index"] for sample in case.timeline["samples"]
        ),
    )
    existing = _existing_anchors(case)
    if anchor_source == "independent":
        spaced = [frames[index] for index in np.linspace(0, len(frames) - 1, 5, dtype=int)]
        anchors = build_independent_anchor_candidates(
            spaced,
            scene_id=case.scene_id,
            expected_count=len(existing),
            comparison_masks={item.object_id: item.mask for item in existing},
        )
    elif anchor_source == "existing":
        anchors = existing
    else:
        raise ValueError(f"unknown anchor source: {anchor_source}")
    override = None
    if override_root is not None:
        path = override_root / f"{case.case_id}.json"
        if path.exists():
            override = validate_override(json.loads(path.read_text(encoding="utf-8")))
    tracker = CuratedSam2Tracker(segmenter or Sam2VideoSegmenter(config))
    result = tracker.track(
        frames,
        anchors=anchors,
        corrections=() if override is None else override.corrections,
        lifecycle=() if override is None else override.lifecycle,
    )
    output = candidate_root / case.scene_id / case.case_id
    output.mkdir(parents=True, exist_ok=True)
    for index, anchor in enumerate(anchors, 1):
        masks = result.masks_by_object[anchor.object_id].astype(np.uint8)
        centroid, bbox, area = _trajectory(masks)
        write_entity_observation(
            output / "entities" / anchor.object_id,
            EntityObservation(
                object_id=anchor.object_id,
                mask_id=f"{index:02d}",
                masks=masks,
                centroid_xy=centroid,
                bbox_xyxy=bbox,
                area_pixels=area,
                state=result.states_by_object[anchor.object_id].astype(np.uint8),
            ),
        )
    evidence = render_anchor_sheet(
        frames[0],
        existing_masks={item.object_id: item.mask for item in existing},
        candidate_masks={item.object_id: item.mask for item in anchors},
        metadata={"case_id": case.case_id, "scene_id": case.scene_id},
    )
    cv2.imwrite(str(output / "anchor.png"), evidence)
    sample_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
        )
    )
    contact = render_dense_event_sheet(
        frames,
        masks_by_object=result.masks_by_object,
        states_by_object=result.states_by_object,
        observation_indices=sample_indices,
        source_indices=tuple(
            int(case.timeline["samples"][index]["source_frame_index"])
            for index in sample_indices
        ),
        columns=4,
    )
    if not cv2.imwrite(str(output / "contact_sheet.png"), contact):
        raise RuntimeError(f"cannot write candidate contact sheet for {case.case_id}")
    existing_observation = load_reference_observation(
        case.asset_root,
        case.observation_manifest_path,
        bundle_root=case.asset_root,
    )
    comparison: dict[str, Any] = {}
    for anchor in anchors:
        candidate_masks = result.masks_by_object[anchor.object_id].astype(bool)
        existing_masks = existing_observation.entities[anchor.object_id].masks.astype(bool)
        intersection = np.logical_and(candidate_masks, existing_masks).reshape(
            len(frames), -1
        ).sum(axis=1)
        union = np.logical_or(candidate_masks, existing_masks).reshape(
            len(frames), -1
        ).sum(axis=1)
        iou = np.divide(
            intersection,
            union,
            out=np.ones(len(frames), dtype=np.float64),
            where=union > 0,
        )
        comparison[anchor.object_id] = {
            "minimum_iou": float(iou.min()),
            "median_iou": float(np.median(iou)),
            "maximum_iou": float(iou.max()),
            "minimum_iou_observation_index": int(np.argmin(iou)),
        }
    metadata_path = output / "candidate_tracking.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "model_id": config["model_id"],
                "anchor_source": anchor_source,
                "seed_frame_by_observation": result.seed_frame_by_observation.tolist(),
                "propagation": result.propagation_metadata,
                "existing_comparison": comparison,
                "evidence": {
                    "anchor": "anchor.png",
                    "contact_sheet": "contact_sheet.png",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/reference_observations/audit_v1.json"))
    parser.add_argument("--override-root", type=Path, default=Path("configs/reference_observations/overrides"))
    parser.add_argument("--anchor-source", choices=("independent", "existing"), default="independent")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    requested = list(arguments.case_id)
    if arguments.case_list is not None:
        requested.extend(read_case_id_list(arguments.case_list))
    if len(requested) != len(set(requested)):
        raise ValueError("requested cases contain a duplicate Case")
    cases = load_curation_cases(arguments.dataset, case_ids=requested or None)
    selected_ids = set(
        shard_case_ids((case.case_id for case in cases), shard_count=arguments.shard_count)[
            arguments.shard_index
        ]
    )
    selected = [case for case in cases if case.case_id in selected_ids]
    segmenter = Sam2VideoSegmenter(config)
    failures: list[dict[str, str]] = []
    for case in selected:
        try:
            metadata = rebuild_case(
                case,
                candidate_root=arguments.candidate_root,
                config=config,
                anchor_source=arguments.anchor_source,
                override_root=arguments.override_root,
                segmenter=segmenter,
            )
            if arguments.install:
                bundle_manifest = metadata.with_name("candidate_bundle.json")
                if arguments.ledger is None or not bundle_manifest.exists():
                    raise ValueError(
                        "install requires a complete candidate bundle and accepted review decision"
                    )
                require_install_authorization(
                    arguments.ledger,
                    case_id=case.case_id,
                    candidate_manifest=bundle_manifest,
                )
                install_candidate_bundle(
                    bundle_manifest, case.observation_manifest_path.parent.parent
                )
        except Exception as exc:
            if not arguments.continue_on_error:
                raise
            failures.append(
                {"case_id": case.case_id, "error_type": type(exc).__name__, "message": str(exc)}
            )
    failure_path = arguments.candidate_root / f"failures-shard-{arguments.shard_index}.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures),
        encoding="utf-8",
    )
    print(json.dumps({"selected_cases": len(selected), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
