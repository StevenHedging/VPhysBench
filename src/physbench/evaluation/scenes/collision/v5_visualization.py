from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file, write_json
from ...common.entities.observer import OpenWorldObservation
from .nbody import NBodyCollisionState


def _artifact_directory(
    request: Any,
    *,
    config: Mapping[str, Any],
) -> tuple[Path, str]:
    environment_name = str(
        config.get(
            "external_root_env",
            "PHYSBENCH_VISUALIZATION_ROOT",
        )
    )
    root_value = os.environ.get(environment_name) or config["external_root"]
    root = Path(root_value).expanduser()
    namespace = str(config.get("namespace", "scene_default_v5"))
    identity = hashlib.sha256(
        str(request.artifact_dir.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    relative = (
        Path(namespace)
        / request.case["case_id"]
        / f"{request.job['job_id']}-{identity}"
    )
    return root / relative, relative.as_posix()


def _identity_color(value: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    hue = int(digest[0]) * 179 // 255
    hsv = np.asarray([[[hue, 210, 245]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _blend_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        return frame
    layer = np.zeros_like(frame)
    layer[binary] = color
    output = cv2.addWeighted(frame, 1.0, layer, 0.34, 0.0)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def _label(
    frame: np.ndarray,
    text: str,
    xy: Sequence[float],
    color: tuple[int, int, int],
) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x + 7, max(y - 7, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )


def _header(
    frame: np.ndarray,
    text: str,
    *,
    secondary: str | None = None,
) -> None:
    height = 58 if secondary else 36
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if secondary:
        cv2.putText(
            frame,
            secondary,
            (12, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )


def _reference_panel(
    frame: np.ndarray,
    *,
    frame_index: int,
    entity_ids: Sequence[str],
    masks: Sequence[Sequence[np.ndarray]],
    xy: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    for entity_index, entity_id in enumerate(entity_ids):
        color = _identity_color(f"reference:{entity_id}")
        output = _blend_mask(
            output,
            masks[entity_index][frame_index],
            color,
        )
        history_valid = valid[: frame_index + 1, entity_index]
        history = xy[: frame_index + 1, entity_index][history_valid]
        if len(history) >= 2:
            cv2.polylines(
                output,
                [np.rint(history).astype(np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        if valid[frame_index, entity_index]:
            _label(
                output,
                f"GT:{entity_id}",
                xy[frame_index, entity_index],
                color,
            )
    _header(
        output,
        f"REFERENCE | frame={frame_index} | N={len(entity_ids)}",
    )
    return output


def _prediction_detections(
    observation: OpenWorldObservation,
) -> dict[int, list[tuple[str, str, float, Any]]]:
    output: dict[int, list[tuple[str, str, float, Any]]] = {}
    for track in observation.tracks:
        for detection in track.detections:
            output.setdefault(detection.frame_index, []).append(
                (
                    track.track_id,
                    track.evidence_tier.value,
                    track.formal_exposure_weight,
                    detection,
                )
            )
    return output


def _prediction_panel(
    frame: np.ndarray,
    *,
    frame_index: int,
    detections: Mapping[
        int,
        Sequence[tuple[str, str, float, Any]],
    ],
    audit: Mapping[str, Any],
    available: bool,
) -> np.ndarray:
    output = frame.copy()
    match_by_track = {
        str(value["prediction_track_id"]): str(value["entity_id"])
        for value in audit["matches"]
    }
    residual = {str(value) for value in audit["residual_track_ids"]}
    ambiguous = {
        str(value)
        for value in audit.get("ambiguous_candidate_track_ids", ())
    }
    for (
        track_id,
        evidence_tier,
        exposure_weight,
        detection,
    ) in detections.get(frame_index, ()):
        color = _identity_color(f"prediction:{track_id}")
        if detection.mask is not None:
            output = _blend_mask(output, detection.mask, color)
        relation = match_by_track.get(track_id)
        suffix = (
            f"->{relation}"
            if relation is not None
            else (
                " EXTRA"
                if track_id in residual
                else (
                    " CANDIDATE"
                    if track_id in ambiguous
                    else ""
                )
            )
        )
        _label(
            output,
            (
                f"{track_id}[{evidence_tier}:{exposure_weight:g}]"
                f"{suffix}"
            ),
            detection.xy,
            color,
        )
    observed_count = float(
        audit.get(
            "formal_prediction_cardinality",
            len(audit["matches"])
            + len(audit["residual_track_ids"])
            + float(audit["overflow_count"]),
        )
    )
    media = "" if available else " | MEDIA UNAVAILABLE"
    _header(
        output,
        f"PREDICTION | frame={frame_index} | N={observed_count:g}{media}",
        secondary=(
            f"missing={len(audit['missing_entity_ids'])} "
            f"extra={len(audit['residual_track_ids'])} "
            f"candidate={len(ambiguous)} "
            f"overflow={float(audit['overflow_count']):g}"
        ),
    )
    return output


def _overlap_panel(
    reference_frame: np.ndarray,
    *,
    frame_index: int,
    reference_union: np.ndarray,
    prediction_union: np.ndarray,
    union_iou: float | None,
) -> np.ndarray:
    gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    output = (0.38 * cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)).astype(
        np.uint8
    )
    reference = np.asarray(reference_union) > 0
    prediction = np.asarray(prediction_union) > 0
    output[reference & ~prediction] = (255, 80, 60)
    output[prediction & ~reference] = (40, 70, 255)
    output[reference & prediction] = (40, 230, 80)
    score = "n/a" if union_iou is None else f"{union_iou:.3f}"
    _header(
        output,
        f"SUBJECT MASK AUDIT | frame={frame_index} | IoU={score}",
        secondary=(
            "blue=GT only | red=prediction only | green=intersection"
        ),
    )
    return output


def _active_contact_pairs(
    state: NBodyCollisionState | None,
    frame_index: int,
) -> list[str] | None:
    if state is None:
        return None
    pairs = {
        "|".join(event.entity_pair)
        for event in state.contact_events
        if event.start_frame <= frame_index <= event.end_frame
    }
    return sorted(pairs)


def _dashboard(
    *,
    width: int,
    height: int,
    frame_index: int,
    time_s: float,
    audit: Mapping[str, Any],
    reference_state: NBodyCollisionState,
    prediction_state: NBodyCollisionState | None,
) -> np.ndarray:
    output = np.full((height, width, 3), 22, dtype=np.uint8)
    matches = list(audit["matches"])
    position = float(
        audit.get(
            "position_diagnostic_score",
            (
                np.mean(
                    [float(value["position_score"]) for value in matches]
                )
                if matches
                else 0.0
            ),
        )
    )
    rejected = list(audit.get("rejected_candidate_matches", ()))
    reference_contacts = _active_contact_pairs(
        reference_state,
        frame_index,
    )
    prediction_contacts = _active_contact_pairs(
        prediction_state,
        frame_index,
    )
    lines = [
        "OPEN-WORLD N-BODY AUDIT",
        f"time={time_s:.3f}s  frame={frame_index}",
        (
            f"matched={len(matches)}  "
            f"missing={len(audit['missing_entity_ids'])}  "
            f"extra={len(audit['residual_track_ids'])}  "
            f"null-rejected={len(rejected)}"
        ),
        f"continuous position similarity={position:.3f}",
        (
            "missing IDs: "
            + (", ".join(audit["missing_entity_ids"]) or "none")
        ),
        (
            "extra tracks: "
            + (", ".join(audit["residual_track_ids"]) or "none")
        ),
        (
            "ambiguous candidates: "
            + (
                ", ".join(
                    audit.get("ambiguous_candidate_track_ids", ())
                )
                or "none"
            )
        ),
        (
            "GT contacts: "
            + (", ".join(reference_contacts or ()) or "none")
        ),
        (
            "prediction contacts: "
            + (
                "unavailable"
                if prediction_contacts is None
                else (", ".join(prediction_contacts) or "none")
            )
        ),
    ]
    y = 36
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66 if index == 0 else 0.5,
            (255, 255, 255) if index == 0 else (210, 210, 210),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 34
    return output


def _write_video(
    path: Path,
    *,
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    reference_masks: Sequence[Sequence[np.ndarray]],
    reference_xy: np.ndarray,
    reference_valid: np.ndarray,
    entity_ids: Sequence[str],
    prediction_observation: OpenWorldObservation,
    comparison: Sequence[Mapping[str, Any]],
    reference_union: Sequence[np.ndarray],
    prediction_union: Sequence[np.ndarray],
    union_ious: Sequence[float | None],
    reference_state: NBodyCollisionState,
    prediction_state: NBodyCollisionState | None,
    prediction_available: Sequence[bool],
    config: Mapping[str, Any],
) -> None:
    panel_width = int(config.get("panel_width", 640))
    panel_height = int(config.get("panel_height", 360))
    fps = float(config.get("fps", 16.0))
    detections = _prediction_detections(prediction_observation)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * panel_width, 2 * panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create v5 collision visualization: {path}")
    try:
        for frame_index, time_s in enumerate(times_s):
            audit = comparison[frame_index]
            panels = [
                _reference_panel(
                    reference_frames[frame_index],
                    frame_index=frame_index,
                    entity_ids=entity_ids,
                    masks=reference_masks,
                    xy=reference_xy,
                    valid=reference_valid,
                ),
                _prediction_panel(
                    prediction_frames[frame_index],
                    frame_index=frame_index,
                    detections=detections,
                    audit=audit,
                    available=bool(prediction_available[frame_index]),
                ),
                _overlap_panel(
                    reference_frames[frame_index],
                    frame_index=frame_index,
                    reference_union=reference_union[frame_index],
                    prediction_union=prediction_union[frame_index],
                    union_iou=union_ious[frame_index],
                ),
                _dashboard(
                    width=reference_frames[frame_index].shape[1],
                    height=reference_frames[frame_index].shape[0],
                    frame_index=frame_index,
                    time_s=float(time_s),
                    audit=audit,
                    reference_state=reference_state,
                    prediction_state=prediction_state,
                ),
            ]
            normalized = [
                cv2.resize(value, (panel_width, panel_height))
                for value in panels
            ]
            writer.write(
                np.vstack(
                    (
                        np.hstack(normalized[:2]),
                        np.hstack(normalized[2:]),
                    )
                )
            )
    finally:
        writer.release()


def write_collision_v5_visualization(
    request: Any,
    *,
    config: Mapping[str, Any],
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    reference_masks: Sequence[Sequence[np.ndarray]],
    reference_xy: np.ndarray,
    reference_valid: np.ndarray,
    entity_ids: Sequence[str],
    prediction_observation: OpenWorldObservation,
    comparison: Sequence[Mapping[str, Any]],
    reference_union: Sequence[np.ndarray],
    prediction_union: Sequence[np.ndarray],
    union_ious: Sequence[float | None],
    reference_state: NBodyCollisionState,
    prediction_state: NBodyCollisionState | None,
    prediction_available: Sequence[bool],
) -> dict[str, Any]:
    """Write an arbitrary-cardinality diagnostic without changing the score."""

    local_manifest = (
        request.artifact_dir / "collision_v5_visualization_manifest.json"
    )
    if not config.get("enabled", True):
        manifest = {
            "schema_version": "1.0",
            "status": "disabled",
            "storage_policy": "external_unsealed_diagnostic",
        }
        write_json(local_manifest, manifest)
        return {
            "collision_v5_visualization_manifest": str(local_manifest)
        }
    try:
        directory, relative = _artifact_directory(request, config=config)
        directory.mkdir(parents=True, exist_ok=True)
        video_path = directory / "open_world_nbody_audit.mp4"
        audit_path = directory / "open_world_nbody_audit.json"
        _write_video(
            video_path,
            times_s=times_s,
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            reference_masks=reference_masks,
            reference_xy=reference_xy,
            reference_valid=reference_valid,
            entity_ids=entity_ids,
            prediction_observation=prediction_observation,
            comparison=comparison,
            reference_union=reference_union,
            prediction_union=prediction_union,
            union_ious=union_ious,
            reference_state=reference_state,
            prediction_state=prediction_state,
            prediction_available=prediction_available,
            config=config,
        )
        write_json(
            audit_path,
            {
                "schema_version": "1.0",
                "case_id": request.case["case_id"],
                "job_id": request.job["job_id"],
                "entity_ids": list(entity_ids),
                "reference_body_count": len(reference_state.entity_ids),
                "prediction_state_available": prediction_state is not None,
                "prediction_body_count": (
                    len(prediction_state.entity_ids)
                    if prediction_state is not None
                    else None
                ),
                "reference_contact_events": [
                    {
                        "entity_pair": list(event.entity_pair),
                        "start_frame": event.start_frame,
                        "event_frame": event.frame_index,
                        "end_frame": event.end_frame,
                        "time_s": event.time_s,
                    }
                    for event in reference_state.contact_events
                ],
                "prediction_contact_events": (
                    [
                        {
                            "entity_pair": list(event.entity_pair),
                            "start_frame": event.start_frame,
                            "event_frame": event.frame_index,
                            "end_frame": event.end_frame,
                            "time_s": event.time_s,
                        }
                        for event in prediction_state.contact_events
                    ]
                    if prediction_state is not None
                    else []
                ),
                "prediction_track_evidence": [
                    {
                        "track_id": track.track_id,
                        "formal_tier": track.evidence_tier.value,
                        "formal_exposure_weight": (
                            track.formal_exposure_weight
                        ),
                        "confirmed": track.confirmed,
                        "detection_count": len(track.detections),
                        "frames": [
                            detection.frame_index
                            for detection in track.detections
                        ],
                        "sources": sorted(
                            {
                                source
                                for detection in track.detections
                                for source in detection.sources
                            }
                        ),
                    }
                    for track in prediction_observation.tracks
                ],
                "per_frame": list(comparison),
            },
        )
        repository_link = str(
            config.get("repository_link", "visualizations")
        )
        paths = {
            "observation_video": video_path,
            "audit_json": audit_path,
        }
        files = {
            name: {
                "path": str(path.resolve()),
                "repository_path": (
                    Path(repository_link) / relative / path.name
                ).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        }
        manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "evaluator_config_sha256": canonical_sha256(
                request.evaluator_config
            ),
            "storage_policy": (
                "external_unsealed_diagnostic_with_local_hashed_manifest"
            ),
            "external_directory": str(directory.resolve()),
            "repository_directory": (
                Path(repository_link) / relative
            ).as_posix(),
            "files": files,
        }
    except Exception as exc:
        manifest = {
            "schema_version": "1.0",
            "status": "failed",
            "storage_policy": "best_effort_never_changes_case_score",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    write_json(local_manifest, manifest)
    return {
        "collision_v5_visualization_manifest": str(local_manifest),
        "collision_v5_visualization": manifest,
    }


__all__ = ["write_collision_v5_visualization"]
